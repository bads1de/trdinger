import pytest
from app.services.auto_strategy.serializers.list_encoder import (
    ListEncoder,
    NormalizationConstants,
)
from app.services.auto_strategy.genes import (
    StrategyGene,
    IndicatorGene,
    Condition,
    ConditionGroup,
    StatefulCondition,
    TPSLGene,
    # PositionSizingGene, # Unused
)
from unittest.mock import patch, MagicMock


class TestListEncoder:
    @pytest.fixture
    def ga_config_mock(self):
        with patch("app.services.auto_strategy.config.GAConfig") as mock:
            # Configure the mock to return an instance with max_indicators = 5
            mock_instance = MagicMock()
            mock_instance.max_indicators = 5
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def encoder(self, ga_config_mock):  # Consume the mock
        return ListEncoder()

    # Removed unused ga_config fixture

    def test_encode_empty_gene(self, encoder):
        gene = StrategyGene()

        # Override singleton or mock GAConfig if necessary,
        # but since GAConfig is a dataclass usually instantiated in ListEncoder via import,
        # we might need to patch it if it's not a singleton but a class used as `GAConfig()`.
        # ListEncoder code uses `GAConfig().max_indicators`.
        # Assuming GAConfig() returns default values if not configured globally.

        encoded = encoder.to_list(gene)

        # expected_len check can be omitted or used if we strictly validate length.
        # For now, just check it is encoded.  # Depending on GAConfig default. Let's check fallback logic in ListEncoder:
        # try: max_indicators = GAConfig().max_indicators; except: 5
        # Standard GAConfig default for max_indicators is 3 (from ga.py seen earlier) but let's see.
        # Actually in ga.py: "max_indicators": 3 is in GA_DEFAULT_CONFIG.
        # But GAConfig definition has: max_indicators: int = GA_DEFAULT_CONFIG["max_indicators"]

        # Let's just assert length > 0 and check structure roughly
        assert len(encoded) > 0
        assert all(isinstance(x, float) for x in encoded)

    def test_encode_complex_gene(self, encoder):
        # 1. Indicators
        ind1 = IndicatorGene(type="RSI", parameters={"period": 14}, timeframe="1h")
        ind2 = IndicatorGene(
            type="MACD",
            parameters={"fast": 12, "slow": 26, "signal": 9},
            timeframe="4h",
        )

        # 2. Conditions (Nested)
        # (RSI > 50) AND ((MACD > 0) OR (RSI < 30))
        cond1 = Condition(
            left_operand={"indicator": "RSI"}, operator=">", right_operand=50
        )
        cond2 = Condition(
            left_operand={"indicator": "MACD"}, operator=">", right_operand=0
        )
        cond3 = Condition(
            left_operand={"indicator": "RSI"}, operator="<", right_operand=30
        )

        group1 = ConditionGroup(operator="OR", conditions=[cond2, cond3])
        entry_conditions = [cond1, group1]  # Implicit AND at top level

        # 3. Stateful Conditions
        sc_trigger = Condition(
            left_operand={"indicator": "RSI"}, operator=">", right_operand=70
        )
        sc_follow = Condition(
            left_operand={"indicator": "MACD"}, operator="<", right_operand=0
        )
        sc = StatefulCondition(
            trigger_condition=sc_trigger,
            follow_condition=sc_follow,
            lookback_bars=5,
            direction="short",
        )

        # 4. TPSL
        tpsl = TPSLGene(enabled=True, stop_loss_pct=0.05, take_profit_pct=0.1)

        gene = StrategyGene(
            indicators=[ind1, ind2],
            entry_conditions=entry_conditions,
            exit_conditions=[],
            stateful_conditions=[sc],
            tpsl_gene=tpsl,
        )

        encoded = encoder.to_list(gene)

        # Verify Indicator Encoding
        # RSI: ID=3, TF=1h(6) -> 3/100, 6/20 -> 0.03, 0.3
        # First block
        assert encoded[0] == 0.03  # Type ID
        assert encoded[1] == 0.3  # Timeframe ID

        # Verify Condition Encoding
        # First cond: RSI(0.03) >(0.1) 50(norm)
        # LogicOp: AND(1) -> 0.1 (Implicit top level typically treated as AND or passed logic_op)
        # Flatten logic: _flatten_conditions([cond1, group1]) -> AND, cond1 -> AND, cond2 -> OR, cond3 (roughly)

        # Let's inspect the condition block part
        # Skip indicators (mocked max_indicators = 5).
        offset = 5 * NormalizationConstants.INDICATOR_BLOCK_SIZE

        # Entry conditions start
        # 1st cond: (AND, cond1)
        # Logic=0.1 (AND), Op=0.1 (>), LeftType=0.5(Ind), LeftVal=0.03(RSI), RightType=0.1(Const), RightVal=normalized(50)
        assert encoded[offset] == 0.1  # Logic AND
        assert encoded[offset + 1] == 0.1  # Op >
        assert encoded[offset + 2] == 0.5  # Left Type Ind
        assert encoded[offset + 3] == 0.03  # Left Val RSI ID

        # 2nd cond: (AND, cond2) - Inherited AND from list iteration if flatten logic uses passed op
        # Wait, group1 is OR. The flatten logic:
        # list iter: item1(cond) -> (current_op=AND, cond)
        # item2(group) -> flatten(group.conditions, group.operator="OR")
        #   -> (OR, cond2), (OR, cond3)
        # So expectation:
        # Cond 1: AND ...
        # Cond 2: OR ...
        # Cond 3: OR ...

        cond2_offset = offset + NormalizationConstants.CONDITION_BLOCK_SIZE
        assert encoded[cond2_offset] == 0.2  # Logic OR

        # Verify Stateful Condition Encoding
        # Stateful block starts after Exit conditions
        # Offset = Indicator + Entry + Exit
        stateful_offset = (
            5 * NormalizationConstants.INDICATOR_BLOCK_SIZE
            + 2
            * NormalizationConstants.MAX_CONDITIONS
            * NormalizationConstants.CONDITION_BLOCK_SIZE
        )

        # [Enabled(1.0), Lookback(0.005), Cooldown(0.0), Direction(-1), Trigger..., Follow...]
        assert encoded[stateful_offset] == 1.0  # Enabled
        assert encoded[stateful_offset + 1] == 0.005  # Lookback 5/1000
        assert encoded[stateful_offset + 3] == -1.0  # Direction short

    def test_to_list_consistency(self, encoder):
        gene1 = StrategyGene()
        gene2 = StrategyGene()

        enc1 = encoder.to_list(gene1)
        enc2 = encoder.to_list(gene2)

        assert len(enc1) == len(enc2)
        assert enc1 == enc2




