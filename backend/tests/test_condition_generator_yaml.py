import sys
import os
import traceback
sys.path.append('./backend')
sys.path.append('./')

try:
    from app.services.auto_strategy.generators.condition_generator import ConditionGenerator
    print("Import successful - Using SmartConditionGenerator!")
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)

# Mock IndicatorGene for testing
class MockIndicatorGene:
    def __init__(self, type, enabled=True):
        self.type = type
        self.enabled = enabled

def test_yaml_conditions():
    print("Testing YAML-based condition generation...")

    gen = SmartConditionGenerator()

    # Test with some common indicators
    indicators = [
        MockIndicatorGene('RSI'),
        MockIndicatorGene('SMA'),
        MockIndicatorGene('MACD'),
        MockIndicatorGene('STOCH'),
        MockIndicatorGene('BBANDS'),
        MockIndicatorGene('CDL_HAMMER')
    ]

    # Test different threshold profiles
    profiles = ['normal', 'aggressive', 'conservative']

    for profile in profiles:
        print(f"\n--- Testing profile: {profile} ---")
        try:
            gen.set_context(threshold_profile=profile)
            result = gen.apply_threshold_context(indicators)

            print(f"Generated {len(result.get('long_conditions', []))} long conditions")
            print(f"Generated {len(result.get('short_conditions', []))} short conditions")

            # Show first condition for each side
            if result.get('long_conditions'):
                first = result['long_conditions'][0]
                print(f"First long condition: {first.left_operand} {first.operator} {first.right_operand}")

            if result.get('short_conditions'):
                first = result['short_conditions'][0]
                print(f"First short condition: {first.left_operand} {first.operator} {first.right_operand}")

        except Exception as e:
            print(f"Error with profile {profile}: {e}")
            print(traceback.format_exc())

    print("\n--- Test completed successfully! ---")

if __name__ == '__main__':
    test_yaml_conditions()