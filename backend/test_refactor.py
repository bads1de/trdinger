#!/usr/bin/env python3
"""
リファクタリング後のGeneSerializer動作確認スクリプト
"""

import sys
import os

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.auto_strategy.serializers import GeneSerializer, DictConverter, ListEncoder, ListDecoder, JsonConverter
from app.services.auto_strategy.models.strategy_models import StrategyGene, IndicatorGene

def test_refactored_serializer():
    """リファクタリングされたGeneSerializerのテスト"""

    print("INFO: GeneSerializer refactoring test starting...")

    # テスト用の戦略遺伝子を作成
    gene = StrategyGene(
        indicators=[
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
            IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
        ],
        entry_conditions=[],
        exit_conditions=[],
        long_entry_conditions=[],
        short_entry_conditions=[],
        risk_management={"position_size": 0.1},
        metadata={"test": True}
    )

    print("SUCCESS: Test strategy gene created")

    # GeneSerializerのテスト
    serializer = GeneSerializer(enable_smart_generation=False)

    # 辞書変換テスト
    dict_data = serializer.strategy_gene_to_dict(gene)
    print(f"SUCCESS: Dict conversion: {len(dict_data)} fields")

    # JSON変換テスト
    json_str = serializer.strategy_gene_to_json(gene)
    print(f"SUCCESS: JSON conversion: {len(json_str)} characters")

    # リスト変換テスト
    list_data = serializer.to_list(gene)
    print(f"SUCCESS: List conversion: {len(list_data)} elements")

    # 逆変換テスト
    reconstructed_gene = serializer.from_list(list_data, StrategyGene)
    print(f"SUCCESS: List reverse conversion: {reconstructed_gene.id}")

    # 個別コンポーネントのテスト
    dict_converter = DictConverter(enable_smart_generation=False)
    list_encoder = ListEncoder()
    list_decoder = ListDecoder(enable_smart_generation=False)
    json_converter = JsonConverter(dict_converter)

    # DictConverterテスト
    dict_result = dict_converter.strategy_gene_to_dict(gene)
    print(f"SUCCESS: DictConverter: {len(dict_result)} fields")

    # ListEncoderテスト
    encoded_list = list_encoder.to_list(gene)
    print(f"SUCCESS: ListEncoder: {len(encoded_list)} elements")

    # ListDecoderテスト
    decoded_gene = list_decoder.from_list(encoded_list, StrategyGene)
    print(f"SUCCESS: ListDecoder: {decoded_gene.id}")

    # JsonConverterテスト
    json_result = json_converter.strategy_gene_to_json(gene)
    print(f"SUCCESS: JsonConverter: {len(json_result)} characters")

    print("\nINFO: GeneSerializer refactoring test completed!")
    print("   All components are working correctly.")

    return True

if __name__ == "__main__":
    try:
        test_refactored_serializer()
        print("\nSUCCESS: Test passed: Refactoring works correctly")
    except Exception as e:
        print(f"\nERROR: Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)