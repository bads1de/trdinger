        # BaseGene from_dict メソッドのより詳細なテスト
        # method enumの変換を確認
        gene_dict = {
            "method": "fixed_percentage",
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.10,
            "enabled": False
        }

        gene_restored = TPSLGene.from_dict(gene_dict)

        # methodがstringからEnumに正しく変換されていることを確認
        assert isinstance(gene_restored.method, TPSLMethod)
        assert gene_restored.method == TPSLMethod.FIXED_PERCENTAGE
        assert gene_restored.stop_loss_pct == 0.05
        assert gene_restored.take_profit_pct == 0.10
        assert gene_restored.enabled == False