from app.services.auto_strategy.genes.tool import ToolGene


class TestToolGene:
    def test_clone_deep_copies_params(self):
        gene = ToolGene(
            tool_name="specific_day_filter",
            params={"enabled": True, "skip_days": [1, 2]},
        )

        cloned = gene.clone()
        cloned.params["skip_days"].append(3)

        assert gene.params["skip_days"] == [1, 2]
        assert cloned.params["skip_days"] == [1, 2, 3]

    def test_from_dict_deep_copies_params(self):
        data = {
            "tool_name": "specific_day_filter",
            "enabled": True,
            "params": {"enabled": True, "skip_days": [1, 2]},
        }

        gene = ToolGene.from_dict(data)
        gene.params["skip_days"].append(3)

        assert data["params"]["skip_days"] == [1, 2]
        assert gene.params["skip_days"] == [1, 2, 3]
