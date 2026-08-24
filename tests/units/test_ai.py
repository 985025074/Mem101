from memkernel.ai import DeepSeekAI


def test_ai():
    dep_cli = DeepSeekAI.get_client()
    ai_provider = DeepSeekAI()
    res = ai_provider.get_ai_response(
        dep_cli, "echo nihao.Dont echo other things", "xxx"
    )
    assert res == "nihao"
