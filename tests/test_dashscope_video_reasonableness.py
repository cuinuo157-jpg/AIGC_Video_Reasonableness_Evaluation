from src.mllm.dashscope_video_reasonableness import call_vlm


class _Resp:
    def __init__(self, status_code=200, message="", output=None):
        self.status_code = status_code
        self.message = message
        self.output = output or {}


class _Chunk:
    def __init__(self, text: str, status_code: int = 200):
        self.status_code = status_code
        self.output = {
            "choices": [{"message": {"content": [{"text": text}]}}],
        }


def test_call_vlm_auto_retry_with_incremental_output():
    calls: list[dict] = []

    class _MM:
        @staticmethod
        def call(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return _Resp(
                    status_code=400,
                    message="This model only supports incremental_output set to True.",
                )
            return [_Chunk("hello "), _Chunk("world")]

    text = call_vlm(
        model="qwen3-vl-8b-thinking",
        api_key="test",
        frame_paths=["/tmp/f1.jpg"],
        system_prompt="sys",
        user_text="user",
        stream=False,
        dashscope=object(),
        MultiModalConversation=_MM,
    )

    assert text == "hello world"
    assert len(calls) == 2
    assert calls[0]["incremental_output"] is False
    assert calls[0]["stream"] is False
    assert calls[1]["incremental_output"] is True
    assert calls[1]["stream"] is True
