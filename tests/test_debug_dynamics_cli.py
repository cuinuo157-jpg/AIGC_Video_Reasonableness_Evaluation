import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import scripts.debug_dynamics as debug_dynamics


def test_parse_args_supports_analysis_mode_and_mllm():
    args = debug_dynamics.parse_args(
        [
            "--input",
            "demo.mp4",
            "--analysis-mode",
            "motion",
            "--enable-mllm",
            "--mllm-provider",
            "dashscope",
            "--mllm-model",
            "qwen3-vl-8b-thinking",
        ]
    )

    assert args.input == "demo.mp4"
    assert args.analysis_mode == "motion"
    assert args.enable_mllm is True
    assert args.mllm_provider == "dashscope"
    assert args.mllm_model == "qwen3-vl-8b-thinking"


def test_build_mllm_client_disabled_returns_none():
    args = SimpleNamespace(
        enable_mllm=False,
        mllm_provider="dashscope",
        mllm_model="qwen3-vl-8b-thinking",
        mllm_api_key="x",
        mllm_base_url="",
        mllm_fps=2,
    )
    assert debug_dynamics.build_mllm_client(args) is None


def test_run_motion_logic_analysis_passes_enable_and_client():
    args = SimpleNamespace(device="cuda", enable_mllm=True)
    mllm_client = MagicMock()
    fake_result = SimpleNamespace(
        motion_logic_score=0.8, naturalness_score=1.0, naturalness_issues=[]
    )
    with patch("scripts.debug_dynamics._build_motion_hub", return_value=MagicMock()) as mock_hub:
        with patch("scripts.debug_dynamics.MotionLogicAnalyzer") as mock_analyzer_cls:
            mock_analyzer = MagicMock()
            mock_analyzer.analyze.return_value = fake_result
            mock_analyzer_cls.return_value = mock_analyzer

            out = debug_dynamics.run_motion_logic_analysis("demo.mp4", args, mllm_client)

    assert out.motion_logic_score == 0.8
    mock_hub.assert_called_once_with("demo.mp4", args)
    called_kwargs = mock_analyzer_cls.call_args.kwargs
    assert called_kwargs["mllm_client"] is mllm_client
    assert called_kwargs["config"].enable_mllm is True


def test_load_repo_dotenv_loads_missing_values(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DASHSCOPE_API_KEY=abc123\nDASHSCOPE_BASE_URL=\"https://dashscope.aliyuncs.com/api/v1\"\n",
        encoding="utf-8",
    )
    with patch.dict(os.environ, {}, clear=True):
        debug_dynamics._load_repo_dotenv(tmp_path)
        assert os.environ["DASHSCOPE_API_KEY"] == "abc123"
        assert os.environ["DASHSCOPE_BASE_URL"] == "https://dashscope.aliyuncs.com/api/v1"


def test_load_repo_dotenv_does_not_override_existing_env(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("DASHSCOPE_API_KEY=from_dotenv\n", encoding="utf-8")
    with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "from_env"}, clear=True):
        debug_dynamics._load_repo_dotenv(tmp_path)
        assert os.environ["DASHSCOPE_API_KEY"] == "from_env"


def test_save_motion_result_json_writes_mllm_payload(tmp_path):
    result = SimpleNamespace(
        motion_logic_score=0.7,
        dynamics_score=0.6,
        smoothness_score=0.8,
        naturalness_score=0.3,
        naturalness_issues=["物体瞬移"],
        naturalness_mllm_result={"is_reasonable": False, "issues": ["物体瞬移"]},
    )
    video_path = str(tmp_path / "demo.mp4")
    out_path = debug_dynamics.save_motion_result_json(video_path, result, out_root=tmp_path)

    assert out_path.is_file()
    content = out_path.read_text(encoding="utf-8")
    assert '"motion_logic_score": 0.7' in content
    assert '"naturalness_score": 0.3' in content
    assert '"is_reasonable": false' in content
