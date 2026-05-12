import json
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock
from types import SimpleNamespace

from src.expression_naturalness.au_extractor import AUExtractor, _patch_scipy_compat


def test_au_extractor_returns_au_dict():
    extractor = AUExtractor()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    mock_det = MagicMock()
    mock_det.detect_faces.return_value = [[1, 2, 3, 4]]
    mock_det.detect_landmarks.return_value = [np.zeros((68, 2), dtype=np.float32)]
    mock_det.detect_aus.return_value = np.array([[0.5, 1.2, 0.3]])
    mock_det.info = {"au_presence_columns": ["AU01", "AU02", "AU04"]}
    extractor._detector = mock_det

    result = extractor.extract(frame)
    assert isinstance(result, dict)
    assert result["AU01"] == 0.5
    assert result["AU02"] == 1.2


def test_patch_scipy_compat_keeps_existing_simps():
    import scipy.integrate as integrate

    original = getattr(integrate, "simps", None)
    _patch_scipy_compat()
    if original is not None:
        assert integrate.simps is original
    else:
        assert hasattr(integrate, "simps")


def test_au_extractor_subprocess_backend_uses_external_python(monkeypatch, tmp_path: Path):
    frames = [
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.ones((8, 8, 3), dtype=np.uint8),
    ]

    recorded = {}

    def fake_run(command, cwd, env, capture_output, text, timeout, check):
        recorded["command"] = command
        recorded["cwd"] = cwd
        recorded["backend"] = env["AIGC_AU_BACKEND"]
        output_path = Path(command[-1])
        output_path.write_text(json.dumps([{"AU01": 0.3}, {"AU01": 0.7}]), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("src.expression_naturalness.au_extractor.subprocess.run", fake_run)

    extractor = AUExtractor(
        backend="subprocess",
        external_python="D:/envs/pyfeat/python.exe",
        timeout_sec=12,
    )
    result = extractor.extract_sequence(frames)

    assert result == [{"AU01": 0.3}, {"AU01": 0.7}]
    assert recorded["command"][0] == "D:/envs/pyfeat/python.exe"
    assert recorded["command"][1].endswith("run_pyfeat_au_bridge.py")
    assert recorded["cwd"]
    assert recorded["backend"] == "local"
