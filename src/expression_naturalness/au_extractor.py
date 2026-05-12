from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Py-Feat 模型缓存统一存放到 third_party/pyfeat
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PYFEAT_CACHE = str(_PROJECT_ROOT / "third_party" / "pyfeat")
_PYFEAT_BRIDGE_SCRIPT = _PROJECT_ROOT / "scripts" / "run_pyfeat_au_bridge.py"
_DEFAULT_AU_BACKEND = "subprocess"
_DEFAULT_AU_PYTHON = r"D:\ProgramData\Anaconda3\envs\pyfeat\python.exe"
_DEFAULT_AU_TIMEOUT_SEC = 1800


def _patch_pyfeat_cache() -> None:
    """将 Py-Feat 模型缓存目录重定向到 third_party/pyfeat。"""
    os.makedirs(_PYFEAT_CACHE, exist_ok=True)
    import feat.utils.io as _fio

    get_resource_path = lambda: _PYFEAT_CACHE
    _fio.get_resource_path = get_resource_path

    # feat.pretrained 在模块导入时可能已经绑定了 get_resource_path，
    # 仅 patch feat.utils.io 不足以覆盖下载/查找逻辑。
    try:
        import feat.pretrained as _fpretrained
    except Exception:
        return

    _fpretrained.get_resource_path = get_resource_path


def _patch_scipy_compat() -> None:
    """兼容新版本 SciPy 移除 integrate.simps 的情况。"""
    try:
        import scipy.integrate as _integrate
    except Exception:
        return

    if hasattr(_integrate, "simps") or not hasattr(_integrate, "simpson"):
        return

    def _simps(y, x=None, dx=1.0, axis=-1, even=None):
        kwargs = {"dx": dx, "axis": axis}
        if x is not None:
            kwargs["x"] = x
        return _integrate.simpson(y, **kwargs)

    _integrate.simps = _simps


def _resolve_au_backend(backend: str | None = None) -> str:
    text = (backend or os.environ.get("AIGC_AU_BACKEND", _DEFAULT_AU_BACKEND)).strip().lower()
    if text not in {"local", "subprocess"}:
        raise ValueError(f"Unsupported AU backend: {text}")
    return text


def _resolve_external_python(external_python: str | None = None) -> str | None:
    text = (
        external_python
        or os.environ.get("AIGC_AU_PYTHON", "")
        or _DEFAULT_AU_PYTHON
    ).strip()
    return text or None


def _resolve_timeout(timeout_sec: int | None = None) -> int:
    if timeout_sec is not None:
        return max(1, int(timeout_sec))
    raw = os.environ.get("AIGC_AU_TIMEOUT_SEC", str(_DEFAULT_AU_TIMEOUT_SEC)).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_AU_TIMEOUT_SEC


class _LocalAUExtractor:
    def __init__(self) -> None:
        self._detector = None

    def _ensure_detector(self) -> None:
        if self._detector is None:
            _patch_scipy_compat()
            _patch_pyfeat_cache()
            from feat import Detector

            self._detector = Detector(au_model="xgb")

    def extract(self, frame: np.ndarray) -> dict[str, float]:
        """从单帧图像提取 AU 强度，接受 BGR/RGB numpy 数组。"""
        self._ensure_detector()
        det = self._detector

        faces = det.detect_faces(frame)
        if not faces or not faces[0]:
            return {}

        landmarks = det.detect_landmarks(frame, faces)
        if not landmarks or len(landmarks[0]) == 0:
            return {}

        aus = det.detect_aus(frame, landmarks)
        if aus is None or len(aus) == 0:
            return {}

        au_arr = np.array(aus)
        au_values = au_arr[0][0] if au_arr.ndim == 3 else au_arr[0]
        au_names = det.info["au_presence_columns"]
        return {name: float(val) for name, val in zip(au_names, au_values)}

    def extract_sequence(self, frames: list[np.ndarray]) -> list[dict[str, float]]:
        return [self.extract(f) for f in frames]


def run_local_au_sequence(frames: list[np.ndarray]) -> list[dict[str, float]]:
    """始终使用当前 Python 环境本地执行 AU 提取。"""
    return _LocalAUExtractor().extract_sequence(frames)


class AUExtractor:
    """AU 提取器。

    支持两种后端：
      - local: 在当前 Python 环境直接加载 py-feat
      - subprocess: 使用外部 Python/Conda 环境子进程执行 py-feat

    外部环境通过环境变量指定：
      - AIGC_AU_BACKEND=subprocess
      - AIGC_AU_PYTHON=/path/to/conda/env/python
    """

    def __init__(
        self,
        backend: str | None = None,
        external_python: str | None = None,
        timeout_sec: int | None = None,
    ) -> None:
        self.backend = _resolve_au_backend(backend)
        self.external_python = _resolve_external_python(external_python)
        self.timeout_sec = _resolve_timeout(timeout_sec)
        self._local_extractor = _LocalAUExtractor() if self.backend == "local" else None
        logger.info(
            "AU extractor configured: backend=%s, external_python=%s, timeout_sec=%s",
            self.backend,
            self.external_python or "-",
            self.timeout_sec,
        )

    @property
    def _detector(self) -> Any:
        if self._local_extractor is None:
            return None
        return self._local_extractor._detector

    @_detector.setter
    def _detector(self, value: Any) -> None:
        if self._local_extractor is None:
            self._local_extractor = _LocalAUExtractor()
        self._local_extractor._detector = value

    def _ensure_detector(self) -> None:
        if self.backend != "local":
            raise RuntimeError("Detector is only available in local AU backend")
        assert self._local_extractor is not None
        self._local_extractor._ensure_detector()

    def _serialize_frames(self, frames: list[np.ndarray], path: Path) -> None:
        try:
            payload = np.stack(frames, axis=0)
            np.save(path, payload, allow_pickle=False)
        except ValueError:
            payload = np.array(frames, dtype=object)
            np.save(path, payload, allow_pickle=True)

    def _extract_sequence_subprocess(self, frames: list[np.ndarray]) -> list[dict[str, float]]:
        if not self.external_python:
            raise RuntimeError(
                "AU backend is 'subprocess' but external python is not configured. "
                "Set AIGC_AU_PYTHON to the py-feat environment python executable."
            )
        if not _PYFEAT_BRIDGE_SCRIPT.exists():
            raise FileNotFoundError(f"Py-Feat bridge script not found: {_PYFEAT_BRIDGE_SCRIPT}")

        with tempfile.TemporaryDirectory(prefix="aigc_pyfeat_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            frames_path = tmpdir_path / "frames.npy"
            output_path = tmpdir_path / "aus.json"
            self._serialize_frames(frames, frames_path)

            env = os.environ.copy()
            env["AIGC_AU_BACKEND"] = "local"

            command = [
                self.external_python,
                os.fspath(_PYFEAT_BRIDGE_SCRIPT),
                "--frames-npy",
                os.fspath(frames_path),
                "--output-json",
                os.fspath(output_path),
            ]
            logger.info("Running py-feat AU extraction via external python: %s", self.external_python)
            completed = subprocess.run(
                command,
                cwd=os.fspath(_PROJECT_ROOT),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                check=False,
            )
            if completed.returncode != 0:
                stderr = completed.stderr.strip()
                stdout = completed.stdout.strip()
                details = stderr or stdout or f"exit code {completed.returncode}"
                raise RuntimeError(f"External py-feat extraction failed: {details}")
            if not output_path.exists():
                raise RuntimeError("External py-feat extraction completed without output file")

            data = json.loads(output_path.read_text(encoding="utf-8"))
            return [
                {str(name): float(value) for name, value in frame_aus.items()}
                for frame_aus in data
            ]

    def extract(self, frame: np.ndarray) -> dict[str, float]:
        return self.extract_sequence([frame])[0] if frame is not None else {}

    def extract_sequence(self, frames: list[np.ndarray]) -> list[dict[str, float]]:
        if not frames:
            return []
        if self.backend == "subprocess":
            return self._extract_sequence_subprocess(frames)
        assert self._local_extractor is not None
        return self._local_extractor.extract_sequence(frames)
