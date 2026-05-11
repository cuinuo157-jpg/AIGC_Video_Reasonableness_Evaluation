from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Py-Feat 模型缓存统一存放到 third_party/pyfeat
_PYFEAT_CACHE = str(
    Path(__file__).resolve().parent.parent.parent / "third_party" / "pyfeat"
)


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


class AUExtractor:
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

        # 底层 API 链式调用: detect_faces → detect_landmarks → detect_aus
        faces = det.detect_faces(frame)
        if not faces or not faces[0]:
            return {}

        landmarks = det.detect_landmarks(frame, faces)
        if not landmarks or len(landmarks[0]) == 0:
            return {}

        aus = det.detect_aus(frame, landmarks)
        if aus is None or len(aus) == 0:
            return {}

        # aus 结构: list/ndarray — 取第一帧第一张脸的 AU 值
        au_arr = np.array(aus)
        au_values = au_arr[0][0] if au_arr.ndim == 3 else au_arr[0]
        au_names = det.info["au_presence_columns"]
        return {name: float(val) for name, val in zip(au_names, au_values)}

    def extract_sequence(
        self, frames: list[np.ndarray]
    ) -> list[dict[str, float]]:
        return [self.extract(f) for f in frames]
