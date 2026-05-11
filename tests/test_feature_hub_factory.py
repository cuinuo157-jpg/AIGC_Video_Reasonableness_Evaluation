from src.feature_hub.hub import VideoProcessingConfig, create_default_hub


def test_create_default_hub_has_extractors():
    hub = create_default_hub("test.mp4", device="cpu")
    features = hub.available_features()
    assert "optical_flow" in features
    assert "face_embedding" in features
    assert "depth" in features
    assert "keypoints" in features
    assert "video_frames" in features
    assert "camera_compensation" in features
    assert "tracking" in features
    assert "au_features" in features


def test_create_default_hub_keeps_video_config():
    cfg = VideoProcessingConfig(sample_stride=4, max_frames=20, max_side=720)
    hub = create_default_hub("test.mp4", device="cpu", video_config=cfg)
    assert hub.video_config == cfg
