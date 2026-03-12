from src.feature_hub.hub import create_default_hub


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
