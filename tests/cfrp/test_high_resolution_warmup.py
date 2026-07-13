import numpy as np

from vlnce_server.cfrp.warmup_audit import _validate_frames


def test_frame_audit_accepts_the_high_resolution_visual_contract(tmp_path):
    frame = tmp_path / "frame.npy"
    np.save(frame, np.zeros((480, 640, 3), dtype=np.uint8))

    _validate_frames((str(frame),), 1, (640, 480))


def test_frame_audit_rejects_the_legacy_128_pixel_frames(tmp_path):
    frame = tmp_path / "frame.npy"
    np.save(frame, np.zeros((128, 128, 3), dtype=np.uint8))

    try:
        _validate_frames((str(frame),), 1, (640, 480))
    except ValueError as exc:
        assert "does not match visual contract" in str(exc)
    else:
        raise AssertionError("expected legacy frame to be rejected")
