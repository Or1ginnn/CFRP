import numpy as np

from vlnce_server.qwen3vl.vision import (
    HABITAT_RGB_HEIGHT,
    HABITAT_RGB_WIDTH,
    QWEN3_VL_MAX_PIXELS,
    prepare_qwen3vl_image,
    qwen3vl_image_size,
    qwen3vl_processor_kwargs,
)


def test_shared_visual_contract_preserves_a_four_by_three_camera():
    assert (HABITAT_RGB_WIDTH, HABITAT_RGB_HEIGHT) == (640, 480)
    assert qwen3vl_image_size() == (384, 288)
    assert QWEN3_VL_MAX_PIXELS == 110592
    assert qwen3vl_processor_kwargs()["max_pixels"] == QWEN3_VL_MAX_PIXELS


def test_in_memory_images_are_resized_for_qwen3vl():
    image = prepare_qwen3vl_image(np.zeros((480, 640, 3), dtype=np.uint8))

    assert image.size == (384, 288)


def test_portable_image_paths_are_not_rewritten():
    assert prepare_qwen3vl_image("/tmp/frame.png") == "/tmp/frame.png"
