import numpy as np
from PIL import Image
from models.color_verifier import ColorVerifier


def test_color_verifier_detects_red_crop():
    # Create a solid red image crop
    red_img = Image.new("RGB", (100, 100), color=(220, 20, 20))
    is_match, conf, details = ColorVerifier.verify_color(red_img, "red", target_class="car")
    assert is_match is True
    assert conf > 0.6
    assert details["hsv_pixel_ratio"] > 0.5


def test_color_verifier_rejects_non_red_crop():
    # Create a solid white image crop
    white_img = Image.new("RGB", (100, 100), color=(240, 240, 240))
    is_match, conf, details = ColorVerifier.verify_color(white_img, "red", target_class="car")
    assert is_match is False
    assert details["hsv_pixel_ratio"] < 0.05

    # Create a solid blue image crop
    blue_img = Image.new("RGB", (100, 100), color=(20, 20, 220))
    is_match, conf, details = ColorVerifier.verify_color(blue_img, "red", target_class="car")
    assert is_match is False


def test_color_verifier_detects_blue_crop():
    blue_img = Image.new("RGB", (100, 100), color=(20, 50, 220))
    is_match, conf, details = ColorVerifier.verify_color(blue_img, "blue", target_class="car")
    assert is_match is True
    assert details["hsv_pixel_ratio"] > 0.5


def test_color_verifier_no_color_returns_true():
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    is_match, conf, details = ColorVerifier.verify_color(img, "", target_class="car")
    assert is_match is True


def test_color_verifier_batch_mixed_crops():
    red_crop = Image.new("RGB", (80, 80), color=(230, 25, 25))
    white_crop = Image.new("RGB", (80, 80), color=(250, 250, 250))
    blue_crop = Image.new("RGB", (80, 80), color=(15, 30, 220))
    black_crop = Image.new("RGB", (80, 80), color=(10, 10, 10))

    crops = [red_crop, white_crop, blue_crop, black_crop]
    results = ColorVerifier.verify_color_batch(crops, "red", target_class="car")

    assert len(results) == 4
    # Red should be matched
    assert results[0][0] is True
    assert results[0][2]["stage"] in {"hsv_prefilter_confirmed", "clip_ai_verified"}
    # White, Blue, Black should be rejected in HSV pre-filter
    assert results[1][0] is False
    assert results[1][2]["stage"] == "hsv_prefilter_rejected"
    assert results[2][0] is False
    assert results[2][2]["stage"] == "hsv_prefilter_rejected"
    assert results[3][0] is False
    assert results[3][2]["stage"] == "hsv_prefilter_rejected"

