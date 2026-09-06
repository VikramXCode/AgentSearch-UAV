from utils.search_utils import (
    canonicalize_target,
    extract_query_components,
    build_verification_prompts,
    normalize_text,
)


def test_canonicalizes_common_aliases():
    assert canonicalize_target("sedan") == "car"
    assert canonicalize_target("vehicle") == "car"
    assert canonicalize_target("van") == "car"
    assert canonicalize_target("motorbike") == "motorcycle"
    assert canonicalize_target("lorry") == "truck"


def test_extracts_target_color_size_and_quantity():
    components = extract_query_components("2 red cars")
    assert components.target == "car"
    assert components.original_target in {"cars", "car"}
    assert components.color == "red"
    assert components.quantity == "2"
    assert components.quantity_value == 2

    components = extract_query_components("three red trucks")
    assert components.target == "truck"
    assert components.quantity == "3"
    assert components.quantity_value == 3

    components = extract_query_components("small white van")
    assert components.target == "car"
    assert components.size == "small"
    assert components.color == "white"


def test_extracts_multi_word_targets():
    components = extract_query_components("red sedan")
    assert components.target == "car"
    assert components.original_target == "sedan"

    components = extract_query_components("cars parked by the road")
    assert components.target == "car"
    assert components.original_target in {"cars", "car"}


def test_extracts_red_car_queries():
    components = extract_query_components("red car")
    assert components.target == "car"
    assert components.color == "red"
    assert components.quantity == "all"

    components = extract_query_components("red cars")
    assert components.target == "car"
    assert components.color == "red"
    assert components.quantity == "all"

    components = extract_query_components("find a red car")
    assert components.target == "car"
    assert components.color == "red"
    assert components.quantity == "one"
    assert components.quantity_value == 1

    components = extract_query_components("crimson car")
    assert components.target == "car"
    assert components.color == "red"


def test_verification_prompts_include_semantic_variants():
    prompts = build_verification_prompts("car", {"color": "red", "size": "small"})
    assert any("red car" in prompt for prompt in prompts)
    assert any(prompt == "car" for prompt in prompts)
    assert len(prompts) >= 4


def test_normalize_text_removes_underscores_and_extra_spaces():
    assert normalize_text("  two   red   cars  ") == "two red cars"
