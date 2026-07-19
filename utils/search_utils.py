import re
from dataclasses import dataclass


COLOR_WORDS = {
    "red",
    "blue",
    "green",
    "white",
    "black",
    "yellow",
    "orange",
    "gray",
    "grey",
    "brown",
}

SIZE_WORDS = {
    "small",
    "medium",
    "large",
    "big",
    "tiny",
    "little",
}

QUANTITY_ALL_WORDS = {
    "all",
    "every",
    "each",
}

QUANTITY_ONE_WORDS = {
    "one",
    "single",
    "only",
}

STOP_WORDS = {
    "find",
    "detect",
    "locate",
    "search",
    "show",
    "spot",
    "look",
    "for",
    "the",
    "a",
    "an",
    "of",
    "all",
    "every",
    "each",
    "one",
    "single",
    "only",
    *COLOR_WORDS,
    *SIZE_WORDS,
}

TARGET_ALIASES = {
    "bike": "motorcycle",
    "bikes": "motorcycle",
    "motorbike": "motorcycle",
    "motorbikes": "motorcycle",
    "bicycle": "bicycle",
    "bicycles": "bicycle",
    "car": "car",
    "cars": "car",
    "truck": "truck",
    "trucks": "truck",
    "lorry": "truck",
    "lorries": "truck",
    "cat": "cat",
    "cats": "cat",
    "dog": "dog",
    "dogs": "dog",
    "person": "person",
    "people": "person",
    "motorcycle": "motorcycle",
    "motorcycles": "motorcycle",
    "bus": "bus",
    "buses": "bus",
    "boat": "boat",
    "boats": "boat",
    "bird": "bird",
    "birds": "bird",
    "truck": "truck",
}

KNOWN_TARGETS = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "boat",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "bottle",
    "cup",
    "chair",
    "couch",
    "bed",
    "tv",
    "laptop",
    "cell phone",
    "backpack",
    "handbag",
    "suitcase",
    "skateboard",
    "surfboard",
    "traffic light",
    "stop sign",
    "bench",
}


@dataclass(frozen=True)
class QueryComponents:
    target: str
    color: str = ""
    size: str = ""
    quantity: str = "all"


def normalize_text(text: str) -> str:

    normalized = text.strip().lower()
    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def normalize_label(label: str) -> str:

    return normalize_text(label)


def extract_query_components(query: str) -> QueryComponents:

    normalized = normalize_text(query)
    tokens = normalized.split()

    color = next((token for token in tokens if token in COLOR_WORDS), "")
    size = next((token for token in tokens if token in SIZE_WORDS), "")

    if any(token in QUANTITY_ALL_WORDS for token in tokens):
        quantity = "all"
    elif any(token in QUANTITY_ONE_WORDS for token in tokens):
        quantity = "one"
    else:
        quantity = "all"

    target = _extract_target(tokens)

    return QueryComponents(
        target=target,
        color=color,
        size=size,
        quantity=quantity,
    )


def canonicalize_target(target: str) -> str:

    normalized = normalize_label(target)

    if normalized in TARGET_ALIASES:
        return TARGET_ALIASES[normalized]

    return normalized


def build_verification_prompts(target: str, attributes: dict[str, str]) -> list[str]:

    canonical_target = canonicalize_target(target)
    color = attributes.get("color", "").strip().lower()
    size = attributes.get("size", "").strip().lower()

    prompts = []

    # Start with the most specific prompt, then add broader fallbacks and
    # comparison options so CLIP can rank the requested object against alternates.
    if color and size:
        prompts.append(f"{color} {size} {canonical_target}")

    if color:
        prompts.append(f"{color} {canonical_target}")

    if size:
        prompts.append(f"{size} {canonical_target}")

    prompts.append(canonical_target)

    if color:
        for alternative_color in sorted(COLOR_WORDS):
            if alternative_color != color:
                prompts.append(f"{alternative_color} {canonical_target}")

    if size:
        for alternative_size in ("small", "medium", "large"):
            if alternative_size != size:
                prompts.append(f"{alternative_size} {canonical_target}")

    unique_prompts = []
    seen = set()

    for prompt in prompts:
        if prompt not in seen:
            seen.add(prompt)
            unique_prompts.append(prompt)

    return unique_prompts


def _extract_target(tokens: list[str]) -> str:

    candidates = [token for token in tokens if token not in STOP_WORDS]

    if not candidates:
        return ""

    for size in range(len(candidates), 0, -1):
        for index in range(0, len(candidates) - size + 1):
            phrase = " ".join(candidates[index:index + size])

            if phrase in TARGET_ALIASES:
                return TARGET_ALIASES[phrase]

            if phrase in KNOWN_TARGETS:
                return phrase

    return TARGET_ALIASES.get(candidates[-1], candidates[-1])
