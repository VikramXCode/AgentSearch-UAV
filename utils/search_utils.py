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
    "silver",
    "brown",
    "pink",
    "purple",
    "gold",
    "cyan",
    "magenta",
}

COLOR_ALIASES = {
    "red": "red",
    "crimson": "red",
    "scarlet": "red",
    "ruby": "red",
    "blue": "blue",
    "navy": "blue",
    "azure": "blue",
    "green": "green",
    "emerald": "green",
    "white": "white",
    "black": "black",
    "dark": "black",
    "yellow": "yellow",
    "orange": "orange",
    "gray": "gray",
    "grey": "gray",
    "silver": "silver",
    "brown": "brown",
    "pink": "pink",
    "purple": "purple",
    "gold": "gold",
}

SIZE_WORDS = {
    "small",
    "medium",
    "large",
    "big",
    "tiny",
    "little",
    "distant",
    "far",
    "blurry",
    "blurred",
}

SIZE_ALIASES = {
    "small": "small",
    "tiny": "small",
    "little": "small",
    "big": "large",
    "large": "large",
    "medium": "medium",
    "distant": "small",
    "far": "small",
    "blurry": "small",
    "blurred": "small",
}

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

QUANTITY_ALL_WORDS = {"all", "every", "each"}
QUANTITY_ONE_WORDS = {"one", "single", "only", "a", "an"}

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
    "of",
    "all",
    "every",
    "each",
    "one",
    "single",
    "only",
    "a",
    "an",
    "by",
    "in",
    "on",
    "at",
    "to",
    "with",
    "and",
    "or",
    "some",
    "many",
    "there",
    "near",
    "around",
    "color",
    "colors",
    "colored",
    "coloured",
    "type",
}

TARGET_ALIASES = {
    "car": "car",
    "cars": "car",
    "vehicle": "car",
    "vehicles": "car",
    "automobile": "car",
    "automobiles": "car",
    "sedan": "car",
    "sedans": "car",
    "van": "car",
    "vans": "car",
    "suv": "car",
    "suvs": "car",
    "jeep": "car",
    "jeeps": "car",
    "taxi": "car",
    "taxis": "car",
    "motorcycle": "motorcycle",
    "motorcycles": "motorcycle",
    "motorbike": "motorcycle",
    "motorbikes": "motorcycle",
    "motor": "motorcycle",
    "motors": "motorcycle",
    "bike": "motorcycle",
    "bikes": "motorcycle",
    "bicycle": "bicycle",
    "bicycles": "bicycle",
    "truck": "truck",
    "trucks": "truck",
    "lorry": "truck",
    "lorries": "truck",
    "person": "person",
    "people": "person",
    "pedestrian": "person",
    "pedestrians": "person",
    "bus": "bus",
    "buses": "bus",
    "tricycle": "tricycle",
    "tricycles": "tricycle",
    "awning-tricycle": "tricycle",
    "awning tricycle": "tricycle",
    "awning tricycles": "tricycle",
    "awning-tricycles": "tricycle",
    "boat": "boat",
    "boats": "boat",
    "bird": "bird",
    "birds": "bird",
    "cat": "cat",
    "cats": "cat",
    "dog": "dog",
    "dogs": "dog",
    "horse": "horse",
    "horses": "horse",
    "cow": "cow",
    "cows": "cow",
    "sheep": "sheep",
    "bottle": "bottle",
    "bottles": "bottle",
    "chair": "chair",
    "chairs": "chair",
    "bench": "bench",
    "benches": "bench",
    "backpack": "backpack",
    "backpacks": "backpack",
    "traffic light": "traffic light",
    "traffic lights": "traffic light",
    "stop sign": "stop sign",
    "stop signs": "stop sign",
}

KNOWN_TARGETS = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "tricycle",
    "van",
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

VISDRONE_CLASSES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

VISDRONE_TARGET_MAP = {
    "car": ["car", "van"],
    "vehicle": ["car", "van", "truck", "bus"],
    "person": ["pedestrian", "people"],
    "pedestrian": ["pedestrian"],
    "people": ["people"],
    "motorcycle": ["motor"],
    "motor": ["motor"],
    "bicycle": ["bicycle"],
    "truck": ["truck"],
    "bus": ["bus"],
    "van": ["van"],
    "tricycle": ["tricycle", "awning-tricycle"],
    "awning-tricycle": ["awning-tricycle"],
}


def get_visdrone_classes_for_target(target: str) -> list[str]:
    """Map a query target to matching VisDrone class names."""
    canonical = canonicalize_target(target)
    if canonical in VISDRONE_TARGET_MAP:
        return VISDRONE_TARGET_MAP[canonical]
    normalized = normalize_label(target)
    if normalized in VISDRONE_TARGET_MAP:
        return VISDRONE_TARGET_MAP[normalized]
    if normalized in VISDRONE_CLASSES:
        return [normalized]
    return []


@dataclass(frozen=True)
class QueryComponents:
    target: str
    original_target: str = ""
    color: str = ""
    size: str = ""
    quantity: str = "all"
    quantity_value: int | None = None
    canonical_query: str = ""


def normalize_text(text: str) -> str:
    normalized = text.strip().lower()
    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def normalize_label(label: str) -> str:
    return normalize_text(label)


def _normalize_word(token: str) -> str:
    token = normalize_text(token)
    return token


def _extract_quantity(tokens: list[str]) -> tuple[str, int | None]:
    numeric_match = re.search(r"\b(\d+)\b", " ".join(tokens))
    if numeric_match:
        return numeric_match.group(1), int(numeric_match.group(1))

    for token in tokens:
        if token in QUANTITY_ALL_WORDS:
            return "all", None
        if token in NUMBER_WORDS:
            return str(NUMBER_WORDS[token]), NUMBER_WORDS[token]

    if any(token in QUANTITY_ONE_WORDS for token in tokens):
        return "one", 1

    return "all", None


def extract_query_components(query: str) -> QueryComponents:
    normalized = normalize_text(query)
    tokens = normalized.split()

    color = ""
    for token in tokens:
        if token in COLOR_ALIASES:
            color = COLOR_ALIASES[token]
            break
        if token in COLOR_WORDS:
            color = token
            break

    size = next((SIZE_ALIASES[token] for token in tokens if token in SIZE_ALIASES), "")

    quantity, quantity_value = _extract_quantity(tokens)
    original_target = _extract_target(tokens)
    target = canonicalize_target(original_target) if original_target else ""

    canonical_query = _build_canonical_query(target, color, size, quantity)

    return QueryComponents(
        target=target,
        original_target=original_target,
        color=color,
        size=size,
        quantity=quantity,
        quantity_value=quantity_value,
        canonical_query=canonical_query,
    )


def canonicalize_target(target: str) -> str:
    normalized = normalize_label(target)
    if not normalized:
        return ""

    aliases = TARGET_ALIASES
    if normalized in aliases:
        return aliases[normalized]

    if normalized.endswith("s"):
        singular = normalized[:-1]
        if singular in aliases:
            return aliases[singular]

    lookup = normalized.replace(" ", " ")
    if lookup in aliases:
        return aliases[lookup]

    return normalized


def build_verification_prompts(target: str, attributes: dict[str, str]) -> list[str]:
    canonical_target = canonicalize_target(target)
    if not canonical_target:
        return []

    color = attributes.get("color", "").strip().lower()
    size = attributes.get("size", "").strip().lower()
    size = SIZE_ALIASES.get(size, size)

    prompts: list[str] = []

    if color and size:
        prompts.extend([
            f"{color} {size} {canonical_target}",
            f"a {color} {size} {canonical_target}",
            f"{color} {canonical_target}",
            f"a {color} {canonical_target}",
            f"{color} vehicle",
            canonical_target,
        ])
    elif color:
        prompts.extend([
            f"{color} {canonical_target}",
            f"a {color} {canonical_target}",
            f"{color} vehicle",
            f"a {color} automobile",
            canonical_target,
        ])
    elif size:
        prompts.extend([
            f"{size} {canonical_target}",
            f"a {size} {canonical_target}",
            f"a {canonical_target}",
            canonical_target,
        ])
    else:
        prompts.extend([
            canonical_target,
            f"a {canonical_target}",
            f"a vehicle",
        ])

    if canonical_target == "car":
        prompts.extend(["vehicle", "automobile", "sedan", "car"]) 
    elif canonical_target == "motorcycle":
        prompts.extend(["motorbike", "bike", "motorcycle"])
    elif canonical_target == "truck":
        prompts.extend(["lorry", "truck", "vehicle"]) 

    unique_prompts: list[str] = []
    seen: set[str] = set()
    for prompt in prompts:
        normalized_prompt = normalize_text(prompt)
        if normalized_prompt and normalized_prompt not in seen:
            unique_prompts.append(normalized_prompt)
            seen.add(normalized_prompt)

    return unique_prompts[:8]


def _build_canonical_query(target: str, color: str, size: str, quantity: str) -> str:
    canonical_parts = []
    if quantity not in {"all", ""}:
        canonical_parts.append(str(quantity))
    if color:
        canonical_parts.append(color)
    if size:
        canonical_parts.append(size)
    if target:
        canonical_parts.append(target)
    return " ".join(canonical_parts).strip()


def _extract_target(tokens: list[str]) -> str:
    phrases = []
    for index in range(len(tokens)):
        for span in range(len(tokens) - index, 0, -1):
            phrase = " ".join(tokens[index:index + span])
            normalized = normalize_text(phrase)
            if not normalized:
                continue
            if normalized in TARGET_ALIASES or normalized in KNOWN_TARGETS:
                phrases.append(normalized)
                break
        if phrases:
            break

    if not phrases:
        for token in reversed(tokens):
            normalized = normalize_text(token)
            if normalized and normalized not in STOP_WORDS:
                canonical = canonicalize_target(normalized)
                if canonical:
                    return normalized
        return ""

    return phrases[0]
