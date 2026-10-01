from typing import Optional
from v2.schemas.state import QuerySpec, QueryConstraint

class QueryAgentV2:
    """
    Deterministic parser for Phase 2.
    Converts text and reference image inputs into a generic QuerySpec.
    """
    
    def run(self, raw_query: str, reference_image_path: Optional[str] = None) -> QuerySpec:
        query_spec = QuerySpec(
            raw_query=raw_query or "",
            reference_image_path=reference_image_path
        )
        
        if not raw_query:
            if reference_image_path:
                query_spec.target = "reference object"
            return query_spec

        # Very basic deterministic parsing for demonstration in Phase 2
        text = raw_query.lower().strip()
        
        # Determine target with comprehensive synonyms
        target_mapping = {
            "person": ["person", "people", "pedestrian", "man", "woman", "child", "boy", "girl", "human", "guy"],
            "car": ["car", "cars", "automobile", "sedan", "suv", "taxi", "jeep", "auto"],
            "vehicle": ["vehicle", "vehicles"],
            "truck": ["truck", "trucks", "pickup", "lorry"],
            "bus": ["bus", "buses", "coach", "minibus"],
            "van": ["van", "vans", "minivan"],
            "motorcycle": ["motorcycle", "motorcycles", "bike", "motorbike", "motor"],
            "bicycle": ["bicycle", "bicycles", "cyclist", "cycle"],
            "boat": ["boat", "ship", "vessel", "watercraft"],
            "airplane": ["airplane", "plane", "aircraft", "jet"]
        }
        
        found_target = None
        for standard_target, synonyms in target_mapping.items():
            if any(syn == text or syn in text.split() for syn in synonyms):
                found_target = standard_target
                break
                
        if found_target:
            query_spec.target = found_target
        else:
            # Fallback to the last word as a naive target if no reference image
            words = text.split()
            if words:
                query_spec.target = words[-1]
            else:
                query_spec.target = "object"

        # Preserve other constraints
        # E.g. "red", "blue", "white", "large", "small" as attributes
        attributes = ["red", "blue", "white", "black", "orange", "yellow", "green", "pink", "purple", "brown", "grey", "gray", "silver", "large", "small", "tiny"]
        for attr in attributes:
            if attr in text:
                query_spec.constraints.append(
                    QueryConstraint(constraint_type="attribute", value=attr)
                )

        # E.g. "wearing an orange shirt"
        if "wearing" in text:
            parts = text.split("wearing")
            if len(parts) > 1:
                clothing = parts[1].split("near")[0].strip()
                if clothing:
                    query_spec.constraints.append(
                        QueryConstraint(constraint_type="attribute", value=f"wearing {clothing}")
                    )

        # Spatial / relational e.g. "near"
        if "near" in text:
            parts = text.split("near")
            if len(parts) > 1:
                obj = parts[1].strip()
                if obj:
                    query_spec.constraints.append(
                        QueryConstraint(
                            constraint_type="relation",
                            value="near",
                            metadata={"object": obj}
                        )
                    )

        # Add descriptive phrase constraint only if there is additional descriptive context
        # not already captured by extracted attributes or simple action prefixes (e.g. 'find a person')
        cleaned_text = text
        for prefix in ["find a ", "find an ", "find the ", "find ", "locate a ", "locate an ", "locate the ", "locate ", "detect a ", "detect an ", "detect the ", "detect "]:
            if cleaned_text.startswith(prefix):
                cleaned_text = cleaned_text[len(prefix):].strip()
                break
        if cleaned_text.startswith("a ") or cleaned_text.startswith("an ") or cleaned_text.startswith("the "):
            cleaned_text = cleaned_text.split(" ", 1)[1].strip()

        if cleaned_text and cleaned_text != query_spec.target and not query_spec.constraints and len(cleaned_text.split()) > 1:
            query_spec.constraints.append(
                QueryConstraint(constraint_type="attribute", value=cleaned_text)
            )

        return query_spec
