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
        
        # Determine target
        if "person" in text:
            query_spec.target = "person"
        elif "car" in text:
            query_spec.target = "car"
        elif "motorcycle" in text or "bike" in text:
            query_spec.target = "motorcycle"
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

        # Add the full descriptive text as a semantic constraint for CLIP verification
        # if there are multiple words (e.g. "person riding two wheeler" or "yellow bus")
        if len(text.split()) > 1:
            query_spec.constraints.append(
                QueryConstraint(constraint_type="attribute", value=text)
            )

        return query_spec
