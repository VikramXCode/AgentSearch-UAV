from v2.models.semantic_adapter import CLIPEngineAdapter

def test_clip():
    try:
        adapter = CLIPEngineAdapter()
        print("CLIPEngineAdapter instantiated successfully.")
    except Exception as e:
        print(f"Failed to instantiate: {e}")

if __name__ == "__main__":
    test_clip()
