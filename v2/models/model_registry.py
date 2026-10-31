from typing import Dict, Any, Optional, List
from v2.models.adapters import SpecialistDetectorAdapter, OpenWorldDetectorAdapter
import torch
import gc

class ModelRegistry:
    """
    Lazy loader and memory manager for V2 models.
    Ensures mutually exclusive loading where needed to protect VRAM.
    """
    
    def __init__(self):
        self._loaded_models: Dict[str, Any] = {}
        # Known registered logical models
        self.REGISTERED_MODELS = {
            "specialist_p2": {
                "type": "specialist", 
                "adapter": SpecialistDetectorAdapter,
                "path": "runs/detect/runs/detect/experiments/model_search/EXP11_yolov8s_p2_1536/weights/best.pt"
            },
            "open_world_yolo_world": {
                "type": "open_world", 
                "adapter": OpenWorldDetectorAdapter,
                "path": "weights/yolov8s-world.pt"
            },
        }
        
    def get_adapter(self, logical_name: str, mock_instance: Any = None) -> Any:
        """
        Returns an instantiated adapter wrapping the loaded model.
        """
        if logical_name not in self.REGISTERED_MODELS:
            raise ValueError(f"Unknown model: {logical_name}")
            
        # If testing, allow injection of mock instances to avoid real loading
        if mock_instance is not None:
            adapter_cls = self.REGISTERED_MODELS[logical_name]["adapter"]
            return adapter_cls(model_instance=mock_instance)
            
        model_instance = self._load_model(logical_name)
        adapter_cls = self.REGISTERED_MODELS[logical_name]["adapter"]
        return adapter_cls(model_instance=model_instance)
        
    def _load_model(self, logical_name: str) -> Any:
        if logical_name in self._loaded_models:
            return self._loaded_models[logical_name]
            
        print(f"[ModelRegistry] Loading {logical_name} into VRAM...")
        
        # Enforce mutual exclusivity to prevent A100 OOM (P2 vs OpenWorld)
        if logical_name == "specialist_p2" and "open_world_yolo_world" in self._loaded_models:
            self._unload_model("open_world_yolo_world")
        elif logical_name == "open_world_yolo_world" and "specialist_p2" in self._loaded_models:
            self._unload_model("specialist_p2")
            
        # Real model loading
        from ultralytics import YOLO
        path = self.REGISTERED_MODELS[logical_name]["path"]
        instance = YOLO(path)
        instance.to("cuda")
            
        self._loaded_models[logical_name] = instance
        return instance
        
    def _unload_model(self, logical_name: str):
        if logical_name in self._loaded_models:
            print(f"[ModelRegistry] Unloading {logical_name} from VRAM...")
            # Real unload: del self._loaded_models[logical_name] and torch.cuda.empty_cache()
            del self._loaded_models[logical_name]
            gc.collect()
            torch.cuda.empty_cache()
            
    def get_loaded_models(self) -> List[str]:
        return list(self._loaded_models.keys())
