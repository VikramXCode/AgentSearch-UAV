# **MAIN START**

# shortcut to run all frontend + backend terminals (way 1)

Terminal -> Run tasks -> Start All

------------------------------------------------------------------------------------------------------------------------------
# MANUAL (way 2)


# Terminal 1: Start the Backend API Server:



cd "c:\\Users\\hp\\OneDrive\\Documents\\Projects\\AgentSearch\\AgentSearch-UAV"

C:\\v\\agentsearch-venv\\Scripts\\Activate.ps1

python -m api.main




# Terminal 2: Start the Frontend UI:

cd "c:\\Users\\hp\\OneDrive\\Documents\\Projects\\AgentSearch\\AgentSearch-UAV"

C:\\v\\agentsearch-venv\\Scripts\\Activate.ps1

cd "c:\\Users\\hp\\OneDrive\\Documents\\Projects\\AgentSearch\\AgentSearch-UAV\\frontend"

npm run dev

-------------------------------------------------------------------------------------------------------


# Benchmark All 5 Detection Systems on 30 images
python scripts\evaluate_all_systems_benchmark.py --num-images 30

# Benchmark Optimized Pipeline on 30 images
python scripts\evaluate_optimized_pipeline.py --num-images 30




\-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



#### **GENERAL** 



# If you are new :O

cd "c:\\Users\\hp\\OneDrive\\Documents\\Projects\\AgentSearch\\AgentSearch-UAV"

C:\\v\\agentsearch-venv\\Scripts\\Activate.ps1



python -m pip install -r requirements.txt

python -m pip install pillow opencv-python sahi ultralytics

python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu







python -m models.inference

python -m workflows.graph









\-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------









\-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------







#### **AGENTUAV ONLY FILES NEED**





AgentSearch-UAV/

│

├── api/

│   ├── \_\_init\_\_.py

│   ├── main.py

│   └── uploads/

│

├── agents/

│   ├── \_\_init\_\_.py

│   ├── detection\_agent.py

│   ├── explanation\_agent.py

│   ├── knowledge\_agent.py

│   ├── query\_agent.py

│   ├── strategy\_agent.py

│   ├── tool\_agent.py

│   └── verification\_agent.py

│

├── workflows/

│   ├── \_\_init\_\_.py

│   ├── graph.py

│   └── state.py

│

├── models/

│   ├── \_\_init\_\_.py

│   ├── base\_detector.py

│   ├── clip\_engine.py

│   ├── color\_verifier.py

│   ├── detection\_config.py

│   ├── detection\_quality\_optimizer.py

│   ├── detector.py

│   ├── enhanced\_detector.py

│   ├── enhanced\_postprocessor.py

│   ├── optimized\_detection\_pipeline.py

│   ├── postprocessor.py

│   ├── sahi\_engine.py

│   ├── schemas.py

│   ├── small\_object\_detector.py

│   ├── super\_resolution.py

│   ├── video\_tracker.py

│   └── yolo\_world.py

│

├── metrics/

│   ├── \_\_init\_\_.py

│   └── evaluator.py

│

├── utils/

│   ├── \_\_init\_\_.py

│   ├── adaptive\_sahi.py

│   ├── class\_conflict\_resolver.py

│   ├── detection\_fusion.py

│   ├── false\_positive\_filter.py

│   ├── image\_saving.py

│   ├── improved\_soft\_nms.py

│   ├── model\_paths.py

│   ├── paths.py

│   ├── scale\_aware\_postprocessing.py

│   ├── search\_utils.py

│   ├── selective\_super\_resolution.py

│   └── visualizer.py

│

├── configs/

│   ├── class\_thresholds.json

│   └── detection\_config.json

│

├── memory/

│   ├── knowledge.json

│   ├── missions.json

│   └── search\_history.json

│

├── weights/

│   ├── best.pt

│   └── yolov8s-world.pt

│

├── sample\_images/

│   ├── cctv.jpg

│   ├── red car.jpg

│   ├── uav1.png

│   └── video\_test/

│       └── flight\_test.mp4

│

├── outputs/

│

├── frontend/

│   ├── index.html

│   ├── package.json

│   ├── package-lock.json

│   ├── vite.config.js

│   └── src/

│       ├── App.css

│       ├── App.jsx

│       ├── index.css

│       ├── main.jsx

│       └── components/

│           ├── AIExplanationPanel.jsx

│           ├── BenchmarkModal.jsx

│           ├── DetectedObjectList.jsx

│           ├── DetectionResults.jsx

│           ├── Header.jsx

│           ├── HistoryModal.jsx

│           ├── ImageComparison.jsx

│           ├── LivePipelineGraph.jsx

│           ├── PipelineToolCards.jsx

│           ├── TargetSearchPanel.jsx

│           ├── TechnicalProcessingLog.jsx

│           └── ZoomModal.jsx

│

├── README.md

└── requirements.txt





\------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

