# AgentSearch-UAV

This project is a small UAV object-search demo built around a YOLO-World detector, a SAHI-based sliced detector, and a simple agent workflow.

## What this repo runs

- `models/inference.py` - interactive object detection demo.
- `models/test_sahi.py` - interactive SAHI detection demo.
- `workflows/graph.py` - main end-to-end multi-agent UAV detection pipeline.
- `tests/test_iou.py` - simple IoU sanity check.
- `tests/upscale.py` - image upscaling sample script.

## Important note for Windows

The project can run into Windows path-length issues when installing PyTorch in a deeply nested folder. The safest working setup is a short path like `C:\v\agentsearch-venv`.

## Setup

1. Open PowerShell.
2. Go to the project root.
3. Create a virtual environment:

```powershell
python -m venv C:\v\agentsearch-venv
```

4. Activate it:

```powershell
C:\v\agentsearch-venv\Scripts\Activate.ps1
```

5. Install the base requirements:

```powershell
python -m pip install -r requirements.txt
```

6. Install the runtime packages used by the code but not listed in `requirements.txt`:

```powershell
python -m pip install pillow opencv-python sahi ultralytics
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```



## Run the demos

Run all commands from the project root so the package imports like `from models...` and `from workflows...` resolve correctly.

### 1. Object detection demo

```powershell
python -m models.inference
```

It will ask for:

- image path
- target object

Example:

```powershell
X:\sample_images\cars.jpg
car
```

Output:

- prints model details and detections in the terminal
- saves an annotated image to `outputs/detection_result.jpg`

### 2. SAHI demo

```powershell
python -m models.test_sahi
```

It will ask for:

- image path
- target object

### 3. Main pipeline (query)

```powershell
python -m workflows.graph
```

It will ask for:

- search query
- image path
  
It then runs query parsing, knowledge loading, strategy selection, detection, optional verification, explanation, annotation saving, and prints the final AgentState JSON.

### 4. IoU sanity check

```powershell
python tests/test_iou.py
```

### 5. Image upscaling sample

```powershell
python tests/upscale.py
```

This reads `sample_images/cars.jpg` and writes `sample_images/upscaled/cars_4x.jpg`.

## Files the demos expect

- `weights/yolov8s-world.pt`
- `sample_images/cars.jpg`
- `memory/search_history.json`

## Troubleshooting

- If `python -m models.inference` says `No module named 'models'`, you are not running from the project root.
- If PyTorch installation fails on Windows, use the short environment path shown above.
- If the detector runs but finds no objects, try a different image or a different target label.
