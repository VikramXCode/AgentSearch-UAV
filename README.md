# **MAIN START**

  # shortcut to run all frontend + backend terminals (way 1)

Terminal -> Run tasks -> Start All

--------*---------

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

--------*--------


# Benchmark All 5 Detection Systems on 30 images
python scripts\evaluate_all_systems_benchmark.py --num-images 30

# Benchmark Optimized Pipeline on 30 images
python scripts\evaluate_optimized_pipeline.py --num-images 30




--------*--------



#### **GENERAL** 



python -m pip install -r requirements.txt

python -m pip install pillow opencv-python sahi ultralytics

python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu







python -m models.inference

python -m workflows.graph














--------------------------------------------------

