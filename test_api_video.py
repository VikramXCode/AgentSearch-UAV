import requests
import time
import sys

files = {'video': open('synthetic_test.mp4', 'rb')}
data = {'query': 'car'}
res = requests.post('http://127.0.0.1:5005/v2/detect-video', files=files, data=data)
print("POST Response:", res.json())
job_id = res.json().get('job_id')

for _ in range(30):
    time.sleep(1)
    progress = requests.get(f'http://127.0.0.1:5005/v2/video-progress/{job_id}')
    print("Progress:", progress.json())
    if progress.json().get('status') in ['completed', 'failed']:
        break
