import urllib.request
import urllib.parse
import json
import time

def test_api():
    print("Testing POST /detect-video...")
    data = urllib.parse.urlencode({
        'query': 'car',
        'video_name': '12 sec.mp4'
    }).encode('utf-8')
    req = urllib.request.Request('http://localhost:5005/detect-video', data=data)
    res = urllib.request.urlopen(req)
    start_res = json.loads(res.read().decode('utf-8'))
    print("Start response:", start_res)
    assert start_res.get("success") is True
    job_id = start_res["job_id"]
    print(f"Job ID: {job_id}")

    start_t = time.time()
    while time.time() - start_t < 120:
        time.sleep(1.0)
        prog_res = urllib.request.urlopen(f'http://localhost:5005/video-progress/{job_id}')
        status_data = json.loads(prog_res.read().decode('utf-8'))
        pct = status_data.get("progress", 0)
        cur = status_data.get("current_frame", 0)
        tot = status_data.get("total_frames", 0)
        st = status_data.get("status")
        stage = status_data.get("stage", "")
        fps = status_data.get("fps", 0)
        print(f"[{st.upper()}] {pct}% - Frame {cur}/{tot} @ {fps} FPS - {stage}")
        if st in ["completed", "failed"]:
            break

    assert status_data["status"] == "completed", f"Job failed: {status_data.get('error')}"
    print("\n--- JOB COMPLETED SUCCESSFULLY ---")
    result = status_data["result"]
    print(f"Target query     : {result['query']}")
    print(f"Tracked targets  : {result['count']}")
    print(f"Output video URL : {result['output_video']}")
    print(f"FPS              : {result['metrics']['fps']}")
    print(f"Total time       : {result['metrics']['total_processing_time']}s")
    print(f"Frames processed : {result['metrics']['frames_processed']}")
    print(f"Resolution       : {result['metrics']['input_resolution']}")

if __name__ == "__main__":
    test_api()
