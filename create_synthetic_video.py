import cv2
import numpy as np

# Load a real image from VisDrone
real_img = cv2.imread('datasets/VisDrone2019-YOLO/val/images/0000333_02745_d_0000015.jpg')
if real_img is None:
    print("Could not load real image")
    exit(1)

# Just crop a patch that might contain something (e.g. center)
h, w = real_img.shape[:2]
car_patch = real_img[h//2:h//2+100, w//2:w//2+150]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('synthetic_real_video.mp4', fourcc, 5.0, (640, 480))

for i in range(15):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Target disappearance/reappearance logic:
    # Frame 0-4: Target visible
    # Frame 5-9: Target missing
    # Frame 10-14: Target visible again
    
    if i < 5 or i >= 10:
        # Paste the real patch into the frame
        x_offset = 100 + i * 10
        y_offset = 200
        
        # Keep within bounds
        ph, pw = car_patch.shape[:2]
        if y_offset+ph <= 480 and x_offset+pw <= 640:
            frame[y_offset:y_offset+ph, x_offset:x_offset+pw] = car_patch
            
    out.write(frame)

out.release()
print("Created synthetic_real_video.mp4")
