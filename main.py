import cv2
import numpy as np
import pandas as pd

cap = cv2.VideoCapture("finger_white_markers.mp4")

data = []
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # threshold white dots
    _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    # small blur and morphological opening to clean noise
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))

    # find connected components (each white dot)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    centers = centroids[1:]  # skip background

    if len(centers) == 4:
        # sort from MCP→PIP→DIP→TIP (by x coordinate)
        centers = sorted(centers, key=lambda c: c[0])
        mcp, pip, dip, tip = np.array(centers)

        def angle(a, b, c):
            ab, bc = a - b, c - b
            cosang = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
            return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

        # compute angles
        pip_angle = angle(mcp, pip, dip)
        dip_angle = angle(pip, dip, tip)
        data.append([frame_idx, *mcp, *pip, *dip, *tip, pip_angle, dip_angle])

    frame_idx += 1

cap.release()

df = pd.DataFrame(
    data,
    columns=[
        "frame",
        "mcp_x", "mcp_y",
        "pip_x", "pip_y",
        "dip_x", "dip_y",
        "tip_x", "tip_y",
        "pip_angle", "dip_angle"
    ]
)

df.to_csv("finger_angles.csv", index=False)
print("Saved -> finger_angles.csv")
