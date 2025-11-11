import cv2
import numpy as np
import pandas as pd

# --- Video path ---
cap = cv2.VideoCapture(r"data\test4_index.MOV")
data = []
frame_idx = 0

# --- Parameters ---
min_area = 800
max_area = 8000
lower_green = np.array([30, 60, 40])
upper_green = np.array([90, 255, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Boost green channel for better visibility
    b, g, r = cv2.split(frame)
    g = cv2.addWeighted(g, 1, g, 0, 0)
    frame = cv2.merge((b, g, r))

    # Convert to HSV and threshold for green
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Clean up mask
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    # Filter centroids by area
    filtered_centers = []
    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            filtered_centers.append(centroids[i])
    filtered_centers = np.array(filtered_centers)

    # Proceed only if exactly 4 markers detected
    if len(filtered_centers) == 4:
        centers = sorted(filtered_centers, key=lambda c: c[0])  # MCP → TIP by X
        mcp, pip, dip, tip = np.array(centers)

        # Angle calculation
        def angle(a, b, c):
            ab = a - b
            bc = c - b
            cosang = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
            return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

        pip_angle = angle(mcp, pip, dip)
        dip_angle = angle(pip, dip, tip)

        data.append([frame_idx, *mcp, *pip, *dip, *tip, pip_angle, dip_angle])

        # Draw detected markers
        for c in [mcp, pip, dip, tip]:
            cv2.circle(frame, tuple(c.astype(int)), 5, (0, 255, 0), -1)

    # Show original frame with markers
    cv2.imshow("Detection", frame)
    # Show mask
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

# Save CSV
df = pd.DataFrame(
    data,
    columns=[
        "frame",
        "mcp_x","mcp_y",
        "pip_x","pip_y",
        "dip_x","dip_y",
        "tip_x","tip_y",
        "pip_angle","dip_angle"
    ]
)
df.to_csv("finger_angles_green.csv", index=False)
print("✅ Saved -> finger_angles_green.csv")
