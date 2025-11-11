import cv2
import numpy as np
import pandas as pd
import itertools

# --- Video path ---
cap = cv2.VideoCapture(r"data\test4_index.MOV")
data = []
frame_idx = 0

# --- Parameters ---
min_area = 800
max_area = 8000
lower_green = np.array([25, 40, 40])   # wide green range
upper_green = np.array([95, 255, 255])

phalange_lengths = None  # store initial lengths: L1, L2, L3

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Boost green channel
    b, g, r = cv2.split(frame)
    g = cv2.addWeighted(g, 1, g, 0, 0)
    frame = cv2.merge((b, g, r))

    # HSV mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    filtered = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            filtered.append(centroids[i])
    filtered = np.array(filtered)

    if len(filtered) == 4:
        # For first frame: assign MCP→TIP by X coordinate
        if phalange_lengths is None:
            sorted_idx = np.argsort(filtered[:,0])
            mcp, pip, dip, tip = filtered[sorted_idx]
            # compute initial phalange lengths
            L1 = np.linalg.norm(pip - mcp)
            L2 = np.linalg.norm(dip - pip)
            L3 = np.linalg.norm(tip - dip)
            phalange_lengths = [L1, L2, L3]
        else:
            # Assign points using phalange-length constraints
            best_score = np.inf
            best_perm = None
            for perm in itertools.permutations(filtered):
                m, p, d, t = perm
                L1c = np.linalg.norm(p - m)
                L2c = np.linalg.norm(d - p)
                L3c = np.linalg.norm(t - d)
                score = abs(L1c - phalange_lengths[0]) + abs(L2c - phalange_lengths[1]) + abs(L3c - phalange_lengths[2])
                if score < best_score:
                    best_score = score
                    best_perm = perm
            mcp, pip, dip, tip = np.array(best_perm)

        # Compute angles
        def angle(a, b, c):
            ab = a - b
            bc = c - b
            cosang = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
            return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

        pip_angle = angle(mcp, pip, dip)
        dip_angle = angle(pip, dip, tip)

        # Save frame data
        data.append([frame_idx, *mcp, *pip, *dip, *tip, pip_angle, dip_angle])

        # Draw markers
        for c in [mcp, pip, dip, tip]:
            cv2.circle(frame, tuple(c.astype(int)), 6, (0,255,0), -1)

    # Show frame and mask
    cv2.imshow("Detection", frame)
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
