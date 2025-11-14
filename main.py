import cv2
import numpy as np
import pandas as pd
import os

# --- Video path ---
input_path = r"data\november 13 test 1\test5.MOV"
cap = cv2.VideoCapture(input_path)
data = []
frame_idx = 0

annotated_dir = r"annotated"
os.makedirs(annotated_dir, exist_ok=True)
base_name = os.path.splitext(os.path.basename(input_path))[0]
output_video_path = os.path.join(annotated_dir, f"annotated_{base_name}.mp4")

# --- Setup video writer for annotated output ---
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# --- Detection parameters ---
min_area = 50  # reduced to catch smaller markers
max_area = 2000
lower_green = np.array([40, 30, 30])
upper_green = np.array([85, 255, 255])

# --- Phalange length ratios (soft constraint) ---
phalange_ratios = np.array([44.75, 26.31, 20.98], dtype=float)
phalange_ratios = phalange_ratios / phalange_ratios[0]  # [1, 0.54, 0.5]

prev_joints = None

def angle(a, b, c):
    ab = a - b
    bc = c - b
    cosang = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Green mask ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.medianBlur(mask,7)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    # --- Detect marker centers ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
                centers.append([cx, cy])
    centers = np.array(centers)

    # --- Assign joints ---
    if len(centers) >= 4:
        if prev_joints is None:
            # First frame: assign MCP → TIP robustly
            center = np.mean(centers, axis=0)
            mcp_idx = np.argmax(np.linalg.norm(centers - center, axis=1))
            mcp = centers[mcp_idx]
            remaining = np.delete(centers, mcp_idx, axis=0)

            # Sort remaining points by distance from MCP
            dists = np.linalg.norm(remaining - mcp, axis=1)
            sorted_remaining = remaining[np.argsort(dists)]
            pip, dip, tip = sorted_remaining[0], sorted_remaining[1], sorted_remaining[2]

            prev_joints = np.array([mcp, pip, dip, tip])
        else:
            # Temporal assignment: assign detected points to closest previous joint
            new_joints = []
            available = centers.copy()
            for prev_joint in prev_joints:
                idx = np.argmin(np.linalg.norm(available - prev_joint, axis=1))
                new_joints.append(available[idx])
                available = np.delete(available, idx, axis=0)
            new_joints = np.array(new_joints)
            mcp, pip, dip, tip = new_joints

            # --- Soft phalange length adjustment ---
            L1 = np.linalg.norm(pip - mcp)
            L2 = np.linalg.norm(dip - pip)
            L3 = np.linalg.norm(tip - dip)

            L2_expected = L1 * phalange_ratios[1]
            L3_expected = L1 * phalange_ratios[2]

            if abs(L2 - L2_expected)/L2_expected > 0.15:
                vec2 = (dip - pip) / np.linalg.norm(dip - pip)
                dip = pip + vec2 * L2_expected

            if abs(L3 - L3_expected)/L3_expected > 0.15:
                vec3 = (tip - dip) / np.linalg.norm(tip - dip)
                tip = dip + vec3 * L3_expected

            prev_joints = np.array([mcp, pip, dip, tip])

        # --- Compute angles ---
        pip_angle = angle(mcp, pip, dip)
        dip_angle = angle(pip, dip, tip)

        # --- Save frame data ---
        data.append([
            frame_idx,
            *mcp, *pip, *dip, *tip,
            pip_angle, dip_angle
        ])

        # --- Draw skeleton ---
        pts = np.array([mcp, pip, dip, tip], dtype=int)
        cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 255), thickness=2)
        for c in pts:
            cv2.circle(frame, (int(round(c[0])), int(round(c[1]))), 6, (0, 255, 0), -1)

        # --- Write annotated frame to output video ---
        out.write(frame)

    scale = 0.5
    display_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    display_mask = cv2.resize(mask, (display_frame.shape[1], display_frame.shape[0]))

    cv2.imshow("Detection", display_frame)
    cv2.imshow("Mask", display_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

# --- Cleanup ---
cap.release()
out.release()
cv2.destroyAllWindows()

# --- Save CSV ---
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
df.to_csv("finger_angles_pinch_exo.csv", index=False)
