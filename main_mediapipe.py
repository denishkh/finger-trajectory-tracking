import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# --- Video path ---
video_path = r"data\test3_index.mp4"
cap = cv2.VideoCapture(video_path)

# --- MediaPipe Hands setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

data = []
frame_idx = 0

def compute_angle(a, b, c):
    """Compute angle at joint b between points a-b-c"""
    ab = a - b
    bc = c - b
    cosang = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]  # take first detected hand

        h, w, _ = frame.shape
        # Get 2D coordinates of index finger landmarks
        # Index finger: MCP=5, PIP=6, DIP=7, TIP=8
        coords = []
        for idx in [5,6,7,8]:
            lm = hand.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            coords.append([x,y])
        mcp, pip, dip, tip = map(np.array, coords)

        # Compute angles
        pip_angle = compute_angle(mcp, pip, dip)
        dip_angle = compute_angle(pip, dip, tip)

        data.append([frame_idx, *mcp, *pip, *dip, *tip, pip_angle, dip_angle])

        # Optional: draw landmarks
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    # Show frame
    cv2.imshow("MediaPipe Hands", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
hands.close()

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
df.to_csv("finger_angles_test3.csv", index=False)
print("Saved -> finger_angles_mediapipe.csv")
