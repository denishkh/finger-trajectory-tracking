import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV
df = pd.read_csv("finger_angles_mediapipe.csv")

# Number of frames to plot along the trajectory (e.g., 10 evenly spaced)
frame_indices = np.linspace(0, len(df)-1, 10, dtype=int)

plt.figure(figsize=(6,8))

# Plot intermediate frames in grey
for idx in frame_indices[1:-1]:  # skip first and last
    mcp = np.array([df.loc[idx, "mcp_x"], df.loc[idx, "mcp_y"]])
    pip = np.array([df.loc[idx, "pip_x"], df.loc[idx, "pip_y"]])
    dip = np.array([df.loc[idx, "dip_x"], df.loc[idx, "dip_y"]])
    tip = np.array([df.loc[idx, "tip_x"], df.loc[idx, "tip_y"]])
    
    # Translate so MCP is origin
    pip_rel = pip - mcp
    dip_rel = dip - mcp
    tip_rel = tip - mcp
    
    x = [0, pip_rel[0], dip_rel[0], tip_rel[0]]
    y = [0, pip_rel[1], dip_rel[1], tip_rel[1]]
    
    plt.plot(x, y, marker='o', color='lightgrey', alpha=0.6)

# First frame in green
idx = 0
mcp = np.array([df.loc[idx, "mcp_x"], df.loc[idx, "mcp_y"]])
pip = np.array([df.loc[idx, "pip_x"], df.loc[idx, "pip_y"]])
dip = np.array([df.loc[idx, "dip_x"], df.loc[idx, "dip_y"]])
tip = np.array([df.loc[idx, "tip_x"], df.loc[idx, "tip_y"]])
pip_rel = pip - mcp
dip_rel = dip - mcp
tip_rel = tip - mcp
plt.plot([0, pip_rel[0], dip_rel[0], tip_rel[0]],
         [0, pip_rel[1], dip_rel[1], tip_rel[1]],
         marker='o', color='green', lw=3, label='First frame')

# Last frame in red
idx = len(df)-1
mcp = np.array([df.loc[idx, "mcp_x"], df.loc[idx, "mcp_y"]])
pip = np.array([df.loc[idx, "pip_x"], df.loc[idx, "pip_y"]])
dip = np.array([df.loc[idx, "dip_x"], df.loc[idx, "dip_y"]])
tip = np.array([df.loc[idx, "tip_x"], df.loc[idx, "tip_y"]])
pip_rel = pip - mcp
dip_rel = dip - mcp
tip_rel = tip - mcp
plt.plot([0, pip_rel[0], dip_rel[0], tip_rel[0]],
         [0, pip_rel[1], dip_rel[1], tip_rel[1]],
         marker='o', color='red', lw=3, label='Last frame')

plt.xlabel("X (relative to MCP)")
plt.ylabel("Y (relative to MCP)")
plt.title("Index Finger Pose Over Time (MCP as Origin)")
plt.legend()
plt.gca().invert_yaxis()  # Match image coordinate system
plt.axis("equal")
plt.show()
