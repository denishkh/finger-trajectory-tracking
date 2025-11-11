import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV
df = pd.read_csv("finger_angles_green_robust.csv")

# Drop rows with missing data
valid_rows = df.dropna(subset=['mcp_x', 'pip_x', 'dip_x', 'tip_x']).reset_index(drop=True)

if len(valid_rows) < 2:
    print("Not enough valid frames to plot.")
    exit()

# Get first, last, and a few evenly spaced intermediate frames
n_samples = 10  # total number of skeletons to display
indices = np.linspace(0, len(valid_rows) - 1, n_samples, dtype=int)
frames_to_plot = valid_rows.iloc[indices]

# Extract coordinate helper
def get_coords(frame):
    x = [frame['mcp_x'], frame['pip_x'], frame['dip_x'], frame['tip_x']]
    y = [frame['mcp_y'], frame['pip_y'], frame['dip_y'], frame['tip_y']]
    return x, y

plt.figure(figsize=(6, 8))

# Plot intermediate frames (gray, faint)
for i, (_, frame) in enumerate(frames_to_plot.iloc[1:-1].iterrows()):
    x, y = get_coords(frame)
    plt.plot(x, y, '-o', color='gray', alpha=0.3)

# Plot first frame (blue)
x_first, y_first = get_coords(frames_to_plot.iloc[0])
plt.plot(x_first, y_first, '-o', color='blue', label='First Frame', markersize=8, linewidth=2)

# Plot last frame (red)
x_last, y_last = get_coords(frames_to_plot.iloc[-1])
plt.plot(x_last, y_last, '-o', color='red', label='Last Frame', markersize=8, linewidth=2)

# Formatting
plt.gca().invert_yaxis()  # Because image coordinates
plt.xlabel("X (pixels)")
plt.ylabel("Y (pixels)")
plt.title("Finger Skeleton: First, Last, and Intermediate Frames")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
