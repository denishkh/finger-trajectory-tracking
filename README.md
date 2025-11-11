# finger-trajectory-tracking
There are a lot of resources to track the finger trajectory from video using computer vision ( example mediapipe, openpose etc.). But when it comes to tracking fingers with glove, these methods are not always useful. <br>

This is a tutorial which shows a method to track finger ( in fact this can be applied to anything in 2D). Here, I demonstrate this with index finger of hand exoskeleton. <br>
This project tracks four **green markers** placed along a finger (MCP, PIP, DIP, and TIP) in a video and computes the **joint angles** (PIP and DIP) across frames. These green markers can be any non-reflective piece of paper that can be glued.
It uses **OpenCV** for image processing, **NumPy** for geometry, and **Pandas** for saving results.

## Requirements
Install the dependencies:
```bash
pip install opencv-python numpy pandas
