import matplotlib.pyplot as plt
plt.plot(df["frame"], df["pip_angle"], label="PIP")
plt.plot(df["frame"], df["dip_angle"], label="DIP")
plt.xlabel("Frame")
plt.ylabel("Angle (deg)")
plt.legend()
plt.show()
