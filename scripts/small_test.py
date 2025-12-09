from matplotlib import pyplot as plt
import numpy as np

def R_y(deg):
    """Rotation matrix for rotation about +y by 'deg' degrees."""
    theta = np.deg2rad(deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c, 0,  s],
        [ 0, 1,  0],
        [-s, 0,  c],
    ])

def R_z(deg):
    theta = np.deg2rad(deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c, s,  0],
        [ -s, c,  0],
        [0, 0,  1],
    ])


# Fixed transform: viper frame → franka frame
# R_fv = R_y(-90.0) 
R_vf_y = R_y(90.0)
R_vf_z = R_z(180)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1, projection='3d')

ax.plot([0, 2], [0, 0], [0, 0], color='red')
ax.plot([0, 0], [0, 2], [0, 0], color='green')
ax.plot([0, 0], [0, 0], [0, 2], color='blue')

matrix2 = R_vf_y @ R_vf_z

xx = matrix2[:, 0]
yy = matrix2[:, 1]
zz = matrix2[:, 2]

ax.plot([0, xx[0]], [0, xx[1]], [0, xx[2]], color='red', linewidth=4)
ax.plot([0, yy[0]], [0, yy[1]], [0, yy[2]], color='green', linewidth=4)
ax.plot([0, zz[0]], [0, zz[1]], [0, zz[2]], color='blue', linewidth=4)


plt.show()

