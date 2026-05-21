from pyk4a import PyK4A

k4a = PyK4A()
k4a.start()

cam_matrix = (
    k4a.calibration.get_camera_matrix(0)
)

print(cam_matrix)

k4a.stop()