import cv2

backends = [
    ("DEFAULT", cv2.CAP_ANY),
    ("DSHOW", cv2.CAP_DSHOW),
    ("MSMF", cv2.CAP_MSMF)
]

for backend_name, backend in backends:
    print("\nTesting backend:", backend_name)

    for i in range(5):
        cam = cv2.VideoCapture(i, backend)

        if cam.isOpened():
            print(f"Camera found: index={i}, backend={backend_name}")

            ret, frame = cam.read()

            if ret:
                print("Frame captured successfully.")
            else:
                print("Camera opened but frame not captured.")

            cam.release()
        else:
            print(f"No camera at index {i}")