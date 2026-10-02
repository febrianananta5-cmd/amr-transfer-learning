import cv2
import os
import glob

camera = cv2.VideoCapture(0)

folder = "dataset/val/chair"
os.makedirs(folder, exist_ok=True)

files = glob.glob(os.path.join(folder, "chair_*.jpg"))
count = len(files)

print("Foto validasi chair yang sudah ada:", count)

while True:
    ret, frame = camera.read()

    if not ret:
        print("Kamera tidak bisa dibuka")
        break

    cv2.imshow("Validasi Chair - SPACE = Foto, Q = Keluar", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 32:
        filename = os.path.join(folder, f"chair_{count:03d}.jpg")
        cv2.imwrite(filename, frame)
        print("Foto disimpan:", filename)
        count += 1

    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()