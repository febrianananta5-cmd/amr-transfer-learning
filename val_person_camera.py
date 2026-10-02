import cv2
import os
import glob

camera = cv2.VideoCapture(0)

folder = "dataset/val/person"
os.makedirs(folder, exist_ok=True)

files = glob.glob(os.path.join(folder, "person_*.jpg"))
count = len(files)

print("Foto validasi person yang sudah ada:", count)

while True:
    ret, frame = camera.read()

    if not ret:
        print("Kamera tidak bisa dibuka")
        break

    cv2.imshow("Validasi Person - SPACE = Foto, Q = Keluar", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 32:
        filename = os.path.join(folder, f"person_{count:03d}.jpg")
        cv2.imwrite(filename, frame)
        print("Foto disimpan:", filename)
        count += 1

    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()