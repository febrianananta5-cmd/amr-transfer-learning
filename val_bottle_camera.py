import cv2
import os
import glob

camera = cv2.VideoCapture(0)

folder = "dataset/val/bottle"
os.makedirs(folder, exist_ok=True)

files = glob.glob(os.path.join(folder, "bottle_*.jpg"))
count = len(files)

print("Foto validasi bottle yang sudah ada:", count)

while True:
    ret, frame = camera.read()

    if not ret:
        print("Kamera tidak bisa dibuka")
        break

    cv2.imshow("Validasi Bottle - SPACE = Foto, Q = Keluar", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 32:
        filename = os.path.join(folder, f"bottle_{count:03d}.jpg")
        cv2.imwrite(filename, frame)
        print("Foto disimpan:", filename)
        count += 1

    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()