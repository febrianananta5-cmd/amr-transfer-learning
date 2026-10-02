import cv2
import os
import glob

folder = "dataset/train/person"
os.makedirs(folder, exist_ok=True)

existing_files = glob.glob(os.path.join(folder, "person_*.jpg"))

numbers = []

for file in existing_files:
    name = os.path.basename(file)
    number = int(name.split("_")[1].split(".")[0])
    numbers.append(number)

if numbers:
    count = max(numbers) + 1
else:
    count = 0

print("Foto person yang sudah ada:", len(existing_files))

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Kamera tidak bisa dibuka.")
    exit()

print("Tekan SPACE untuk mengambil foto.")
print("Tekan Q untuk keluar.")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Gagal membaca kamera.")
        break

    text = f"Foto berikutnya: person_{count:03d}.jpg"

    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Capture Person", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        filename = os.path.join(
            folder,
            f"person_{count:03d}.jpg"
        )

        cv2.imwrite(filename, frame)

        print("Tersimpan:", filename)

        count += 1

    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Program selesai.")