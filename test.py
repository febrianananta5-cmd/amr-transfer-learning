import torch
import torch.nn as nn
import cv2

from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


# =========================
# 1. Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# =========================
# 2. Nama kelas
# =========================
classes = [
    "backpack",
    "bottle",
    "chair",
    "person"
]


# =========================
# 3. Preprocessing
# =========================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# 4. Load ResNet-18
# =========================
model = resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    4
)


# =========================
# 5. Load model hasil training
# =========================
model.load_state_dict(
    torch.load(
        "resnet18_feature.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Model berhasil dimuat.")


# =========================
# 6. Buka kamera laptop
# =========================
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Kamera tidak bisa dibuka.")
    exit()

print("Kamera berhasil dibuka.")
print("Tekan tombol Q untuk keluar.")


# =========================
# 7. Prediksi kamera
# =========================
while True:

    ret, frame = camera.read()

    if not ret:
        print("Gagal membaca kamera.")
        break

    # BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocessing
    image = transform(rgb)

    # Tambahkan dimensi batch
    image = image.unsqueeze(0)

    image = image.to(device)

    # Prediksi
    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    label = classes[predicted.item()]

    confidence_value = confidence.item() * 100


    # Tampilkan hasil
    text = f"{label}: {confidence_value:.1f}%"

    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "AMR Object Classification",
        frame
    )


    # Tekan Q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# 8. Tutup kamera
# =========================
camera.release()
cv2.destroyAllWindows()

print("Program selesai.")