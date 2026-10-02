import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# =========================
# 1. Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# =========================
# 2. Kelas
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
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# 4. Dataset validation
# =========================
val_dataset = datasets.ImageFolder(
    "dataset/val",
    transform=transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False
)


# =========================
# 5. Load model Partial
# =========================
model = resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    4
)

model.load_state_dict(
    torch.load(
        "resnet18_partial.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()


# =========================
# 6. Prediksi
# =========================
all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)

        outputs = model(images)

        _, predictions = torch.max(
            outputs,
            1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# =========================
# 7. Confusion Matrix
# =========================
cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix:")
print(cm)


# =========================
# 8. Tampilkan grafik
# =========================
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

display.plot()

plt.title(
    "Confusion Matrix - ResNet-18 Partial Fine-Tuning"
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix_partial.png"
)

plt.show()

print(
    "\nConfusion matrix disimpan sebagai "
    "confusion_matrix_partial.png"
)