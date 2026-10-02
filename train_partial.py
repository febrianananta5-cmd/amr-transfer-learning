import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader


# =========================
# 1. Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# =========================
# 2. Preprocessing
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
# 3. Dataset
# =========================
train_dataset = datasets.ImageFolder(
    "dataset/train",
    transform=transform
)

val_dataset = datasets.ImageFolder(
    "dataset/val",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False
)

print("Kelas:", train_dataset.classes)
print("Jumlah data train:", len(train_dataset))
print("Jumlah data val:", len(val_dataset))


# =========================
# 4. Load ResNet-18
# =========================
model = resnet18(
    weights=ResNet18_Weights.DEFAULT
)


# =========================
# 5. Bekukan semua layer
# =========================
for parameter in model.parameters():
    parameter.requires_grad = False


# =========================
# 6. Buka layer4
# =========================
for parameter in model.layer4.parameters():
    parameter.requires_grad = True


# =========================
# 7. Ganti classifier
# =========================
model.fc = nn.Linear(
    model.fc.in_features,
    4
)


model = model.to(device)


# =========================
# 8. Loss
# =========================
criterion = nn.CrossEntropyLoss()


# =========================
# 9. Optimizer
# =========================
optimizer = torch.optim.Adam(
    [
        {
            "params": model.layer4.parameters(),
            "lr": 0.0001
        },
        {
            "params": model.fc.parameters(),
            "lr": 0.001
        }
    ]
)


# =========================
# 10. Training
# =========================
epochs = 10

for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total


    # =========================
    # Validation
    # =========================
    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {running_loss / len(train_loader):.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Acc: {val_accuracy:.2f}%"
    )


# =========================
# 11. Simpan model
# =========================
torch.save(
    model.state_dict(),
    "resnet18_partial.pth"
)

print("Training selesai.")
print("Model disimpan sebagai resnet18_partial.pth")