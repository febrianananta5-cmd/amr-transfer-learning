import torch
import torch.nn as nn
import time

from torchvision.models import resnet18


# =========================
# DEVICE
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# =========================
# LOAD MODEL
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

print("Model berhasil dimuat.")


# =========================
# INPUT GAMBAR
# =========================

input_tensor = torch.randn(
    1, 3, 224, 224
).to(device)


# =========================
# WARM UP
# =========================

print("Melakukan warm-up...")

for i in range(10):
    with torch.no_grad():
        model(input_tensor)


# =========================
# UKUR LATENCY
# =========================

jumlah_pengujian = 100

start_time = time.perf_counter()

with torch.no_grad():

    for i in range(jumlah_pengujian):
        model(input_tensor)

end_time = time.perf_counter()


# =========================
# HASIL
# =========================

total_time = end_time - start_time

latency_ms = (
    total_time / jumlah_pengujian
) * 1000

fps = 1000 / latency_ms


print()
print("==============================")
print("HASIL PENGUJIAN LATENCY")
print("==============================")
print(f"Jumlah pengujian : {jumlah_pengujian}")
print(f"Total waktu      : {total_time:.4f} detik")
print(f"Latency rata-rata: {latency_ms:.2f} ms")
print(f"FPS teoritis     : {fps:.2f}")
print("==============================")