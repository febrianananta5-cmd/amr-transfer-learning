# Implementasi Transfer Learning ResNet-18 untuk Klasifikasi Objek pada AMR KIT

## Deskripsi

Project ini merupakan implementasi klasifikasi objek menggunakan metode Transfer Learning dengan arsitektur ResNet-18.

Project dikembangkan sebagai bagian dari pembelajaran robotika dan ditujukan untuk mendukung kemampuan pengenalan objek pada Autonomous Mobile Robot (AMR) KIT.

Empat kelas objek yang digunakan adalah:

- Backpack
- Bottle
- Chair
- Person

Pada penelitian ini dilakukan perbandingan tiga metode:

1. Feature Extraction
2. Partial Fine-Tuning
3. Training from Scratch

Model dilatih menggunakan dataset gambar yang dikumpulkan menggunakan webcam laptop.

---

## Tujuan

Tujuan project ini adalah:

- Membangun model klasifikasi objek menggunakan ResNet-18.
- Menerapkan Transfer Learning menggunakan bobot ImageNet.
- Membandingkan Feature Extraction, Partial Fine-Tuning, dan Scratch.
- Menguji model menggunakan webcam.
- Mengukur latency dan FPS inferensi model.

---

## Dataset

Dataset terdiri dari empat kelas:

| Class | Train | Validation |
|---|---:|---:|
| Backpack | 50 | 5 |
| Bottle | 56 | 5 |
| Chair | 50 | 5 |
| Person | 51 | 5 |
| **Total** | **207** | **20** |

Dataset disimpan dalam struktur:

```text
dataset/
├── train/
│   ├── backpack/
│   ├── bottle/
│   ├── chair/
│   └── person/
│
└── val/
    ├── backpack/
    ├── bottle/
    ├── chair/
    └── person/