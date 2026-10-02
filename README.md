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

---

## Latency dan FPS

Pengujian latency dilakukan menggunakan model ResNet-18 Partial Fine-Tuning pada CPU.

Hasil pengujian sebanyak 100 kali:

| Parameter | Hasil |
|---|---:|
| Device | CPU |
| Jumlah pengujian | 100 |
| Average Latency | 40.58 ms |
| Theoretical FPS | 24.64 FPS |

Latency tersebut merupakan waktu inferensi model dan belum mencakup seluruh pipeline kamera seperti pengambilan frame, preprocessing, dan tampilan hasil.

---

## Pengujian Webcam

Model Partial Fine-Tuning telah diuji menggunakan webcam laptop.

Hasil pengujian terhadap empat kelas:

| Objek | Hasil |
|---|---|
| Backpack | Berhasil dikenali |
| Bottle | Berhasil dikenali |
| Chair | Berhasil dikenali |
| Person | Berhasil dikenali |

Program pengujian dapat dijalankan dengan:

```bash
python test_partial.py