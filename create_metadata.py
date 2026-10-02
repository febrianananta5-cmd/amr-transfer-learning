import os
import csv

# Folder dataset
base_folder = "dataset"

# Nama file output
output_file = "metadata.csv"

# Header CSV
headers = [
    "filename",
    "class",
    "split",
    "distance",
    "lighting",
    "background",
    "occlusion"
]

rows = []

# Kelas yang digunakan
classes = [
    "backpack",
    "bottle",
    "chair",
    "person"
]

# Baca folder train dan val
for split in ["train", "val"]:

    for class_name in classes:

        folder = os.path.join(
            base_folder,
            split,
            class_name
        )

        if not os.path.exists(folder):
            print("Folder tidak ditemukan:", folder)
            continue

        for filename in sorted(os.listdir(folder)):

            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):

                rows.append([
                    filename,
                    class_name,
                    split,

                    # Diisi sebagai dokumentasi awal.
                    # Bisa diperbaiki manual jika diperlukan.
                    "tidak dicatat",
                    "tidak dicatat",
                    "indoor",
                    "tidak dicatat"
                ])


# Tulis metadata.csv
with open(
    output_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow(headers)

    writer.writerows(rows)


print()
print("==============================")
print("METADATA BERHASIL DIBUAT")
print("==============================")
print("File:", output_file)
print("Jumlah data:", len(rows))
print("==============================")