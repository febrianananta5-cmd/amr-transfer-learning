import matplotlib.pyplot as plt

# =========================================================
# DATA HASIL TRAINING
# =========================================================

epochs = range(1, 11)

# ---------------------------------------------------------
# 1. FEATURE EXTRACTION
# ---------------------------------------------------------

feature_train = [
    64.25, 88.41, 88.89, 87.92, 93.24,
    93.24, 94.20, 94.69, 97.58, 95.17
]

feature_val = [
    80.00, 95.00, 90.00, 95.00, 100.00,
    100.00, 100.00, 100.00, 100.00, 100.00
]

# ---------------------------------------------------------
# 2. PARTIAL FINE-TUNING
# ---------------------------------------------------------

partial_train = [
    89.37, 96.62, 98.55, 98.55, 97.58,
    96.14, 99.52, 95.17, 100.00, 99.52
]

partial_val = [
    100.00, 100.00, 100.00, 100.00, 100.00,
    100.00, 100.00, 100.00, 100.00, 100.00
]

# ---------------------------------------------------------
# 3. SCRATCH
# ---------------------------------------------------------

scratch_train = [
    83.57, 91.30, 94.20, 95.17, 92.75,
    99.52, 99.03, 98.07, 99.03, 100.00
]

scratch_val = [
    25.00, 75.00, 90.00, 100.00, 75.00,
    100.00, 100.00, 95.00, 100.00, 100.00
]


# =========================================================
# GRAFIK 1 - FEATURE EXTRACTION
# =========================================================

plt.figure()

plt.plot(
    epochs,
    feature_train,
    marker="o",
    label="Train Accuracy"
)

plt.plot(
    epochs,
    feature_val,
    marker="o",
    label="Validation Accuracy"
)

plt.title("Feature Extraction - ResNet-18")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.xticks(range(1, 11))
plt.ylim(0, 105)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "feature_extraction_accuracy.png",
    dpi=300
)

plt.show()


# =========================================================
# GRAFIK 2 - PARTIAL FINE-TUNING
# =========================================================

plt.figure()

plt.plot(
    epochs,
    partial_train,
    marker="o",
    label="Train Accuracy"
)

plt.plot(
    epochs,
    partial_val,
    marker="o",
    label="Validation Accuracy"
)

plt.title("Partial Fine-Tuning - ResNet-18")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.xticks(range(1, 11))
plt.ylim(0, 105)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "partial_finetuning_accuracy.png",
    dpi=300
)

plt.show()


# =========================================================
# GRAFIK 3 - SCRATCH
# =========================================================

plt.figure()

plt.plot(
    epochs,
    scratch_train,
    marker="o",
    label="Train Accuracy"
)

plt.plot(
    epochs,
    scratch_val,
    marker="o",
    label="Validation Accuracy"
)

plt.title("Scratch - ResNet-18")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.xticks(range(1, 11))
plt.ylim(0, 105)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "scratch_accuracy.png",
    dpi=300
)

plt.show()


# =========================================================
# GRAFIK 4 - PERBANDINGAN VALIDATION ACCURACY
# =========================================================

plt.figure()

plt.plot(
    epochs,
    feature_val,
    marker="o",
    label="Feature Extraction"
)

plt.plot(
    epochs,
    partial_val,
    marker="o",
    label="Partial Fine-Tuning"
)

plt.plot(
    epochs,
    scratch_val,
    marker="o",
    label="Scratch"
)

plt.title("Perbandingan Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.xticks(range(1, 11))
plt.ylim(0, 105)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "comparison_validation_accuracy.png",
    dpi=300
)

plt.show()


print("========================================")
print("SEMUA GRAFIK BERHASIL DIBUAT")
print("========================================")
print("1. feature_extraction_accuracy.png")
print("2. partial_finetuning_accuracy.png")
print("3. scratch_accuracy.png")
print("4. comparison_validation_accuracy.png")