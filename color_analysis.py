from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = Path("data/archive/melanoma_cancer_dataset")

categories = ["benign", "malignant"]
colors = ["red", "green", "blue"]

results = {}

for category in categories:
    folder = DATA_DIR / "train" / category
    images = sorted(folder.glob("*.jpg"))[:200]

    rgb_values = []

    for path in images:
        image = np.array(Image.open(path).convert("RGB"))
        means = image.mean(axis=(0, 1))
        rgb_values.append(means)

    results[category] = np.mean(rgb_values, axis=0)

# Graphique
x = np.arange(3)
width = 0.35

plt.figure(figsize=(9, 5))

plt.bar(
    x - width / 2,
    results["benign"],
    width,
    label="Benign"
)

plt.bar(
    x + width / 2,
    results["malignant"],
    width,
    label="Malignant"
)

plt.xticks(x, colors)
plt.ylabel("Mean pixel intensity (0-255)")
plt.title("Average RGB Color Comparison")
plt.legend()

plt.tight_layout()
plt.savefig("reports/figures/rgb_comparison.png", dpi=300)
plt.show()

print("Benign RGB:", results["benign"])
print("Malignant RGB:", results["malignant"])