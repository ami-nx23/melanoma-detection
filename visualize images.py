from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Chemin vers le dataset
DATA_DIR = Path("data/archive/melanoma_cancer_dataset")

# Récupération de quelques images
benign_images = list((DATA_DIR / "train" / "benign").glob("*.jpg"))
malignant_images = list((DATA_DIR / "train" / "malignant").glob("*.jpg"))

# Création de la figure
fig, axes = plt.subplots(2, 4, figsize=(12, 7))

# Affichage des images bénignes
for i in range(4):
    image = Image.open(benign_images[i])
    axes[0, i].imshow(image)
    axes[0, i].set_title("Benign")
    axes[0, i].axis("off")

# Affichage des images malignes
for i in range(4):
    image = Image.open(malignant_images[i])
    axes[1, i].imshow(image)
    axes[1, i].set_title("Malignant")
    axes[1, i].axis("off")

plt.suptitle("Comparison of Benign and Malignant Lesions",
             fontsize=15)

plt.tight_layout()
plt.savefig("reports/figures/sample_images.png", dpi=300)
plt.show()