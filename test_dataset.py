from pathlib import Path
from PIL import Image

DATA_DIR = Path("data/archive/melanoma_cancer_dataset")

for split in ["train", "test"]:
    for category in ["benign", "malignant"]:
        folder = DATA_DIR / split / category
        images = list(folder.glob("*"))

        print(f"{split} - {category} : {len(images)} images")

        if images:
            image = Image.open(images[0])
            print("Dimensions :", image.size)
            print("Format :", image.format)