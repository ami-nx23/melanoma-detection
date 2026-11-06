from pathlib import Path
import numpy as np
import pandas as pd
import cv2

DATA_DIR = Path("data/archive/melanoma_cancer_dataset")

rows = []

for split in ["train", "test"]:
    for category in ["benign", "malignant"]:

        folder = DATA_DIR / split / category

        for path in folder.glob("*.jpg"):

            image = cv2.imread(str(path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            features = {
                "filename": path.name,
                "split": split,
                "label": category,

                "red_mean": image_rgb[:,:,0].mean(),
                "green_mean": image_rgb[:,:,1].mean(),
                "blue_mean": image_rgb[:,:,2].mean(),

                "red_std": image_rgb[:,:,0].std(),
                "green_std": image_rgb[:,:,1].std(),
                "blue_std": image_rgb[:,:,2].std(),

                "hue_mean": image_hsv[:,:,0].mean(),
                "saturation_mean": image_hsv[:,:,1].mean(),
                "value_mean": image_hsv[:,:,2].mean(),

                "brightness": image_rgb.mean(),
                "contrast": image_rgb.std()
            }

            rows.append(features)

df = pd.DataFrame(rows)

df.to_csv("reports/image_features.csv", index=False)

print(df.head())
print("\nDataset shape:", df.shape)
print("\nClass distribution:")
print(df.groupby(["split", "label"]).size())