import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les caractéristiques
df = pd.read_csv("reports/image_features.csv")

# Sélectionner les variables numériques
features = df.select_dtypes(include="number")

# Calculer les corrélations
correlation_matrix = features.corr(method="spearman")

# Afficher la matrice
plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    center=0
)

plt.title("Feature Correlation Matrix")
plt.tight_layout()

plt.savefig(
    "reports/figures/correlation_matrix.png",
    dpi=300
)

plt.show()