import matplotlib.pyplot as plt

categories = ["Benign", "Malignant"]
counts = [5000, 4605]

total = sum(counts)
percentages = [count / total * 100 for count in counts]

plt.figure(figsize=(9, 6))

bars = plt.bar(categories, counts, width=0.55)

plt.title(
    "Distribution of Lesion Categories",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Lesion category", fontsize=12)
plt.ylabel("Number of images", fontsize=12)

for bar, count, percentage in zip(bars, counts, percentages):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 50,
        f"{count}\n({percentage:.1f}%)",
        ha="center",
        fontsize=11
    )

plt.ylim(0, 5700)
plt.tight_layout()

plt.savefig("reports/figures/class_distribution.png", dpi=300)
plt.show()