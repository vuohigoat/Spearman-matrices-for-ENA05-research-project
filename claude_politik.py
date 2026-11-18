import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "A Questionnaire regarding political endorsement in Hellu.csv"
METHOD = "spearman"  # "pearson" or "spearman" (spearman suits ordinal survey scales)

df = pd.read_csv(CSV_PATH)

# Short names for the columns
df.columns = [
    "timestamp", "gender", "year", "social_media_use", "platforms",
    "news_source", "political_interest", "trust_politicians",
    "political_content_where", "economic_scale", "value_scale", "party",
]

# --- Encode ordinal / categorical variables as numbers ---
usage_order = {
    "Less than 1 hour a day": 1,
    "1-2 hours a day": 2,
    "2-4 hours a day": 3,
    "4-6 hours a day": 4,
    "More than 6 hours a day": 5,
}
df["social_media_use_num"] = df["social_media_use"].map(usage_order)

# "2. year" -> 2, "3. year" -> 3, alumni -> NaN (dropped pairwise)
df["study_year"] = pd.to_numeric(df["year"].str.extract(r"(\d+)\.")[0])

df["is_female"] = (df["gender"] == "Female").astype(int)
df["is_male"] = (df["gender"] == "Male").astype(int)


# Multi-select answers -> number of items chosen
def count_items(s):
    if pd.isna(s) or not str(s).strip():
        return 0
    return len([x for x in str(s).split(";") if x.strip()])


df["n_platforms"] = df["platforms"].apply(count_items)
df["n_news_sources"] = df["news_source"].apply(count_items)
df["n_content_channels"] = df["political_content_where"].apply(count_items)

# Which items are used (binary flags)
df["news_from_social_media"] = df["news_source"].fillna("").str.contains("Social media").astype(int)
df["uses_tiktok"] = df["platforms"].fillna("").str.contains("Tiktok").astype(int)

cols = [
    "social_media_use_num", "study_year", "is_female", "is_male",
    "political_interest", "trust_politicians",
    "economic_scale", "value_scale",
    "n_platforms", "n_news_sources", "n_content_channels",
    "news_from_social_media", "uses_tiktok",
]

corr = df[cols].corr(method=METHOD)
print(corr.round(2))
corr.to_csv("correlation_matrix.csv")

# --- Heatmap ---
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(cols)))
ax.set_yticks(range(len(cols)))
ax.set_xticklabels(cols, rotation=45, ha="right")
ax.set_yticklabels(cols)
for i in range(len(cols)):
    for j in range(len(cols)):
        v = corr.iloc[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, label=f"{METHOD.capitalize()} correlation")
ax.set_title("Correlation matrix of survey variables")
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=150)
plt.show()
