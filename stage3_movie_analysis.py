
from __future__ import annotations

import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu, f_oneway, kruskal

warnings.filterwarnings("ignore")

# =========================
# EDIT THESE IF NEEDED
# =========================
RAW_DATA_PATH = "movie_metadata.csv"
PROCESSED_DATA_PATH = "data/processed/movies_cleaned.csv"
FIGURES_DIR = "figures"

# Change these to match your dataset column names
TITLE_COL = "movie_title"
RUNTIME_COL = "duration"
RATING_COL = "imdb_score"
GENRE_COL = "genres"
YEAR_COL = "title_year"      

# =========================
# HELPERS
# =========================
def parse_runtime_to_minutes(value):
    """
    Converts common runtime formats to minutes.
    Handles:
    - 142
    - 142 min
    - 2h 22m
    - 2 h 22 m
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    s = str(value).strip().lower()

    # plain numeric string
    try:
        return float(s)
    except Exception:
        pass

    # examples: '142 min'
    if "min" in s and "h" not in s:
        num = "".join(ch for ch in s if ch.isdigit() or ch == ".")
        try:
            return float(num)
        except Exception:
            return np.nan

    # examples: 2h 22m
    hours = 0
    minutes = 0
    if "h" in s:
        h_part = s.split("h")[0].strip()
        try:
            hours = float("".join(ch for ch in h_part if ch.isdigit() or ch == "."))
        except Exception:
            hours = 0

        after_h = s.split("h", 1)[1]
        if "m" in after_h:
            m_part = after_h.split("m")[0].strip()
            try:
                minutes = float("".join(ch for ch in m_part if ch.isdigit() or ch == "."))
            except Exception:
                minutes = 0
        return hours * 60 + minutes

    return np.nan


def save_plot(filename: str):
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(Path(FIGURES_DIR) / filename, dpi=200, bbox_inches="tight")
    plt.close()


def first_genre(x):
    if pd.isna(x):
        return np.nan
    return str(x).split(",")[0].strip()


# =========================
# LOAD
# =========================
print("Loading dataset...")
df = pd.read_csv(RAW_DATA_PATH)

needed_cols = [TITLE_COL, RUNTIME_COL, RATING_COL, GENRE_COL, YEAR_COL]
missing_cols = [c for c in needed_cols if c not in df.columns]
if missing_cols:
    raise ValueError(
        f"These columns were not found in your dataset: {missing_cols}\n"
        f"Dataset columns are: {list(df.columns)}\n"
        f"Edit the column names at the top of this script."
    )

df = df[needed_cols].copy()
df.columns = ["title", "runtime_raw", "imdb_rating_raw", "genre_raw", "release_year_raw"]

print(f"Original rows: {len(df)}")

# =========================
# CLEAN
# =========================
df["runtime"] = df["runtime_raw"].apply(parse_runtime_to_minutes)
df["imdb_rating"] = pd.to_numeric(df["imdb_rating_raw"], errors="coerce")
df["release_year"] = pd.to_numeric(df["release_year_raw"], errors="coerce")
df["genre"] = df["genre_raw"].apply(first_genre)

# remove missing key values
df = df.dropna(subset=["runtime", "imdb_rating", "genre", "release_year"]).copy()

# filter unrealistic values
df = df[(df["runtime"] >= 40) & (df["runtime"] <= 300)]
df = df[(df["imdb_rating"] >= 0) & (df["imdb_rating"] <= 10)]
df = df[(df["release_year"] >= 1900) & (df["release_year"] <= 2030)]

# remove duplicate titles if needed
df = df.drop_duplicates(subset=["title"]).copy()

# save cleaned data
Path(PROCESSED_DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(PROCESSED_DATA_PATH, index=False)

print(f"Rows after cleaning: {len(df)}")
print("\nBasic summary:")
print(df[["runtime", "imdb_rating", "release_year"]].describe())

# =========================
# EDA
# =========================

# 1) Runtime histogram
plt.figure(figsize=(8, 5))
plt.hist(df["runtime"], bins=30)
plt.xlabel("Runtime (minutes)")
plt.ylabel("Count")
plt.title("Distribution of Movie Runtime")
save_plot("runtime_histogram.png")

# 2) Rating histogram
plt.figure(figsize=(8, 5))
plt.hist(df["imdb_rating"], bins=20)
plt.xlabel("IMDb Rating")
plt.ylabel("Count")
plt.title("Distribution of IMDb Ratings")
save_plot("rating_histogram.png")

# 3) Runtime boxplot
plt.figure(figsize=(6, 4))
plt.boxplot(df["runtime"])
plt.ylabel("Runtime (minutes)")
plt.title("Boxplot of Runtime")
save_plot("runtime_boxplot.png")

# 4) Rating boxplot
plt.figure(figsize=(6, 4))
plt.boxplot(df["imdb_rating"])
plt.ylabel("IMDb Rating")
plt.title("Boxplot of IMDb Rating")
save_plot("rating_boxplot.png")

# 5) Scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(df["runtime"], df["imdb_rating"], alpha=0.5)
z = np.polyfit(df["runtime"], df["imdb_rating"], 1)
p = np.poly1d(z)
x_line = np.linspace(df["runtime"].min(), df["runtime"].max(), 200)
plt.plot(x_line, p(x_line))
plt.xlabel("Runtime (minutes)")
plt.ylabel("IMDb Rating")
plt.title("Runtime vs IMDb Rating")
save_plot("runtime_vs_rating.png")

# 6) Top genres only for clearer plots
top_genres = df["genre"].value_counts().head(8).index.tolist()
df_top = df[df["genre"].isin(top_genres)].copy()

# Genre vs runtime boxplot
genre_runtime_data = [df_top[df_top["genre"] == g]["runtime"] for g in top_genres]
plt.figure(figsize=(10, 5))
plt.boxplot(genre_runtime_data, tick_labels=top_genres, vert=True)
plt.xticks(rotation=30, ha="right")
plt.ylabel("Runtime (minutes)")
plt.title("Runtime by Genre (Top Genres)")
save_plot("genre_runtime_boxplot.png")

# Genre vs rating boxplot
genre_rating_data = [df_top[df_top["genre"] == g]["imdb_rating"] for g in top_genres]
plt.figure(figsize=(10, 5))
plt.boxplot(genre_rating_data, tick_labels=top_genres, vert=True)
plt.xticks(rotation=30, ha="right")
plt.ylabel("IMDb Rating")
plt.title("IMDb Rating by Genre (Top Genres)")
save_plot("genre_rating_boxplot.png")

# Average runtime by year
yearly = df.groupby("release_year", as_index=False).agg(
    avg_runtime=("runtime", "mean"),
    avg_rating=("imdb_rating", "mean"),
    movie_count=("title", "count"),
)
yearly = yearly[yearly["movie_count"] >= 5]

plt.figure(figsize=(10, 5))
plt.plot(yearly["release_year"], yearly["avg_runtime"])
plt.xlabel("Release Year")
plt.ylabel("Average Runtime")
plt.title("Average Runtime by Release Year")
save_plot("avg_runtime_by_year.png")

plt.figure(figsize=(10, 5))
plt.plot(yearly["release_year"], yearly["avg_rating"])
plt.xlabel("Release Year")
plt.ylabel("Average IMDb Rating")
plt.title("Average IMDb Rating by Release Year")
save_plot("avg_rating_by_year.png")

# =========================
# SIMPLE STATS
# =========================
print("\nCorrelations:")
pearson_r, pearson_p = pearsonr(df["runtime"], df["imdb_rating"])
spearman_rho, spearman_p = spearmanr(df["runtime"], df["imdb_rating"])
print(f"Pearson r = {pearson_r:.4f}, p = {pearson_p:.6f}")
print(f"Spearman rho = {spearman_rho:.4f}, p = {spearman_p:.6f}")

# =========================
# HYPOTHESIS TESTS
# =========================

# Test 1: short vs long movies
median_runtime = df["runtime"].median()
short_movies = df[df["runtime"] <= median_runtime]["imdb_rating"]
long_movies = df[df["runtime"] > median_runtime]["imdb_rating"]

print("\nTest 1: Short vs Long Movies (based on median runtime)")
print(f"Median runtime = {median_runtime:.2f} minutes")
print(f"Short movies mean rating = {short_movies.mean():.3f}")
print(f"Long movies mean rating  = {long_movies.mean():.3f}")

ttest_stat, ttest_p = ttest_ind(short_movies, long_movies, equal_var=False)
mw_stat, mw_p = mannwhitneyu(short_movies, long_movies, alternative="two-sided")
print(f"T-test p-value = {ttest_p:.6f}")
print(f"Mann-Whitney U p-value = {mw_p:.6f}")

# Test 2: Runtime-rating relationship already checked with correlation p-values

# Test 3: Runtime differs by genre?
print("\nTest 3: Runtime differences by genre")
runtime_groups = [df_top[df_top["genre"] == g]["runtime"] for g in top_genres]
anova_runtime_stat, anova_runtime_p = f_oneway(*runtime_groups)
kruskal_runtime_stat, kruskal_runtime_p = kruskal(*runtime_groups)
print(f"ANOVA p-value (runtime by genre) = {anova_runtime_p:.6f}")
print(f"Kruskal-Wallis p-value (runtime by genre) = {kruskal_runtime_p:.6f}")

# Test 4: Rating differs by genre?
print("\nTest 4: Rating differences by genre")
rating_groups = [df_top[df_top["genre"] == g]["imdb_rating"] for g in top_genres]
anova_rating_stat, anova_rating_p = f_oneway(*rating_groups)
kruskal_rating_stat, kruskal_rating_p = kruskal(*rating_groups)
print(f"ANOVA p-value (rating by genre) = {anova_rating_p:.6f}")
print(f"Kruskal-Wallis p-value (rating by genre) = {kruskal_rating_p:.6f}")

# =========================
# SAVE TEXT SUMMARY
# =========================
summary_path = Path("stage3_results_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("STAGE 3 RESULTS SUMMARY\n")
    f.write("=======================\n\n")
    f.write(f"Original rows loaded: see console output\n")
    f.write(f"Rows after cleaning: {len(df)}\n")
    f.write(f"Mean runtime: {df['runtime'].mean():.2f}\n")
    f.write(f"Median runtime: {df['runtime'].median():.2f}\n")
    f.write(f"Mean IMDb rating: {df['imdb_rating'].mean():.2f}\n")
    f.write(f"Median IMDb rating: {df['imdb_rating'].median():.2f}\n\n")
    f.write("Correlation tests\n")
    f.write(f"Pearson r = {pearson_r:.4f}, p = {pearson_p:.6f}\n")
    f.write(f"Spearman rho = {spearman_rho:.4f}, p = {spearman_p:.6f}\n\n")
    f.write("Short vs Long movies\n")
    f.write(f"Median runtime = {median_runtime:.2f}\n")
    f.write(f"Short mean rating = {short_movies.mean():.3f}\n")
    f.write(f"Long mean rating = {long_movies.mean():.3f}\n")
    f.write(f"T-test p-value = {ttest_p:.6f}\n")
    f.write(f"Mann-Whitney U p-value = {mw_p:.6f}\n\n")
    f.write("Genre tests\n")
    f.write(f"ANOVA p-value (runtime by genre) = {anova_runtime_p:.6f}\n")
    f.write(f"Kruskal-Wallis p-value (runtime by genre) = {kruskal_runtime_p:.6f}\n")
    f.write(f"ANOVA p-value (rating by genre) = {anova_rating_p:.6f}\n")
    f.write(f"Kruskal-Wallis p-value (rating by genre) = {kruskal_rating_p:.6f}\n")

print("\nDone.")
print(f"Cleaned data saved to: {PROCESSED_DATA_PATH}")
print(f"Figures saved to: {FIGURES_DIR}")
print("Summary text file saved to: stage3_results_summary.txt")
