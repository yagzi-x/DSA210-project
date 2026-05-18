# DSA210 Project

**Name:** Yağız Çuhadar  
**Student ID:** 33930  

## Project Title

**To what extent does film length affect its IMDb rating?**

## Project Description

This project analyzes whether movie length has an effect on IMDb ratings. The main idea is to examine if longer movies tend to receive higher ratings, or if the relationship is weak and affected by other variables such as genre and release year.

The project uses a public IMDb movie dataset and applies data cleaning, exploratory data analysis, hypothesis testing, and machine learning methods to study the relationship between movie duration and IMDb rating.

## Dataset

The dataset used in this project is a public IMDb movie metadata dataset saved as `data/raw/movie_metadata.csv`.

Main variables used in the analysis:

- `duration`: movie length in minutes
- `imdb_score`: IMDb rating of the movie
- `genres`: movie genre information
- `title_year`: release year of the movie

After data cleaning, 4790 movies remained in the dataset for the EDA stage.

## Stage 3: EDA and Hypothesis Testing

For the EDA stage, I cleaned the movie dataset and analyzed the relationship between movie length and IMDb rating. I used visualizations such as histograms, boxplots, scatter plots, and genre-based comparisons.

The results showed a moderate positive relationship between runtime and IMDb rating. The Pearson correlation was 0.3538 and the Spearman correlation was 0.3723. I also compared short and long movies based on the median runtime. Longer movies had a higher average IMDb rating than shorter movies.

I also tested statistical significance using t-tests, Mann-Whitney U tests, ANOVA, and Kruskal-Wallis tests. The results were statistically significant, which suggests that the observed relationship is not only due to random chance. However, genre differences also seem to play an important role, so movie length alone is not enough to fully explain IMDb ratings.

## Stage 4: Machine Learning

For the ML stage, I applied several regression models to predict IMDb ratings using movie runtime, genre, and release year.

The models tested were:

- Baseline Mean Model
- Linear Regression
- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

The models were compared using MAE, RMSE, and R² score. Based on MAE, Ridge Regression performed the best among the tested models.

The best model results were approximately:

- **Best model:** Ridge Regression
- **MAE:** 0.704
- **RMSE:** 0.924
- **R²:** 0.254

These results show that runtime, genre, and release year provide some useful information for predicting IMDb ratings. However, the R² score also shows that these variables do not fully explain movie ratings. This makes sense because IMDb ratings are subjective and can be affected by many other factors, such as actors, director, budget, number of votes, popularity, and audience expectations.

## Main Findings

The analysis suggests that longer movies tend to have higher IMDb ratings on average. However, the relationship is not strong enough to say that movie length alone determines rating. Genre and release year also matter, and the machine learning results show that more variables would be needed for a stronger prediction model.

## Repository Structure

Current repository organization:

```text
DSA210-project/
│
├── data/
│   ├── raw/
│   │   └── movie_metadata.csv
│   └── processed/
│       └── movies_cleaned.csv
│
├── figures/
│   ├── stage3/
│   └── stage4/
│
├── scripts/
│   ├── stage3_movie_analysis.py
│   └── stage4_ml_analysis.py
│
├── reports/
│   ├── Proposal_yagiz.pdf
│   └── DSA210_Final_Report_Yagiz_Cuhadar.pdf
│
├── stage3_results_summary.txt
├── stage4_ml_results_summary.txt
├── requirements.txt
└── README.md
```

## Files in This Repository

- `reports/Proposal_yagiz.pdf`: project proposal
- `reports/DSA210_Final_Report_Yagiz_Cuhadar.pdf`: final report
- `data/raw/movie_metadata.csv`: original movie dataset
- `data/processed/movies_cleaned.csv`: cleaned dataset used in analysis
- `scripts/stage3_movie_analysis.py`: EDA and hypothesis testing script
- `scripts/stage4_ml_analysis.py`: machine learning script
- `stage3_results_summary.txt`: summary of EDA and hypothesis testing results
- `stage4_ml_results_summary.txt`: summary of ML model results
- `figures/stage3/`: visualizations created during the EDA stage
- `figures/stage4/`: visualizations created during the ML stage
- `requirements.txt`: Python dependencies

## How to Run the Code

Install the required Python packages:

```powershell
py -3.13 -m pip install pandas numpy matplotlib scipy scikit-learn
```

Run the EDA and hypothesis testing script:

```powershell
py -3.13 scripts/stage3_movie_analysis.py
```

Run the machine learning script:

```powershell
py -3.13 scripts/stage4_ml_analysis.py
```

## Limitations and Future Work

This project only uses a limited number of features, mainly runtime, genre, and release year. IMDb ratings are influenced by many other factors that are not fully included in this analysis. In future work, the project could be improved by adding variables such as number of votes, budget, gross revenue, director information, actors, or critic scores.

## AI Use Disclosure

I used AI tools as a support tool during this project, mainly for organizing the project structure, improving the clarity of writing, and getting help while debugging code errors. I did not use AI as the main source of analysis. The dataset selection, code execution, model outputs, result interpretation, and final decisions were completed and checked by me.

