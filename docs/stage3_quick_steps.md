# Stage 3 quick steps

1. Put your dataset here:
`data/raw/movies.csv`

2. Open `stage3_movie_analysis.py`

3. At the top of the file, edit these only if your dataset uses different names:
- `TITLE_COL`
- `RUNTIME_COL`
- `RATING_COL`
- `GENRE_COL`
- `YEAR_COL`

4. Install packages:
```bash
pip install pandas numpy matplotlib scipy
```

5. Run:
```bash
python stage3_movie_analysis.py
```

6. Upload these to GitHub after running:
- `data/processed/movies_cleaned.csv`
- `figures/`
- `stage3_results_summary.txt`
- `stage3_movie_analysis.py`

7. README progress note you can paste:
```md
## Stage 3 Progress
I collected a public movie dataset and cleaned the variables related to title, runtime, IMDb rating, genre, and release year. Then I performed exploratory data analysis to examine the distributions of runtime and rating, as well as the relationship between them. I also carried out hypothesis tests to compare short and long movies and to check whether runtime and rating differ across genres.
```
