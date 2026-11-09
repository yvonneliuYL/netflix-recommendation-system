# Recommendation Systems for a Movie Company

IE 7275 Data Mining in Engineering (Northeastern University) — Group Project 1, Spring 2026

A decision-support recommender for a film production company, not a user-based recommender: given a target genre (for example Comedy), rank the actors and directors linked to popular and commercially successful movies. The data is Netflix movie metadata up to 2025 with TMDb popularity, vote, budget and revenue fields. The dataset has no user-level ratings, so collaborative filtering (e.g. SVD) was not used.

**Members:** Tanatswa Manzini, Chan Nhi Nguyen, Yi-Fan Liu

## Results

16,000 movies × 18 columns; 15,324 movies remain after cleaning.

**Models A and B: genre → actor/director ranking.** Test split of 2,299 movies (70 / 15 / 15 split, seed 42). For each test movie, its genres are the query and its director and up to three cast members are the relevant items (only names in the top-20 lists count). 124 test movies were scored.

| Model | Precision@10 | Recall@10 | F1@10 | NDCG@10 |
|---|---|---|---|---|
| A. Association rules (Apriori) | 0.0306 | 0.2419 | 0.0532 | 0.1370 |
| B. Hybrid (association + popularity) | 0.0306 | 0.2419 | 0.0532 | 0.1256 |

Model B reported 15 empty outputs; these are test movies whose genres matched no rule, and they are not among the 124 scored movies.

**Model C: XGBoost success classifier.** "Success" means the top 20% of movies by log revenue. Stratified 80 / 20 split; 3,065 test movies; threshold 0.5.

| Precision | Recall | F1 | AUC | RMSE (probability) |
|---|---|---|---|---|
| 0.7052 | 0.8548 | 0.7729 | **0.9501** | 0.2774 |

The talent ranking built from Model C could not be scored: for Comedy, the test-set ground truth was empty (`GT_size = 0`), so Precision@10, Recall@10 and NDCG@10 are all 0.0000. The notebook explains this as a mismatch between the movie-level training target and the talent-level evaluation.

Example output of Model C for **Comedy** (4,433 movies, Bayesian-smoothed average success probability, m = 10):

| Rank | Actor | Titles | Score | Director | Titles | Score |
|---|---|---|---|---|---|---|
| 1 | Robert De Niro | 18 | 0.635658 | Tyler Perry | 10 | 0.629884 |
| 2 | Samuel L. Jackson | 12 | 0.542086 | David Gordon Green | 5 | 0.438970 |
| 3 | James Franco | 7 | 0.518447 | Steven Soderbergh | 5 | 0.435701 |

All numbers above are from the saved notebook outputs.

## Methods

- **Data preparation:** drop the empty `duration` column; drop rows missing both genres and description; fill missing genres from keywords in the description; drop rows missing director, cast or country. Keep the 20 most frequent directors and actors (others become "Other"); one-hot encode genres, language and director; multi-label encode cast. Log-transform budget and revenue and standardize the numeric columns.
- **EDA:** vote count vs. vote average, rating distribution, average popularity and frequency by genre, top directors and cast, correlation heatmap. OLS regressions of revenue and log revenue to see which features drive revenue.
- **Model A, association rules:** Apriori (`mlxtend`) on one transaction per movie (genres, director, up to three cast members), min support 0.0003, min confidence 0.05. This gives 1,984 frequent itemsets and 21,898 rules; 144 rules have only genres on the left and one person on the right. Score = 0.7 × confidence + 0.3 × lift.
- **Model B, hybrid:** Model A candidates re-ranked by 0.6 × normalized association score + 0.4 × normalized average popularity in the target genres. People with fewer than 2 titles in those genres are dropped.
- **Model C, XGBoost:** binary classifier (300 trees, max depth 5, learning rate 0.05, `scale_pos_weight` ≈ 4.0) on 138 features: budget, popularity, vote count, vote average, release year, rating and the one-hot genre, director, cast and language columns. Revenue is not a feature. Predicted success probabilities are averaged per actor or director within the target genre and shrunk toward the genre mean (Bayesian smoothing).

## Repository contents

```
netflix_recommendation.ipynb                 # final notebook: preparation, EDA, regression, Models A-C, evaluation
netflix_movies_detailed_up_to_2025.csv       # dataset used by the notebook (16,000 movies)
reports/
└── Recommendation_Systems_for_Movie_Company.pdf   # project report
requirements.txt
```

## How to run

**Data:** [Netflix Movies and TV shows till 2025](https://www.kaggle.com/datasets/bhargavchirumamilla/netflix-movies-and-tv-shows-till-2025) on Kaggle (file `netflix_movies_detailed_up_to_2025.csv`, included here). The TV-shows file of the same dataset is not used and not included.

The notebook was run on **Google Colab**. To run it locally:

1. `pip install -r requirements.txt`
2. Open `netflix_recommendation.ipynb` from the repository folder.
3. Skip the `drive.mount` cell and the next cell, which reads the CSV from a Google Drive path. The cell after that reads `netflix_movies_detailed_up_to_2025.csv` from the current folder.
4. Near the end, three cells save `top_actors.csv`, `top_directors.csv` and `xgb_model.pkl` to a Drive path. Change the path or skip them.

**Notes**
- The notebook was saved from Colab without execution counts, so cells are not numbered.
- The target genre for the examples is set in the notebook (`["Comedy"]`); other genres can be passed to `recommend_A`, `recommend_B` or the Model C ranking cells.
