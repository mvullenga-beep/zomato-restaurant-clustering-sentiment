# Zomato Restaurant Clustering & Sentiment Analysis

Unsupervised machine learning + NLP project on **100 Hyderabad restaurants** and **10,000 customer reviews** from Zomato.

**Goal:** help customers find the right restaurant and help Zomato see which kinds of restaurants it has and where it is lagging, by (1) clustering restaurants into segments and (2) analysing the sentiment of customer reviews.

> Author: [your name] | Video walkthrough: [paste your video link here]

---

## Data

| File | Rows | Description |
|---|---|---|
| Restaurant names and metadata | 105 | Name, link, cost for one person, collections, cuisines, timings |
| Restaurant reviews | 10,000 | Restaurant, reviewer, review text, rating, reviewer metadata, time, pictures |

Reviews are from May 2016 to May 2019 (exactly 100 per restaurant, mostly from 2018-2019).

## What I did

1. **Data cleaning:** fixed cost stored as text, a junk rating value, 36 duplicate rows, and split the reviewer metadata into reviews/followers.
2. **EDA:** 15 charts (univariate, bivariate, multivariate) with insights and business impact.
3. **Hypothesis tests:** cost vs rating (Spearman), pictures vs rating (Mann-Whitney), critics vs other reviewers (Mann-Whitney).
4. **Feature engineering:** outlier capping (IQR), log transform, scaling, multi-hot encoding of cuisines, VADER sentiment scores.
5. **Text pre-processing:** contraction expansion, lower-casing, punctuation/URL/number removal, stop-word removal (negations kept), lemmatization, POS tagging, TF-IDF.
6. **Clustering models:** K-Means, Agglomerative (hierarchical), Gaussian Mixture Model, tuned and compared with silhouette, Davies-Bouldin, Calinski-Harabasz and a resampling stability test.
7. **Bonus supervised model:** Logistic Regression, Naive Bayes and Linear SVM predicting positive vs negative reviews, with 5-fold cross-validation and Grid Search.
8. **Explainability and deployment:** top words per sentiment, feature importance for the clusters, models saved with `joblib` and tested on unseen data.

## Results

**Final model: K-Means with 3 clusters** (silhouette about 0.27, so the segments overlap somewhat, which is normal for real business data).

| Segment | Restaurants | Avg cost (Rs.) | Avg rating | Sentiment | Pictures / review | Review length (words) |
|---|---|---|---|---|---|---|
| Budget, low-buzz | 46 | 512 | 3.35 | 0.29 | 0.27 | 33 |
| Premium, top-rated | 16 | 1,341 | 4.21 | 0.74 | 0.64 | 54 |
| Buzzy, mid-range | 38 | 1,089 | 3.65 | 0.59 | 1.35 | 70 |

**Key findings**
- Ratings are skewed high (about 38% are 5 stars), but about 17% of reviews are 1 star.
- More expensive restaurants tend to rate higher (Spearman rho about 0.38, significant).
- Reviews with pictures rate higher (3.86 vs 3.54, significant).
- Experienced "critic" reviewers do not rate differently from other reviewers (not significant).
- Review sentiment classifier: [add your best macro-F1 and negative-class recall from your notebook output].

**Business use:** feature premium restaurants in curated collections, support budget restaurants with quality programmes and review prompts, and use the high-engagement mid-range restaurants for promotions.

## How to run

1. Open the notebook in **Google Colab** (upload it, or use "Open in Colab" from GitHub).
2. Run the first cell (installs `vaderSentiment`).
3. When asked, upload the two CSV files from this repository (the files named *Zomato Restaurant ...csv*).
4. Choose **Runtime -> Run all**.

To run locally: `pip install -r requirements.txt`, then open the notebook in Jupyter.

## Repository contents

```
.
├── Zomato_Capstone_Submission.ipynb        # full analysis (with outputs)
├── Zomato Restaurant names and Metadata.csv
├── Zomato Restaurant reviews.csv
├── requirements.txt
└── README.md
```

## Limitations and future work

- Only 100 restaurants in one city; the data is the latest 100 reviews per restaurant, so long-term trends cannot be measured.
- VADER is a general-purpose sentiment tool and may miss food-specific language.
- Cluster structure is moderate (silhouette about 0.27).
- Next steps: add location data and menu prices, try DBSCAN and transformer-based sentiment models, and deploy the saved models in a small web app.
