# features & preprocessing · lab notes

## experiment 1: kNN with and without scaling

**what people usually predict:** scaling helps "a bit", maybe a few points; the unit of income shouldn't matter much since income is irrelevant anyway.

**what happens** (600 applicants, 5-fold CV):

| income recorded in | raw features | StandardScaler | MinMaxScaler |
|---|---|---|---|
| dollars (spread ≈ 19,000) | **0.553** | 0.823 | 0.818 |
| thousands (spread ≈ 19) | 0.808 | 0.823 | 0.818 |
| millions (spread ≈ 0.02) | **0.833** | 0.823 | 0.818 |

on raw features with income in dollars, kNN is barely above the 0.50 baseline. the scaled pipelines don't care about the unit at all. and with income in millions, the raw model is *better* than the scaled ones.

**why:** a Euclidean distance adds squared differences, so the feature with the largest spread decides who your neighbours are. in dollars, income's spread is about 1,500 times age's: the "nearest" applicants are those with the most similar income, which says nothing about the label. in millions, income's spread is tiny, so it effectively drops out of the distance, and the model uses age and tenure (with age, spread 12, weighted more than tenure, spread 5). that's accidental feature selection, and it happens to remove the one useless feature, which is why it beats standardization: standardizing gives the irrelevant feature exactly the same weight as the useful ones. the lesson isn't "never scale": it's that a distance-based model's behaviour depends on an arbitrary choice of units unless you fix the scales deliberately. standardizing is the neutral default; choosing weights (or features) on purpose is better still. scaling is per-feature and monotone, so a decision tree would give the same answer in every row of this table.

## experiment 2: one typo in a hundred

**what people usually predict:** both scalers suffer a little; the outliers are only 1% of the rows.

**what happens:**

| % of rows with tenure 500 | 0 | 0.2 | 0.5 | 1 | 2 |
|---|---|---|---|---|---|
| StandardScaler | 0.873 | 0.852 | 0.813 | 0.745 | 0.690 |
| MinMaxScaler | 0.873 | **0.727** | 0.728 | 0.690 | 0.683 |

a *single* corrupted row (0.2% of 600) costs MinMaxScaler 15 points of accuracy. StandardScaler degrades gradually and stays ahead. with 1% corrupted, min–max squeezes the normal tenures into $[0, 0.048]$ while age spans $[0, 1]$.

**why:** min–max scaling divides by the range, which one extreme value sets: with a maximum of 500 instead of about 25, every normal tenure is shrunk twenty-fold, and the informative feature effectively vanishes from the distance. the standard deviation also grows with outliers, but more slowly: $m$ outliers at distance $D$ among $n$ rows add about $mD^2/n$ to the variance, so it takes several before the column is crushed. both scalers are fitted per fold, so this isn't leakage: it's fragility. the fix is upstream of the scaler: find and correct (or drop) impossible values, or transform skewed features, or use a scaler built on robust statistics (median and interquartile range) that a few extreme values can't move.

## experiment 3: how to encode a category

**what people usually predict:** a small edge for one-hot, or no difference, since kNN "finds same-city neighbours anyway".

**what happens:** the integer code gets **0.560**, one-hot **0.777** (the churn rates by city are about 0.8, 0.2, 0.7, 0.2, 0.8, 0.2).

**why:** two things go wrong with the integer code. first, it invents a geometry: city 2's closest cities are 1 and 3, whose customers behave in the *opposite* way, so near neighbours across a city boundary are misleading. second, after standardization the whole city effect lives in one column with spread 1, competing with four numeric columns, three of them pure noise, so the distance is dominated by noise. one-hot puts every pair of different cities at the same distance and spreads the city information over six columns, each standardized; a neighbour from another city is now far away, so neighbours come from the same city, which is what predicts churn. as for the encoder fitted on all rows: it only learns the list of city names, so it leaks next to nothing here, but inside a pipeline you wouldn't have to ask (and a city that only appears in one test fold would correctly come out as unknown).
