# splitting & cross-validation · lab notes

## experiment 1: preprocessing before splitting

**what people usually predict:** both leaky variants are "a bit optimistic", maybe 55–60%; or, if they've heard about leakage, that both are badly inflated.

**what happens** ($d = 2{,}000$ noise features, 10 datasets):

| variant | mean CV accuracy | range |
|---|---|---|
| honest | 0.49 | 0.38 – 0.62 |
| leaky selection | **0.94** | 0.92 – 0.97 |
| leaky scaling | 0.49 | 0.40 – 0.60 |

selecting features on all the rows turns pure noise into 94% "accuracy", every single time. scaling on all the rows does nothing measurable. the leaky-selection accuracy grows with $d$: about 0.65 at $d = 50$, 0.88 at $d = 500$, 0.95 at $d = 5{,}000$, while the honest estimate stays at chance.

**why:** the leak that matters is the one that carries **label** information. the top-20 features were chosen because, on these 60 rows *including the test rows*, they happen to separate the classes; with $d$ candidates the best chance correlations are about $\sqrt{2\log d/n}$ (on paper 3), so more features means stronger fake signal. the centroid classifier then finds that separation again in every test fold, because it was built into the test rows when the features were chosen. scaling with the full-data mean and standard deviation leaks a little about the test rows' *inputs*, but nothing about their labels, and with 60 rows the statistics barely move when 12 are removed. (it's not always harmless: with few rows, heavy tails, or steps like PCA or target encoding, unsupervised and especially label-aware preprocessing can leak measurably.) the honest variant puts selection inside `fit`, so `cross_val_score` clones and refits it on each training fold. that's the structural fix, and topic 05's `Pipeline` makes it the default.

the honest estimates spread from 0.38 to 0.62 around the truth: with 60 rows, a CV estimate is noisy. that spread is another reason to report uncertainty, not a single number.

## experiment 2: patients, not recordings

**what people usually predict:** KFold somewhat above 50% (it's "mixed up"), GroupKFold at 50%.

**what happens:** shuffled KFold reports **1.00** on every dataset; GroupKFold reports 0.46 (between 0.38 and 0.61). as the within-patient spread grows, the KFold score falls: about 0.69 at spread 1.0 and 0.53 at spread 2.0, where recordings of the same patient are no more alike than recordings of different patients.

**why:** under KFold, almost every test recording has 6 or 7 recordings of the same patient in the training folds, and those are its nearest neighbours: 1-NN predicts the patient's label by *recognizing the patient*. the score answers "can you label a new recording of someone we've already labelled?", which nobody needs. GroupKFold holds out whole patients, so the nearest neighbour is always a different patient, whose label is unrelated: chance. the mean slightly *below* 0.5 is a real and well-known artefact: holding out a group of sick patients leaves the training set with more healthy ones, which tilts a neighbour-based classifier against exactly the held-out class. (it shows up with balanced labels and small samples; it doesn't mean the model is "anti-learning".) the general rule: split by the unit that will be new at deployment.

## experiment 3: the future leaks into the past

**what people usually predict:** both near 50%, since features and label are independent processes.

**what happens:** shuffled KFold reports **0.70**; TimeSeriesSplit reports 0.52, varying from about 0.40 to 0.59 between series.

**why:** the features drift slowly, so in feature space the closest training rows to a test day are the days just before and after it, which shuffled KFold puts in the training set. the label process also changes slowly (its autocorrelation at lag 1 is 0.95), so a neighbouring day almost always has the same label. 1-NN is really doing "copy the label from yesterday or tomorrow", using time as a hidden feature. at deployment, tomorrow isn't available. TimeSeriesSplit trains only on the past: the nearest training days are at the end of the training block, and their label only predicts the first few days of the test block (the label forgets in about 20 days, the blocks are 100 days long), so the score is close to chance. its big variation between series is honest, too: a single series gives few, highly correlated test days, so any estimate from it is noisy. (if the label forgot more slowly, the honest score would rise above 0.5: "persistence", predicting that tomorrow looks like today, is a genuine forecasting skill, and TimeSeriesSplit would correctly credit it.)
