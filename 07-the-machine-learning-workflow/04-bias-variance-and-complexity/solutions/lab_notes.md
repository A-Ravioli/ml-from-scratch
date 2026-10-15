# bias–variance & complexity · lab notes

## experiment 1: sweeping the degree

**what people usually predict:** a smooth U-shape with its minimum around degree 3–5, and variance at degree 12 "a few times" the noise.

**what happens** ($n = 30$, 200 training sets per degree):

| degree | 0 | 1 | 2 | 3 | 5 | 7 | 9 | 10 | 12 |
|---|---|---|---|---|---|---|---|---|---|
| bias² | 0.50 | 0.20 | 0.21 | 0.005 | 0.0003 | 0.003 | 0.002 | 4.3 | 626 |
| variance | 0.02 | 0.02 | 0.05 | 0.02 | 0.04 | 1.0 | 9.5 | 473 | 139,000 |
| expected error | 0.61 | 0.32 | 0.35 | **0.11** | 0.13 | 1.1 | 9.6 | 477 | 140,000 |

the minimum is at degree 3 (0.113, close to the noise floor 0.09), and the single-dataset CV curve also picks degree 3. the variance at degree 12 is about $1.5\times10^6$ times the noise variance at the lab's seed (139,000 / 0.09), and across 20 seeds it ranges from about $3\times10^4$ to $6\times10^7$ (median $3\times10^5$): the edges are wildly unstable. bias² drops only at the odd degrees 1, 3 and 5, then is essentially zero, until the measured value climbs again at degree 10 and beyond. with $n = 100$, everything calms down: the best degrees are 5–7 (about 0.096), and the variance at degree 12 is only 0.10.

**why:** $\sin(\pi x)$ is an odd function, so the even powers contribute nothing to its best polynomial approximation: going from degree 0 to 1, 2 to 3, 4 to 5 adds a useful odd power, and the in-between steps add only variance. by degree 3 the class can represent $\sin(\pi x)$ on $[-1, 1]$ to within about 0.005 in squared error, so higher degrees add almost nothing but variance. the variance explodes because 30 random points leave gaps, especially near the edges, and a high-degree polynomial that must pass near all of them swings wildly in the gaps (experiment 2). the measured bias² climbing at degree 10+ is not real bias, and it isn't random noise either: it's a *systematic* bias of the estimator (lesson, idea 2). $\bar h$ is an average of only $R = 200$ fits, so it misses the true average model by an error of variance $\text{variance}/R$, and squaring that error adds $\text{variance}/R$ to the measured bias² on average: at degree 12 that's about $139{,}000/200 \approx 700$, which accounts for essentially all of the 626 measured. subtracting $\widehat{\text{variance}}/(R - 1)$ leaves roughly zero (within noise), and with $R = 2{,}000$ rounds the inflated value drops to around 90, shrinking like $1/R$. the red mean curve of experiment 2 leaving $f$ at the edges is the same artifact: there the variance of a single fit is about 650, so the mean of 200 fits has a standard error of about $\sqrt{650/200} \approx 1.8$. with $n = 100$ the gaps are smaller, the variance at every degree drops roughly like $p/n$ (on paper 3, except near the edges), and the optimum moves to higher degree: more data buys more complexity.

## experiment 2: where does the variance live?

**what people usually predict:** spread evenly, or largest wherever the curve is steepest.

**what happens** (degree 10, 30 points): in the middle ($|x| \le 0.5$) the 200 fits agree to a variance of about 0.04, *less* than the noise variance. at the edges ($|x| \ge 0.9$) the average variance is about 650, more than ten thousand times larger. the thin lines all hug the true curve in the middle and fan out wildly at both ends; the mean of the 200 fits (red) leaves the true function there too.

**why:** near the middle every fit is pinned down by many nearby points. at the edges there are few points (on a random sample of 30 there may be none in $[0.9, 1]$), and the high powers $x^9, x^{10}$ are tiny in the middle but large at the ends, so the coefficients on them are poorly determined by the data, and their errors show up exactly where those powers are large. in least-squares terms, the edge points have high **leverage** (the diagonal of the hat matrix; ch 02 · topic 04's lab saw the same effect for a line). this is why flexible models are dangerous where the data is thin, and why extrapolating a high-degree polynomial, even a little beyond the data, is almost always a mistake. the same pattern (Runge's phenomenon) appears in interpolation theory.

## experiment 3: learning curves

**what people usually predict:** more data helps every model, the line included.

**what happens** (5 shuffled folds of 400 points):

| degree | validation MSE at 32 points | at 320 points | training MSE at 320 points |
|---|---|---|---|
| 1 | 0.32 | 0.31 | 0.30 |
| 3 | 0.10 | 0.10 | 0.09 |
| 9 | **9.6** | 0.10 | 0.09 |

the line's validation error barely moves from 32 to 320 points: its two curves converge to about 0.30, three times the noise. degree 9 starts catastrophically (9.6) and drops to 0.10 by 320 points, where it's as good as degree 3.

**why:** the line's plateau is its bias² (about 0.20, the best line's distance from $\sin(\pi x)$) plus the noise 0.09: with more data its training error *rises* a little and its validation error falls a little, and they meet at the best line, which is still wrong. no amount of data fixes the hypothesis class. degree 9's problem is variance, which falls roughly like $p/n$, so data cures it. the diagnostic rule: curves that meet early and high mean bias (change the model or the features); a large gap that shrinks with data means variance (get data, regularize, or simplify). these are the two pictures you'll recognize in every chapter from here on.
