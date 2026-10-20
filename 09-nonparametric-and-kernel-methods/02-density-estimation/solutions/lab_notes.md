# density estimation · lab notes

numbers from the reference solution with the lab's seeds and default slider value; experiment 3 uses the real California housing data in `~/.cache/mlfs`.

## experiment 1: same data, same bins, different origin

**what people usually predict:** small changes in bar heights, but the same overall shape and the same number of peaks.

**what happens:** with bin width 0.6, the 20 origins give histograms with anywhere from **2 to 6** peaks (6 origins give 2 peaks, 9 give 3, 3 give 4, and one each give 5 and 6). the two extreme histograms look like different datasets: one has two broad humps, the other a tall spike, a deep gap and a ragged right side.

**why:** each bar counts the points that happen to fall between two edges, and with 39 points a bar holds 1 to 7 of them. moving the edges by a fraction of a bin moves a point or two from one bar to its neighbour, which is a large relative change in both heights, and enough to create or erase a local maximum. the origin is an arbitrary choice that the data never asked for. features that survive every origin (here: more mass on the left, three loose groups) are trustworthy; individual peaks are not. a KDE centres a bump on every point instead of counting points between fixed edges, so it has no origin at all (it's roughly the average of the histograms over all origins, the "averaged shifted histogram" idea).

## experiment 2: the bandwidth trade-off, measured

**what people usually predict:** that a well-known rule of thumb is close to optimal, and that cross-validation is noisy and only roughly right.

**what happens:** the MISE (40 samples of 200 points) is minimized at $h \approx$ **0.23**, with MISE **0.0067**. Silverman's rule picks a median of **0.76**, more than three times too wide, with MISE **0.055**, eight times the optimum: on a sample, its estimate is a pair of low, wide humps with a hump of fake density between the modes. 5-fold cross-validation picks a median of **0.23**, with the middle half of its choices between 0.21 and 0.25, and its median choice has essentially the optimal MISE. on the log-log plot the variance falls roughly like $1/h$ and the bias² rises steeply (a slope between 3 and 4) until it saturates near the largest bandwidths, close to idea 5's small-$h$ predictions of $-1$ and $4$. (at the smallest bandwidths the measured bias² flattens and turns up: that's Monte Carlo noise, since the average of 40 noisy estimates still carries a variance of $1/40$ of theirs.)

**why:** Silverman's rule computes the bandwidth that would be optimal *if the data were one Gaussian* with the sample's standard deviation. this mixture has a standard deviation of about 2.06 (dominated by the distance between the modes) but its curvature $p''$ comes from components of width 0.5, so $R(p'')$ is far larger than the Gaussian assumption implies. for two well-separated equal components of width $\sigma_c = 0.5$, $R(p'') \approx \frac12 R(\varphi''_{\sigma_c})$, and since $h^\star \propto R(p'')^{-1/5}$ and $R(\varphi''_\sigma) \propto \sigma^{-5}$, the rule's bandwidth is too large by about $(\sigma/\sigma_c)\,2^{-1/5} \approx 4.1 \times 0.87 \approx 3.6$ (measured: $0.76/0.23 \approx 3.3$). cross-validation makes no shape assumption: it asks the held-out points directly, so it adapts to the narrow modes. its price is variance (the chosen $h$ varies from sample to sample) and computation ($k$ fits per candidate), both small here. on the offline stand-in in experiment 3, which really is a Gaussian blob, the two choices agree (0.37° against 0.36°): the rule of thumb is right exactly when its assumption is.

## experiment 3: where Californians live

**what people usually predict:** a factor of 2 or so between the rule of thumb and cross-validation.

**what happens:** cross-validation on 3,000 districts picks **0.100°** (about 10 km), Silverman's rule **0.544°**, **5.4×** wider. on 3,000 other districts, the CV bandwidth has a mean held-out log-likelihood of **−1.771** nats per district against **−2.740** for Silverman's: 0.97 nats better, so a typical real district gets about **2.6×** more density. the CV curve is lopsided: it collapses for small bandwidths (−24.6 at 0.01°, where held-out districts fall between the spikes) and declines gently for large ones. the KDE's samples reproduce the Bay Area, Los Angeles, San Diego and the Central Valley, but a few land in the Pacific just off the coast and in empty desert.

**why:** the population is concentrated in a few metropolitan areas and spread thinly elsewhere, so the density has sharp peaks (cities) and long, flat plateaus (rural counties). Silverman's rule sees one big blob with a standard deviation of about 2° and smooths accordingly. cross-validation trades off the cities, which want a narrow bandwidth, against isolated rural districts, which need a wide one so that held-out neighbours still get some density; 0.1° is that compromise, too wide for Los Angeles and too narrow for the desert. a Gaussian bump on a coastal district puts half its mass on the sea side, which is where the ocean samples come from: the KDE knows nothing about coastlines. adaptive (variable-bandwidth) KDEs, where each point's bandwidth grows with the distance to its $k$-th nearest neighbour, are the standard fix for both problems.
