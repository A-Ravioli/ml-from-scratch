# graphical models · lab notes

## experiment 1: explaining away

**what people usually predict:** that the alarm makes a burglary very likely (the alarm is 94% reliable, so "about 90%"), and that the earthquake report doesn't change that much, or even makes things more alarming.

**what happens:** $P(B \mid \text{alarm}) = 0.3736$ (not 0.9), and with the earthquake report it drops to $P(B \mid \text{alarm}, \text{quake}) = 0.0033$, a factor of **114** lower, back near the prior of 0.001. learning there was *no* quake raises it to 0.4848. the quake on its own leaves $P(B) = 0.001$ unchanged. `d_separated(B, E)` is `True`; `d_separated(B, E | A)` is `False`.

**why:** $B \to A \leftarrow E$ is a collider. with $A$ unobserved the causes are independent, which is why "quake only" doesn't move the burglary probability. observing the common effect couples them: given the alarm, *something* must have set it off, so evidence for one cause is evidence against the other. the alarm makes the two hypotheses compete, and the earthquake wins (it's twice as likely a priori and still sets the alarm off 29% of the time). the low 0.37 is the base-rate effect: false alarms happen with probability 0.001 every day, burglaries with 0.001, quakes with 0.002 and a 29% trigger rate, so most alarms are not burglaries even with a 94% detection rate.

## experiment 2: Berkson's paradox

**what people usually predict:** a correlation near zero among the famous, because talent and looks are independent ("selection doesn't create relationships"), or perhaps a slightly positive one.

**what happens:** in the whole population the correlation is $-0.013$ (zero up to sampling noise). among the 2,935 famous people (threshold 1.5) it's **$-0.682$**, and the curve on the right shows it growing more negative as the threshold rises, from $-0.30$ at threshold $-1$ to about $-0.80$ at threshold 3. `d_separated(talent, looks | famous)` is `False`.

**why:** "famous" is a collider, talent → famous ← looks, and restricting the dataset to famous people *is* conditioning on it. among the famous, a person with little talent must have great looks to have made it, and vice versa: the selection boundary talent + looks > threshold is a line with slope $-1$, and slicing a round Gaussian cloud above such a line leaves a thin band that's elongated along it. the higher the threshold, the thinner the band, the stronger the correlation. this is the same structure as explaining away, and it's everywhere: hospital patients (Berkson's original 1946 example, two diseases that both cause admission), published papers (effect size and sample size both drive publication), surviving companies, admitted students. whenever a dataset is selected on an outcome, correlations between the outcome's causes are suspect.

## experiment 3: one more observation, dependence again

**what people usually predict:** that conditioning on more variables can only reduce dependence, so $I(S; R) \ge I(S; R \mid C) \ge I(S; R \mid C, W)$, with the last one zero.

**what happens:** exactly: $I(S; R) = 0.0505$ bits, $I(S; R \mid C) = 0$, and $I(S; R \mid C, W) = $ **0.1099** bits, twice the unconditional dependence. `d_separated` agrees: not separated, separated, not separated. the plug-in estimates from 2,000 samples (shipped seed) are 0.0443, 0.00005 and 0.1107 bits. the estimate for the d-separated case is never exactly zero: under independence the plug-in estimator is approximately $\chi^2_{\text{df}}/(2n\ln 2)$ with df = 2 here, so its average is about $0.0007$ bits at $n = 2000$ and shrinks like $1/n$ (this seed happened to land low).

**why:** there are two paths between $S$ and $R$: the fork through $C$ (cloudy days mean rain and no sprinkler: a negative dependence) and the collider through $W$. with nothing observed only the fork is open. observing $C$ blocks the fork, and the collider is still blocked: independent. observing $W$ as well opens the collider: wet grass is explained by either the sprinkler or the rain, so given that it's wet, the two compete, which is explaining away again. conditioning can create dependence as well as destroy it; mutual information has no monotonicity in the conditioning set. for structure learning (the PC algorithm, ch 29), the estimate's positive bias is why every independence decision has to be a statistical test (a $G$-test or a permutation test against the $\chi^2$ null), and why errors in early tests propagate into the learned graph.
