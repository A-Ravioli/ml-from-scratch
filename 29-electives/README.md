# 29 · electives

> go deep on two or more specialized areas.

by now you've built the core of modern machine learning yourself, from autograd to diffusion to PPO. the field doesn't stop there: it branches into subfields with their own questions, tools and habits of evidence. a causal-inference researcher asks *what would happen if we intervened*, which no amount of predictive accuracy answers. a continual-learning researcher asks why every network trained by gradient descent forgets, and what it costs to stop it. a quantum-ML researcher asks whether a quantum kernel ever beats a classical one, and has to be honest when the answer is no. this chapter is ten short tracks into ten such subfields. you pick at least two. each one teaches its core ideas in the usual way (lessons, staged builds with live checks, labs that surprise you) and ends with a small project in which your method meets a classical baseline. the capstone then asks the question every subfield shares: is the method actually better than the simple thing, by how much, and how sure are you?

**you will build:** in each elective block, the subfield's core algorithms from scratch (see the table), and a project that compares your method with a classical baseline. in the capstone, a small fair-comparison toolkit (paired seeds, $t$ and bootstrap intervals, an exact sign-flip test, Holm's correction, budget tables with explicit units) and an *elective portfolio*: one structured report per elective you completed. the electives own no `mlfs` modules: they use earlier modules through their contracts, and nothing later depends on them.
**prerequisites:** depend on the block; see the table. every block assumes chapter 05's bootstrap and permutation tests and chapter 07's habits of honest evaluation, and the capstone uses your `mlfs.stats`.
**time · scaffolding:** about 60 hours per elective block, plus a few hours for the capstone's toolkit and write-up · L4 (spec-driven): lessons fix the names and conventions, notebooks give signatures with one-line docstrings and the checks, one hint per stage; the capstone gives you a brief and checks.

## how the electives work

- **pick at least two blocks.** each block is 3–4 topics in the usual format, numbered across the whole chapter so every elective has its own range (causal inference is 01–04, continual learning 12–14, and so on). the last topic's lab ends with the block's project.
- **blocks are independent.** no block requires another, and you can take them in any order. a few pairs reinforce each other: the Fisher information shows up as an importance weight in continual learning (topic 13) and as a metric in information geometry (18–20), and optimal transport (17) and audio codecs (26) both reuse your generative-model code from chapters 21–22.
- **each block's project feeds the capstone.** keep the per-seed scores your project prints: the portfolio entry is built from them, not from means.
- **choosing.** pick what you'll use. if you work with observational data (medicine, economics, policy, online experiments), causal inference. if your models keep learning after deployment, continual learning. if you work on sequences, time series and state-space models. the theory-minded will like information geometry and optimal transport; the product-minded, recommender systems.

## the blocks

| block | topics | prerequisites (chapters) | time | the project | newly unlocked library components |
|---|---|---|---|---|---|
| **causal inference** | [01](01-structural-causal-models-and-do-calculus/) SCMs & do-calculus · [02](02-potential-outcomes-and-weighting/) potential outcomes & weighting · [03](03-double-machine-learning/) double machine learning · [04](04-causal-discovery/) causal discovery | 00, 04, 05, 07, 08, 10, 12 | ~60 h | recover a causal graph with PC on semi-synthetic data built on real California-housing covariates, then estimate the effect of $T$ on $Y$ with IDA: 0.960 ± 0.018 against a truth of 0.96, where the naive regression gives 1.12 | after the block you may use `DoubleML` or `EconML` for DML and AIPW estimates, and `causal-learn`'s PC with Fisher-z tests for discovery, since you've built each of them |
| **time series & state-space models** | 05 classical forecasting · 06 Kalman filters · 07 linear SSMs (S4) · 08 selective SSMs | 04, 05, 07, 08, 12, 15, 17, 18 | ~60 h <!-- fill: measured time from the block report --> | a selective state-space model against a transformer on a long-sequence task | <!-- fill: unlocked components from the block report --> |
| **meta-learning** | 09 MAML · 10 prototypical networks · 11 in-context learning as meta-learning | 06, 07, 15, 18 (16 helps) | ~60 h <!-- fill: measured time from the block report --> | in-context learning compared with least squares and MAML on few-shot linear regression | <!-- fill: unlocked components from the block report --> |
| **continual learning** | [12](12-catastrophic-forgetting/) catastrophic forgetting · [13](13-regularization-approaches/) regularization approaches · [14](14-replay-and-architectural-approaches/) replay & architectural approaches | 04, 05, 06, 07, 08, 12, 15 | ~60 h | beat fine-tuning on class-incremental split MNIST within a 39,200-float memory budget: a per-class Gaussian generator reaches ACC 0.780 ± 0.018 against fine-tuning's 0.196 (a plain 50-image reservoir: 0.565 ± 0.010) | after the block you may use Avalanche's continual-learning benchmarks, metrics and its EWC, SI and replay strategies, since you've built the harness and each method |
| **optimal transport** | 15 Wasserstein distances · 16 entropic OT & Sinkhorn · 17 OT in machine learning | 02, 05, 06, 15, 21, 22 | ~60 h <!-- fill: measured time from the block report --> | minibatch OT couplings straighten flow-matching paths (few-step sampling quality, OT vs independent coupling) | <!-- fill: unlocked components from the block report --> |
| **information geometry** | 18 the Fisher metric · 19 natural gradient · 20 K-FAC | 02, 04, 05, 06, 15 (13 helps) | ~60 h <!-- fill: measured time from the block report --> | K-FAC against Adam on your chapter-15 networks | <!-- fill: unlocked components from the block report --> |
| **recommender systems** | 21 matrix factorization · 22 implicit feedback · 23 two-tower retrieval | 00, 02, 06, 07, 08, 11, 15 (17 helps) | ~60 h | a two-stage retrieve-then-rank recommender against popularity and matrix-factorization baselines <!-- fill: headline numbers from the block report --> | after the block you may use `implicit` (ALS, BPR) and FAISS's flat and IVF indexes, since you've built them; not HNSW or product quantization, which the block doesn't build |
| **audio & speech** | 24 audio representations · 25 speech recognition with CTC · 26 neural audio codecs | 03, 04, 15, 16, 17, 21 | ~60 h <!-- fill: measured time from the block report --> | bitrate against reconstruction quality for your residual-VQ codec | <!-- fill: unlocked components from the block report --> |
| **federated learning** | 27 federated averaging · 28 heterogeneity · 29 privacy & secure aggregation | 05, 06, 07, 15, 28 | ~60 h <!-- fill: measured time from the block report --> | the privacy–accuracy–communication triangle of DP-FedAvg with secure aggregation | <!-- fill: unlocked components from the block report --> |
| **quantum ML** | 30 qubits & circuits · 31 variational circuits · 32 quantum kernels | 02, 04, 06, 09 | ~60 h <!-- fill: measured time from the block report --> | when (if ever) a quantum-kernel SVM beats a classical-kernel SVM | <!-- fill: unlocked components from the block report --> |

the per-topic tables (core ideas, what you build, what each lab finds) are in [`CURRICULUM.md`](../CURRICULUM.md) under *29 · electives*, and each block's lessons say precisely which earlier topics they use.

## capstone

*the elective portfolio*: you build a fair-comparison toolkit (pairing scores by seed, a $t$ or bootstrap interval for the paired difference, an exact sign-flip test and the smallest p-value your seed count allows, Holm's correction across claims, a budget table that makes its unit explicit, and a structured portfolio entry with a markdown renderer), then write one entry and a short report per elective you completed. two worked entries show the format: PC + IDA against a naive regression (absolute error 0.015 vs 0.160, 20 paired draws, verdict *better*), and a Gaussian generator against fine-tuning (ACC difference +0.584, 95% CI [+0.562, +0.606] over 5 seeds, verdict *better*, with a sign-flip p of 0.0625, the floor for five seeds). the continual-learning budget table shows the unit flipping the winner: in stored numbers the generator wins; in bytes, 200 uint8 images do. small CPU config: a few seconds (the notebook trains nothing; the numbers come from your block projects).

see [`capstone/brief.md`](capstone/brief.md).

## checkpoint

[`checkpoint.md`](checkpoint.md): the cross-cutting skills every elective exercises (choosing a baseline, matching a budget, paired seeds and intervals, spotting a simulator that favours a method, reading a new subfield's paper), short per-block questions for the blocks you took, three by-hand problems, and a 60-minute blind rebuild of the capstone toolkit.

## notation

shared across the chapter; each block's lessons add their own (they're consistent with the chapter notation tables of the chapters they build on).

| symbol | meaning |
|---|---|
| $n$ | number of paired seeds (or draws, splits, task streams) in a comparison |
| $d_i$, $\bar d$, $s_d$ | paired difference method − baseline on seed $i$; their mean; their standard deviation (with $n - 1$) |
| $t_{1-\alpha/2,\,n-1}$ | Student $t$ quantile with $n - 1$ degrees of freedom |
| $T^*$ | the statistic recomputed under a random sign flip (the null distribution) |
| $m$, $p_{(j)}$, $\tilde p$ | number of claims tested together; the $j$-th smallest p-value; a Holm-adjusted p-value |
| $\mathcal G$, $\mathrm{do}(T = t)$ | a causal graph; an intervention setting $T$ to $t$ (causal inference) |
| $Y(1), Y(0)$, $e(x)$ | potential outcomes; the propensity score $P(T = 1 \mid X = x)$ (causal inference) |
| $R_{ij}$, ACC, BWT | accuracy on task $j$ after training on task $i$; average final accuracy; backward transfer (continual learning) |
| $F$ | Fisher information matrix (continual learning's importance weights, information geometry's metric) |
| $\mathcal D_t$ | the data available at task, client or time step $t$ (continual and federated learning, time series) |

## how this connects

- builds on: the bootstrap, permutation tests and multiple-testing control (ch 05), honest evaluation with baselines and seeds (ch 07), and each block's own prerequisites in the table: regression and graphical models for causal inference (ch 08, ch 12), PyTorch training (ch 15) for most blocks, transformers (ch 18) for state-space models and meta-learning, generative models (ch 21–22) for optimal transport and audio codecs, DP-SGD (ch 28) for federated learning, SVMs (ch 09) for quantum kernels.
- used later in: research practice and the thesis (ch 30), whose experimental-rigor topic turns the capstone's toolkit into `mlfs.research` (multi-seed runs, intervals, paired comparisons, ablation tables), and whose thesis can grow out of an elective project. the electives themselves are leaves: no later chapter depends on any one of them.

## core references

the comparison methodology, and one anchor per finished block; every block's lessons have their own *going deeper* lists.

- Holm (1979), "A simple sequentially rejective multiple test procedure", *Scandinavian Journal of Statistics* 6(2). [jstor.org/stable/4615733](https://www.jstor.org/stable/4615733)
- Dror, Baumer, Shlomov & Reichart (2018), "The hitchhiker's guide to testing statistical significance in natural language processing", *ACL*. [aclanthology.org/P18-1128](https://aclanthology.org/P18-1128/)
- Bouthillier et al. (2021), "Accounting for variance in machine learning benchmarks", *MLSys*. [arXiv:2103.03098](https://arxiv.org/abs/2103.03098)
- Pearl (2009), *Causality: Models, Reasoning, and Inference*, 2nd ed., Cambridge University Press. [doi:10.1017/CBO9780511803161](https://doi.org/10.1017/CBO9780511803161)
- Peters, Janzing & Schölkopf (2017), *Elements of Causal Inference*, MIT Press (open access).
- van de Ven, Tuytelaars & Tolias (2022), "Three types of incremental learning", *Nature Machine Intelligence* 4. [doi:10.1038/s42256-022-00568-3](https://doi.org/10.1038/s42256-022-00568-3)

starting points for the other blocks: Hyndman & Athanasopoulos (2021), *Forecasting: Principles and Practice*, 3rd ed. ([otexts.com/fpp3](https://otexts.com/fpp3/)) and Gu & Dao (2023), "Mamba" ([arXiv:2312.00752](https://arxiv.org/abs/2312.00752)); Finn, Abbeel & Levine (2017), "Model-agnostic meta-learning" ([arXiv:1703.03400](https://arxiv.org/abs/1703.03400)); Peyré & Cuturi (2019), *Computational Optimal Transport* ([arXiv:1803.00567](https://arxiv.org/abs/1803.00567)); Amari (2016), *Information Geometry and Its Applications*, Springer ([doi:10.1007/978-4-431-55978-8](https://doi.org/10.1007/978-4-431-55978-8)); Koren, Bell & Volinsky (2009), "Matrix factorization techniques for recommender systems", *IEEE Computer* 42(8) ([doi:10.1109/MC.2009.263](https://doi.org/10.1109/MC.2009.263)); Graves, Fernández, Gomez & Schmidhuber (2006), "Connectionist temporal classification", *ICML* ([doi:10.1145/1143844.1143891](https://doi.org/10.1145/1143844.1143891)); McMahan, Moore, Ramage, Hampson & Agüera y Arcas (2017), "Communication-efficient learning of deep networks from decentralized data" ([arXiv:1602.05629](https://arxiv.org/abs/1602.05629)); Schuld & Petruccione (2021), *Machine Learning with Quantum Computers*, Springer ([doi:10.1007/978-3-030-83098-4](https://doi.org/10.1007/978-3-030-83098-4)).
