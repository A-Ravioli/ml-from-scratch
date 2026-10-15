# 20 · capstone: GNNs on molecular energies

## goal

predict the energy of small atomic clusters from their atoms and positions, with four graph networks that see different amounts of geometry, and find out, with honest uncertainty, what each one can and can't learn:

- **GCN** and **GIN** see only the *graph*: which atoms are within a cutoff distance of each other, and their species. no coordinates.
- **`gin_pos`** is GIN with the raw coordinates appended to the node features: it sees all the geometry, but not in a symmetry-respecting way.
- **EGNN** sees the geometry through squared distances and relative vectors, and is exactly invariant to rotations, reflections, translations and relabelling.

you'll build the benchmark, the models, a training loop, an invariance audit and bootstrap intervals, then run the comparison and write it up. this is the chapter's whole story in one experiment: what symmetry buys, what expressivity limits cost, and what the audit can prove.

## data: a substitute for QM9

the curriculum plans this capstone on a subset of QM9 (Ramakrishnan et al., 2014), 134k small organic molecules with DFT-computed properties. QM9 isn't available in this course: it isn't among the cached datasets, its usual loaders need PyTorch Geometric (which the course doesn't install), and nothing here downloads data ad hoc. instead you **generate** a molecular-style benchmark with an exact, cheap label, which turns out to be a better instrument for this particular comparison: the target is invariant to every symmetry by construction, the labels have no noise, and you can make as much data as you want.

**clusters.** each "molecule" has $n$ atoms, $n$ uniform in $\{5, \dots, 12\}$, each atom of species A or B with probability ½. the first atom sits at the origin; each next atom is placed at distance $r \sim U(1.0, 1.5)\,\sigma_{ab}$ from a uniformly chosen existing atom, in a uniformly random direction, and accepted only if it's at least $0.95\,\sigma$ from every atom already placed (otherwise redrawn). positions are finally centred on their centroid. (`make_dataset(n_molecules, *, n_atoms=(5, 12), seed=0)`, NumPy `default_rng(seed)`.)

**energy.** the truncated and shifted Lennard-Jones potential, summed over atom pairs closer than $r_c = 2.0$:

$$E = \sum_{i<j,\ r_{ij} < r_c} V_{ab}(r_{ij}) - V_{ab}(r_c), \qquad V_{ab}(r) = 4\varepsilon_{ab}\Big[\Big(\frac{\sigma_{ab}}{r}\Big)^{12} - \Big(\frac{\sigma_{ab}}{r}\Big)^{6}\Big],$$

with $\sigma_{AA} = 1.0$, $\sigma_{BB} = 1.2$, $\sigma_{AB} = 1.1$, $\varepsilon_{AA} = 1.0$, $\varepsilon_{BB} = 0.6$, $\varepsilon_{AB} = \sqrt{0.6}$. the steep $r^{-12}$ wall makes the target sharply sensitive to the closest contacts, which is what makes it hard. on the 1,000-molecule test set (`seed=2`) the per-atom-mean baseline, which knows each cluster's size and nothing else, has MAE 1.04.

**splits.** training molecules are the first 250, 1,000 or 4,000 of `make_dataset(4000, seed=1)`; the test set is `make_dataset(1000, seed=2)` and is used only for the final numbers.

**graphs.** `collate` turns a list of molecules into one batched graph whose edges connect every ordered pair of atoms of the same molecule closer than the cutoff $r_c = 2.0$, so the graph contains exactly the pairs that interact. node features are the one-hot species.

## the specification

the checks in `test_capstone.py` pin down what each function must do; the internals are yours.

| stage | function | specification |
|---|---|---|
| 1 | `lj_energy(pos, species)` | the energy above, a float (arrays or tensors in) |
| 1 | `make_dataset(n_molecules, *, n_atoms=(5, 12), seed=0)` | list of `{"pos": float64 (n, 3), centred; "species": int64 (n,); "energy": float}`, generated as above, deterministic in `seed` |
| 2 | `collate(molecules, *, cutoff=R_CUT, dtype=torch.float32)` | dict with `x` (N, 2) one-hot, `pos` (N, 3), `edge_index` (radius graph per molecule, node ids offset), `batch` (N,), `y` (B,), `num_graphs` |
| 3 | `build_model(kind, *, hidden=64, n_layers=3, seed=0)` | an `nn.Module` for `kind` in `{"gcn", "gin", "gin_pos", "egnn"}` mapping a collated batch to (B,) energies; built from your `mlfs.gnn` layers (`GCNConv`, `GINConv`, `EGNNLayer`, `global_add_pool`); invariant to relabelling, and (except `gin_pos`) to rotations, reflections and translations; `ValueError` for other kinds |
| 4 | `fit(model, molecules, *, epochs=60, batch_size=32, lr=3e-3, cutoff=R_CUT, seed=0)` | trains in place and returns the model |
| 4 | `evaluate(model, molecules, *, cutoff=R_CUT, batch_size=256)` | per-molecule absolute errors in energy units, a float64 array |
| 5 | `invariance_report(model, molecules, *, n_transforms=3, cutoff=R_CUT, seed=0)` | in float64 (on a copy of the model), the largest change in predicted energy under random rotations, reflections, translations and relabellings: a dict with those four keys |
| 5 | `mae_ci(errors, ...)`, `mae_difference_ci(errors_a, errors_b, ...)` | (MAE, lo, hi) by `mlfs.stats.bootstrap_ci`; the difference with a **paired** bootstrap over test molecules |

design notes from the reference solution (not requirements): the energy is extensive, so every model predicts a per-atom contribution and sums it (`global_add_pool`); targets are normalized per atom (a per-atom mean energy and a scale) inside the model, so `evaluate` works in energy units; GCN and GIN layers are residual ($h \leftarrow h + \operatorname{ReLU}(\text{conv}(h))$); training uses Adam with a cosine schedule and gradient clipping at 1. the learning rate matters more than anything else: with `lr=1e-3` the EGNN plateaus near GIN's error for 60 epochs; with `3e-3` it breaks through.

## configurations

| config | where | data | training | runtime (laptop CPU) |
|---|---|---|---|---|
| **small** (tests, sanity checks) | the reference notebook by default; the checks use their own | notebook: 100 and 300 training molecules, 200 test; checks: two datasets of 200 training molecules of 4–7 atoms, 150 test each | notebook: width 32, 20 epochs, seed 0; checks: one EGNN, width 32, 2 layers, 30 epochs | about 15 s for the notebook, under a minute for the checks |
| **full** (your write-up) | `MLFS_CAPSTONE=full` in the reference | 250 / 1,000 / 4,000 training molecules, 1,000 test | width 64, 3 layers, 60 epochs, `lr=3e-3`, batch 32, seeds 0, 1, 2 | 14–18 minutes for all 36 runs |

## deliverables

1. `capstone.py` with every stage implemented: all checks green (`pytest test_capstone.py`).
2. the full config: a table of test MAE at 4,000 training molecules for the per-atom-mean baseline and the four models, each with a 95% bootstrap interval (over test molecules, seed 0) and the three seeds' MAEs; learning curves over 250 / 1,000 / 4,000 molecules; the invariance audit of each trained model on 100 test molecules.
3. paired intervals for the comparisons you make claims about (at least GIN − EGNN and `gin_pos` − GIN).
4. the write-up at the end of the notebook, answering the analysis questions below.

## requirements

| check | threshold |
|---|---|
| `test_capstone.py` | all pass, including an EGNN trained by your `fit` beating the per-atom-mean baseline by ≥ 15% on at least one of the checks' two small datasets (the reference lands at 0.55–0.81 of the baseline on 30 datasets and 0.93 on the hardest of 100; an untrained model is at about 1) |
| full config, EGNN at 4,000 molecules | **test MAE ≤ 0.25** (the reference reaches 0.007–0.017 over seeds and reruns), with interval and seed spread |
| full config, every model | beats the per-atom-mean baseline, with a paired interval that excludes 0 (the reference does, in every run) |
| invariance audit | GCN, GIN and EGNN $\le 10^{-8}$ in all four columns (float64); `gin_pos` reported and explained |

## reference results

the reference solution, full config, on an Apple-silicon laptop CPU (PyTorch's default threading; 14–18 minutes for the whole config, depending on load). MAE in energy units on the 1,000 test molecules. CPU training here isn't bit-reproducible (scatter-adds accumulate in a thread-dependent order), so the numbers below are **ranges over four full runs** (three by the author, one by a reviewer), each with seeds 0, 1, 2; treat them as the spread you should expect, not targets to match.

| model | test MAE at 4,000 molecules, all seeds and runs | example: one run, seed 0 [95% CI] | train time per seed |
|---|---|---|---|
| per-atom mean | 1.040 (deterministic) | 1.040 [0.988, 1.093] | |
| GCN | 0.804–0.835 | 0.823 [0.778, 0.864] | 15–55 s |
| GIN | 0.682–0.685 | 0.684 [0.648, 0.721] | 30–60 s |
| `gin_pos` | 0.691–0.703 | 0.700 [0.661, 0.740] | 20–40 s |
| EGNN | **0.007–0.017** | 0.013 [0.012, 0.014] | 60–125 s |

the EGNN's seed spread (a factor of about 2 within one run, e.g. 0.007 / 0.013 / 0.017) is much wider than its bootstrap interval (±7%): which seed you train matters more than which test set you draw.

paired differences (seed 0, typical of every run): every model beats the per-atom mean (GCN about −0.22 [−0.26, −0.18], GIN −0.36 [−0.40, −0.31], `gin_pos` −0.34 [−0.39, −0.29], EGNN −1.03 [−1.08, −0.98]), and GIN − EGNN is about 0.67 [0.64, 0.71]. `gin_pos` − GIN is **not** reliably different from zero: 0.020 [0.000, 0.039], 0.016 [−0.003, 0.035] and 0.009 [−0.010, 0.028] in three runs.

learning curves (test MAE; mean over seeds, with the per-seed values from one run in brackets):

| model | 250 | 1,000 | 4,000 |
|---|---|---|---|
| per-atom mean | 1.042 | 1.039 | 1.040 |
| GCN | 0.884 (0.884 / 0.882 / 0.885) | 0.865 (0.867 / 0.864 / 0.865) | 0.827 (0.823 / 0.835 / 0.823) |
| GIN | 0.701 (0.702 / 0.704 / 0.698) | 0.684 (0.684 / 0.683 / 0.685) | 0.684 (0.684 / 0.684 / 0.684) |
| `gin_pos` | 0.979 (0.995 / 0.975 / 0.967) | 0.750 (0.737 / 0.773 / 0.741) | 0.699 (0.700 / 0.702 / 0.694) |
| EGNN | 0.716 (0.708 / 0.669 / 0.770) | 0.118 (0.180 / 0.068 / 0.104) | 0.012 (0.013 / 0.007 / 0.017) |

at 1,000 molecules the EGNN's seeds are far apart: 0.068–0.180 in this run and 0.055–0.342 in the reviewer's. that's the size at which it learns the steep repulsive wall in some runs and not yet in others. with 250 molecules it's no better than GIN; with 4,000 every seed has broken through, while GIN is flat from 1,000 molecules on and `gin_pos` approaches GIN from above.

the invariance audit (largest change in predicted energy, float64, 100 test molecules, 3 random transforms each; one run, the others agree to the digits shown except `gin_pos`, whose values vary between runs):

| model | rotation | reflection | translation | permutation |
|---|---|---|---|---|
| GCN | 0 | 0 | 0 | 2.7e-15 |
| GIN | 0 | 0 | 0 | 2.7e-15 |
| `gin_pos` | 1.7 | 1.6 | 5.6 | 3.6e-15 |
| EGNN | 2.6e-14 | 2.5e-14 | 3.4e-14 | 3.6e-15 |

## analysis questions

1. **what do GCN and GIN learn without coordinates?** both beat the baseline. what information about the energy is in the radius graph alone (count contacts within $r_c$), and what isn't (how close each contact is)? why is GIN better than GCN here (topic 04), and why do three GIN seeds give the same MAE to within 0.002?
2. **why does `gin_pos` not beat GIN?** it sees strictly more information. connect your answer to its audit row: how much do its predictions move when a molecule is rotated, compared with its test error? what would rotation augmentation, or a canonical frame, change? (topic 05's lab is the template.)
3. **where does EGNN's advantage come from?** the energy is a sum over pairs of a function of distance; which part of the EGNN layer can represent that exactly? why does its learning curve keep falling when the others flatten?
4. **which differences are real?** use paired intervals and the seed spread. is GIN better than `gin_pos` (the reference's paired intervals straddle zero in some runs and not others)? are the EGNN seeds (a factor of 2 apart at 4,000 molecules, up to 6 at 1,000) different outcomes, or noise? compare the seed spread with the bootstrap interval.
5. **what does the audit prove, and what doesn't it?** zero change under 3 random rotations of 100 molecules: a proof, evidence, or neither? what kind of bug would it miss?
6. **QM9.** what would change on real molecules (bonds as edges, long-range electrostatics, many more atom types, noise-free but approximate DFT labels)? which of your conclusions would you expect to survive?

## stretch goals

- **forces.** differentiate your EGNN's energy with respect to the input positions (`torch.autograd.grad`) and check numerically that the forces rotate with the molecule. compare them with the analytic Lennard-Jones forces.
- **give GIN the distances.** feed squared distances (or a radial basis expansion of them) to GIN as edge features, in a GIN variant you write. how close does it get to EGNN? is it still exactly invariant?
- **beyond 1-WL.** add a second task that message passing can't solve from topology, e.g. counting 3-cycles in the radius graph, or a chirality label (the signed volume of the first four atoms). which models can learn it, and why (topics 04 and 05)?
- **sample efficiency, precisely.** fit a power law to each learning curve and report the exponent with uncertainty over seeds.

## self-assessment

grade your write-up with [`rubric.md`](rubric.md).

## references

- Ramakrishnan, Dral, Rupp & von Lilienfeld (2014), "Quantum chemistry structures and properties of 134 kilo molecules", *Scientific Data* 1: 140022. [doi:10.1038/sdata.2014.22](https://doi.org/10.1038/sdata.2014.22)
- Jones (1924), "On the determination of molecular fields. II. From the equation of state of a gas", *Proceedings of the Royal Society A* 106(738): 463–477. the Lennard-Jones potential. [doi:10.1098/rspa.1924.0082](https://doi.org/10.1098/rspa.1924.0082)
- Satorras, Hoogeboom & Welling (2021), "E(n) equivariant graph neural networks", *ICML*. [arXiv:2102.09844](https://arxiv.org/abs/2102.09844)
- Xu, Hu, Leskovec & Jegelka (2019), "How powerful are graph neural networks?", *ICLR*. [arXiv:1810.00826](https://arxiv.org/abs/1810.00826)
- Gilmer, Schoenholz, Riley, Vinyals & Dahl (2017), "Neural message passing for quantum chemistry", *ICML*. GNNs on QM9. [arXiv:1704.01212](https://arxiv.org/abs/1704.01212)
