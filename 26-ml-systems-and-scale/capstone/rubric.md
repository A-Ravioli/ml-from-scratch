# 26 · capstone rubric

score each criterion 0–3. a solid capstone scores at least 12 out of 15, with no zeros.

| criterion | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **correctness** | tests fail | most tests pass | all tests pass | all pass, plus your own checks: the optimized model reaches the same validation loss as the baseline from the same initialization, cached generation still works after `fused_attention`, and the fp16 run's skipped steps are reported |
| **results** | no comparison | a speedup with no baseline measured on the same machine | time to the target loss for baseline and optimized runs, back to back, with the spread over at least 3 seeds of the optimized run | as 2, plus a second baseline run (or a rerun of the whole race) showing how much the machine's noise moves the headline number |
| **profiling and ablation** | no profile | a profile, or an ablation, without intervals | a profile of the baseline step that names where the time goes, and an ablation (each optimization alone, all, all minus one) with bootstrap intervals | as 2, plus each optimization's speedup explained with topic 01's model (which ops it removes bytes, launches or FLOPs from), and one optimization you tried that didn't help, with the measurement |
| **analysis** | none | describes the numbers | separates the systems speedup (same steps, faster) from the optimization speedup (fewer steps), and says what limits further gains on your hardware, with an MFU estimate | as 2, plus a defensible prediction of what the same optimizations would give on different hardware (an NVIDIA GPU with tensor cores, or the CPU), checked where you can |
| **clarity** | hard to follow | readable | a results table, the loss-versus-wall-clock plot and a narrative | a peer could rerun every number from the write-up and the config alone, and would know which differences are within noise |
