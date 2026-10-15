# 16 · capstone rubric

score each criterion 0–3. a solid capstone scores at least 9 out of 12, with no zeros.

| criterion | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **correctness** | the pipeline doesn't run end to end | most checks pass | all checks pass, and the full config reaches $\ge 90\%$ test accuracy on real CIFAR-10 with your own ResNet | all of that, with augmentation done on whole batches on the device, no test-set tuning anywhere, and a pipeline another person could rerun from the notebook alone |
| **results** | no test number | a test accuracy without an interval or baseline | the headline accuracy with its 95% interval, the majority-class baseline, runtime and hardware, and the full ablation table with paired intervals | all of that plus at least two seeds of some configuration, and a statement of which ablation effects are larger than the seed-to-seed noise (not just the test set's) |
| **analysis** | rows of numbers without comment | each ablation row described | each row's *mechanism* explained (why removing augmentation, the schedule or the shortcuts costs what it costs at this budget), with one curve that shows it | all of that plus a discussion of how the effects would change with a longer schedule or a bigger network, and a costed plan for getting further |
| **clarity** | hard to follow | readable | one table per question, labelled figures, numbers with sensible precision | a reader could reproduce any number (config, seed, epochs, device, minutes) from the write-up alone |

## what graders look for

- **an honest headline.** "0.929 [0.924, 0.934] after 20 epochs, 35 minutes on an M2 GPU, majority class 0.100" beats "93%". if you didn't reach 90%, say so, show where your curve was heading, and what you'd change first.
- **noise, both kinds.** the paired bootstrap interval measures the test set's sampling noise for *these* trained models. retraining with another seed adds more. an ablation effect of one point is only interesting if it beats both.
- **mechanisms, not just signs.** "the constant learning rate cost 17 points" is a measurement; "at lr 0.2 SGD keeps bouncing around the noise floor (ch 15 · topic 04), and the cosine decay is what lets it settle; the training loss at the end shows it" is an explanation.
- **budget-awareness.** ingredients interact with the budget: regularizers (augmentation, weight decay, label smoothing) matter more the longer you train, residual connections matter more the deeper the network. the best write-ups say which conclusions they expect to survive a 200-epoch run and which they don't.
- **no tuning on the test set.** comparing the six ablation rows on the test set is a measurement. picking a learning rate, an epoch count or an augmentation strength by looking at test accuracy and then reporting that test accuracy is not; if you tune anything, hold out a validation split first.
