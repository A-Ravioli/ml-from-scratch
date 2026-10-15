# 06 · capstone rubric

score each criterion 0–3. a solid capstone scores at least 9 out of 12, with no zeros.

| criterion | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **correctness** | tests fail | most tests pass | all tests pass, and the full config runs end to end | all pass, plus your own checks (e.g. your counted costs against a hand count for one run of each method, `logistic_objective` against `scipy.optimize.check_grad`, and a sanity check that every method reaches the same $f^*$) |
| **results** | no numbers | final losses only | cost to reach several tolerances for every method on every problem, iterations and seconds, and the minibatch methods over several seeds with their spread | the same, plus a baseline that makes the numbers meaningful (e.g. plain gradient descent with its best fixed step) and a learning-rate sensitivity sweep for at least two methods |
| **analysis** | none | describes the plots | explains each ranking with the chapter's theory: $\kappa$ versus $\sqrt\kappa$, quadratic and superlinear convergence, preconditioning, local minima versus saddles | also identifies where the theory *doesn't* predict the outcome (e.g. L-BFGS beating BFGS, Newton crawling on gradient fallbacks for hundreds of steps on 10-D Rosenbrock, Adam failing on raw features) and gives a tested explanation |
| **clarity** | hard to follow | readable | labelled figures with cost on the x-axis and a report table; all five prompts answered | a reader could reproduce every number from the write-up (config, seeds, learning rates, tolerances), and the recommendations follow from specific numbers |

## what graders look for

- **a fair cost measure.** iterations hide everything that happens inside an iteration (line searches, two gradients for Nesterov, a Hessian for Newton). a good report compares methods at equal *work*, and says what one unit of work is.
- **conditioning, named.** the raw-versus-standardized contrast is the single biggest effect in the benchmark. the best write-ups connect it to the condition number of the Hessian, explain why Newton is immune (affine invariance) and L-BFGS mostly so, and note that standardizing is a diagonal preconditioner, which is also what Adam tries to learn.
- **seeds and spread.** a minibatch method's result is a distribution. one run per method is an anecdote.
- **honest failures.** every method fails somewhere in this benchmark. a report that only shows successes hasn't looked.
- **recommendations with reasons.** "use L-BFGS" is a slogan; "for this 9-parameter logistic regression Newton is cheapest at every tolerance (61 gradient-equivalents to $10^{-8}$), but its cost grows like $d^3$" is a recommendation.
