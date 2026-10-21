# thesis project · hints

one hint per stage. at this level the spec in the lesson is the main help; open a hint only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

`topo_order` is topic 01's `reading_order` with `after` in place of `cites`. the critical path needs two passes over that order: forwards for the earliest start and finish, backwards (through each task's successors) for the latest finish. slack is the difference, and the path is found by walking back from the task that finishes last.

</details>

## stage 2

<details><summary>hint</summary>

the same two passes, but on arrays of simulations instead of numbers: every "max" and "min" becomes its elementwise NumPy version, and no loop over simulations is needed. for reproducibility, draw the durations in the order the tasks are given, not in the order you process them.

</details>

## stage 3

<details><summary>hint</summary>

the only new part is where the durations come from: each task, in each simulation, gets its own ratio drawn from your history. after that, stage 2 does the rest.

</details>
