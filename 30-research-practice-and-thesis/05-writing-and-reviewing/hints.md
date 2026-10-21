# writing & reviewing · hints

one hint per stage. at this level the spec in the lesson is the main help; open a hint only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

a matplotlib figure is a tree you can walk: `fig.axes`, and in each axes its lines, collections, containers, patches, images and texts, each of which knows its label, font size, colormap or transform. decide first which artists count as data (the spec's definitions), then each rule is a short check over them.

</details>

## stage 2

<details><summary>hint</summary>

strip comments first (an escaped `\%` isn't one), then one regular expression per kind of command. environments need a stack, not a regex. for floats, find each environment's body and compare the positions of its `\caption` and its `\label`.

</details>

## stage 3

<details><summary>hint</summary>

each criterion is one boolean expression once you have two helpers: one that turns text into lowercase word tokens, one that turns tokens into a set of 4-grams. the consistency criterion depends on the scores being valid, so compute it last.

</details>

## stage 4

<details><summary>hint</summary>

turn each committee's scores into a boolean "accepted" mask with a stable sort; agreement, flip rate and kappa are then means of boolean comparisons plus the chance-agreement formula.

</details>
