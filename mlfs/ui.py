"""Notebook UI for the learning loop: predict-before-you-see gates.

In a lab notebook:

    p1 = ui.predict("with all weights initialized to zero, what will the loss do?")
    p1                      # cell 1: shows a text box + "lock in" button

    ui.gate(p1)             # cell 2: the experiment waits until a prediction is locked in
    run_experiment()

Run headless (e.g. `python lab.py` in CI), the gate is skipped so experiments always run.
"""
from __future__ import annotations


def predict(question: str, *, button: str = "lock in my prediction"):
    import marimo as mo

    return mo.ui.text_area(
        placeholder="write your prediction (and why) before running anything…",
        label=f"🔮 **predict:** {question}",
        full_width=True,
    ).form(submit_button_label=button)


def gate(prediction) -> None:
    """Stop the calling cell until `prediction` (from `predict`) has been submitted."""
    import marimo as mo

    mo.stop(
        mo.running_in_notebook() and prediction.value is None,
        mo.callout(mo.md("lock in a prediction above to run this experiment."), kind="info"),
    )
