"""Guards config values against silently drifting from whatever actually produced them:
- configs/{classifier,extractor}.yaml's training hyperparameters vs. the notebook cell that
  actually trains with those values (Kaggle notebooks can't import this repo, so this can only
  be kept in sync by convention otherwise).
- configs/calibration.yaml's fitted temperature/model_revision vs. the results/*.json a
  calibration run actually produced (same drift risk, different source of truth).
Both are the same class of bug this project has hit more than once (a hand-copied value with
nothing checking it stays copied correctly) -- this file makes both checked invariants instead
of just a comment.
"""
import ast
import json
import re
from pathlib import Path

import pytest
import yaml

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _notebook_source(notebook_name: str) -> str:
    nb = json.loads((NOTEBOOKS_DIR / notebook_name).read_text())
    return "\n".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    )


def _training_arguments_call_source(source: str) -> str:
    """Slice out just the TrainingArguments(...) call, matching balanced parens.

    Without this, `_value_in_notebook` would regex-search the *entire* notebook for the first
    `key=value`-looking text anywhere -- e.g. an unrelated `dataset.shuffle(seed=42)` earlier in
    the notebook would silently be matched instead of the real training config, and the test
    would validate against the wrong value while still reporting green.
    """
    start = source.index("TrainingArguments(") + len("TrainingArguments(") - 1
    depth = 0
    for i in range(start, len(source)):
        if source[i] == "(":
            depth += 1
        elif source[i] == ")":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
    raise AssertionError("unterminated TrainingArguments(...) call in notebook source")


def _value_in_notebook(source: str, key: str):
    """Literal value assigned to `key=...` within the given source (already scoped to the
    TrainingArguments(...) call by _training_arguments_call_source)."""
    match = re.search(rf"\b{re.escape(key)}\s*=\s*([^,\n)]+)", source)
    assert match, f"no `{key}=...` assignment found in the TrainingArguments(...) call"
    return ast.literal_eval(match.group(1).strip())


def _assert_train_config_matches_notebook(config_name: str, notebook_name: str):
    config = yaml.safe_load((CONFIGS_DIR / config_name).read_text())["train"]
    source = _training_arguments_call_source(_notebook_source(notebook_name))
    for key, expected in config.items():
        actual = _value_in_notebook(source, key)
        assert actual == expected, (
            f"{config_name}'s {key}={expected!r} doesn't match {notebook_name}'s actual "
            f"{key}={actual!r} -- the config and the notebook's real TrainingArguments have "
            "drifted apart"
        )


def test_training_arguments_call_source_ignores_earlier_decoy_assignment():
    # Reproduces the exact failure mode the scoping fix guards against: an unrelated `seed=`
    # assignment earlier in the notebook (e.g. a dataset shuffle) must not be picked up instead
    # of the real training call.
    decoy_source = (
        'dataset = dataset.shuffle(seed=999)\n'
        'args = TrainingArguments(output_dir="results/x", seed=42)\n'
    )
    call_source = _training_arguments_call_source(decoy_source)
    assert _value_in_notebook(call_source, "seed") == 42


def test_classifier_config_matches_training_notebook():
    _assert_train_config_matches_notebook(
        "classifier.yaml", "01_train_classifier_rvl_cdip_mini.ipynb"
    )


def test_extractor_config_matches_training_notebook():
    _assert_train_config_matches_notebook(
        "extractor.yaml", "02_train_extractor_funsd.ipynb"
    )


def test_extraction_calibration_matches_calibration_script_output():
    # scripts/calibrate_extractor.py wrote both configs/calibration.yaml's `extraction:` block
    # and results/extractor_calibration_summary.json from the same run -- this catches either
    # one being hand-edited (or re-run) without updating the other.
    calibration = yaml.safe_load((CONFIGS_DIR / "calibration.yaml").read_text())["extraction"]
    summary = json.loads((RESULTS_DIR / "extractor_calibration_summary.json").read_text())
    assert calibration["model_revision"] == summary["model_revision"]
    assert calibration["temperature"] == pytest.approx(summary["temperature"], rel=1e-3)


def test_classifier_calibration_temperature_matches_summary():
    # Only `temperature` is checked here, not `model_revision` -- unlike the extraction summary
    # above, results/calibration_summary.json predates the model_revision concept: it was produced
    # by an earlier run of notebooks/03_calibrate_classifier.ipynb, before that notebook captured
    # the model's Hub commit hash. The notebook now captures `model.config._commit_hash` into the
    # summary for any *future* run (see its model-loading and summary-saving cells), which will
    # let this check extend to model_revision too once that notebook is re-run and its output
    # committed -- until then, this is a real but intentionally partial guard, not a design flaw.
    calibration = yaml.safe_load((CONFIGS_DIR / "calibration.yaml").read_text())["classifier"]
    summary = json.loads((RESULTS_DIR / "calibration_summary.json").read_text())
    assert calibration["temperature"] == pytest.approx(summary["temperature"], rel=1e-3)
