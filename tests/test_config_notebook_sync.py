"""Guards configs/{classifier,extractor}.yaml against silently drifting from the notebook cell
that actually trains with those values. Kaggle notebooks can't import this repo, so the config
and the notebook can only be kept in sync by convention -- this test makes that convention a
checked invariant instead of just a comment.
"""
import ast
import json
import re
from pathlib import Path

import yaml

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


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
