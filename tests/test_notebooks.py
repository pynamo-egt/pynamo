import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted(ROOT.glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[path.name for path in NOTEBOOKS])
def test_notebooks_are_stored_without_outputs(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    dirty = [
        index
        for index, cell in enumerate(notebook["cells"])
        if cell["cell_type"] == "code" and (cell.get("outputs") or cell.get("execution_count"))
    ]
    assert not dirty, (
        f"{path.name} has outputs in cells {dirty}; strip them with "
        f"`pip install nbstripout && nbstripout {path.name}`"
    )
