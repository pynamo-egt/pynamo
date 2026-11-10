import re
import tomllib
from pathlib import Path

import pynamo_egt as pn

ROOT = Path(__file__).resolve().parents[1]


def test_version_is_consistent_across_metadata_package_and_citation():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    citation_version = re.search(r"^version:\s*(\S+)\s*$", citation, re.MULTILINE).group(1)

    assert pyproject["project"]["version"] == pn.__version__ == citation_version
