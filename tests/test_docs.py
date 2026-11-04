import re
from pathlib import Path

import pynamo_egt as pn

API_REFERENCE = Path(__file__).resolve().parents[1] / "docs" / "api.rst"


def test_api_reference_lists_every_public_name():
    listed = set(re.findall(r"^\s{3}(\w+)\s*$", API_REFERENCE.read_text(encoding="utf-8"), re.MULTILINE))
    assert set(pn.__all__) <= listed, sorted(set(pn.__all__) - listed)
