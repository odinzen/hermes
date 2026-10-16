"""The documentation names only tests that exist and cites only references the reference check covers."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def checked_identifiers():
    tree = ast.parse((ROOT / "tools" / "check_references.py").read_text(encoding="utf-8"))
    (node,) = [n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "REFERENCES"]
    return {entry.elts[0].value.lower() for entry in node.value.elts}


def test_validation_page_names_only_existing_tests():
    page = (ROOT / "docs" / "validation.md").read_text(encoding="utf-8")
    named = re.findall(r"`(tests/[\w/]+\.py)::(\w+)`", page)
    assert named
    for path, name in named:
        source = (ROOT / path).read_text(encoding="utf-8")
        assert re.search(rf"^def {name}\(", source, re.M), f"{path}::{name} does not exist"


def test_every_cited_doi_is_checked():
    checked = checked_identifiers()
    for doc in ("README.md", "docs/validation.md"):
        for doi in re.findall(r"https://doi\.org/(\S+)", (ROOT / doc).read_text(encoding="utf-8")):
            doi = doi.rstrip(".")
            while doi.endswith(")") and doi.count(")") > doi.count("("):  # a DOI may hold balanced parentheses
                doi = doi[:-1]
            assert doi.lower() in checked, f"{doc} cites {doi}, which tools/check_references.py does not check"
