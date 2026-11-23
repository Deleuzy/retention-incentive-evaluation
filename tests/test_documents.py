"""The generated documents match the run outputs and follow the writing rules."""

import re

from rie import report
from rie.data import ROOT

PROSE = [
    ROOT / "README.md",
    ROOT / "brief" / "beslutningsoplaeg.md",
    ROOT / "brief" / "beslutningsoplaeg.html",
    ROOT / "design" / "pre_analysis_plan.md",
    ROOT / "causal" / "RESULTS.md",
]


def test_documents_are_up_to_date():
    for path, text in report.documents().items():
        assert path.read_text() == text, f"{path} is stale; run python -m rie.report"


def test_no_en_or_em_dashes_in_prose():
    for p in PROSE:
        lines = p.read_text().splitlines()
        bad = [i + 1 for i, line in enumerate(lines) if re.search("[\u2013\u2014]", line)]
        assert not bad, f"{p.name} lines {bad}"


def test_brief_has_three_separated_parts_and_decision_fields():
    text = (ROOT / "brief" / "beslutningsoplaeg.md").read_text()
    heads = ["Hvad data viser", "Sandsynlige forklaringer", "Hvad vi endnu ikke har belæg for"]
    pos = [text.index(h) for h in heads]
    assert pos == sorted(pos)
    for field in ("Forventet effekt", "Ressourcebehov", "Gennemførlighed", "Risiko"):
        assert field in text
    assert "simulerede" in text


def test_brief_numbers_come_from_the_run():
    d = report.load()
    text = (ROOT / "brief" / "beslutningsoplaeg.md").read_text()
    assert report.dk(d["summary"]["benchmark"]) in text
    assert report.dk(d["pilot"]["n_pilot"]) in text
    assert report.dk(d["pilot"]["primary"]["effect_pp"], 1) in text


def test_simulated_material_is_labelled():
    pap = (ROOT / "design" / "pre_analysis_plan.md").read_text()
    assert "HYPOTHETICAL" in pap and "SIMULATED" in pap
    assert "simulated" in (ROOT / "design" / "simulated_pilot_data.csv").open().readline()
