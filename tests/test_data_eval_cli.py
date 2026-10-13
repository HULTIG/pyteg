from pathlib import Path

import pytest

from pyteg import Pair, binary_report, direction_accuracy, load_jsonl, load_rte_xml, load_tsv, save_jsonl
from pyteg.cli import main

FIX = Path(__file__).parent / "fixtures"


def test_rte1_format():
    pairs = load_rte_xml(FIX / "rte1_sample.xml", source="rte1_test")
    assert [p.label for p in pairs] == [True, False]
    assert pairs[0].id == "rte1_test:754" and pairs[0].extra["task"] == "CD"
    assert pairs[0].t.startswith("Mexico City has")


def test_three_way_format():
    pairs = load_rte_xml(FIX / "rte4_sample.xml")
    assert [p.label for p in pairs] == [True, False, False]
    assert [p.raw_label for p in pairs] == ["ENTAILMENT", "CONTRADICTION", "UNKNOWN"]


def test_assin_format():
    pairs = load_rte_xml(FIX / "assin_sample.xml")
    assert [p.label for p in pairs] == [True, False, True]
    assert pairs[0].extra["similarity"] == "4.5"


def test_jsonl_and_tsv(tmp_path):
    pairs = load_rte_xml(FIX / "rte1_sample.xml")
    save_jsonl(pairs, tmp_path / "p.jsonl")
    assert load_jsonl(tmp_path / "p.jsonl") == pairs
    tsv = tmp_path / "p.tsv"
    tsv.write_text("id\tt\th\tlabel\na\tI ate a strawberry.\tI ate fruit.\tYES\nb\tx\ty\t\n", encoding="utf-8")
    loaded = load_tsv(tsv)
    assert loaded[0] == Pair(id="p:a", t="I ate a strawberry.", h="I ate fruit.", label=True, raw_label="YES", source="p")
    assert loaded[1].label is None


def test_binary_report():
    r = binary_report([True, True, False, False, True], [True, False, False, True, True])
    assert (r.tp, r.fp, r.fn, r.tn) == (2, 1, 1, 1)
    assert r.accuracy == pytest.approx(3 / 5)
    assert r.precision_pos == pytest.approx(2 / 3)
    assert r.precision_neg == pytest.approx(1 / 2)
    assert r.precision_macro == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert r.precision_weighted == pytest.approx((2 / 3 * 3 + 1 / 2 * 2) / 5)
    assert r.as_dict()["accuracy"] == pytest.approx(0.6)
    with pytest.raises(ValueError):
        binary_report([True], [])


def test_direction_accuracy():
    d = direction_accuracy([1.0, -0.5, 0.0, 2.0])
    assert d == {"n": 4, "accuracy": 0.5, "decided": 0.75, "accuracy_decided": pytest.approx(2 / 3)}


def test_cli_end_to_end(tmp_path, capsys):
    corpus = tmp_path / "corpus.txt"
    lines = []
    for i in range(40):
        lines.append("i ate some fruit today" + (" strawberry" if i < 10 else ""))
    for i in range(60):
        lines.append("the weather was nice today" + (" fruit" if i % 6 == 0 else ""))
    corpus.write_text("\n".join(lines) + "\n", encoding="utf-8")
    index = tmp_path / "idx.npz"
    assert main(["index", "--corpus", str(corpus), "--out", str(index)]) == 0
    assert main(["info", "--index", str(index)]) == 0
    assert main(["score", "--index", str(index), "--t", "I ate a strawberry", "--h", "I ate fruit", "--all-measures"]) == 0
    out = capsys.readouterr().out
    assert "T entails H by generality" in out
    units = tmp_path / "mwu.txt"
    assert main(["mwu", "--corpus", str(corpus), "--out", str(units), "--min-freq", "3"]) == 0


def test_cli_mwu_level_needs_units(tmp_path):
    with pytest.raises(SystemExit):
        main(["score", "--index", "x.npz", "--t", "a", "--h", "b", "--level", "mwu"])
