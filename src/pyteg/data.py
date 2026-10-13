"""Loading text-hypothesis pairs.

Supported formats:

* the XML of the PASCAL/TAC RTE challenges (RTE-1 uses ``value="TRUE"``,
  RTE-2 and RTE-3 ``entailment="YES"``, RTE-4 and RTE-5 a three-way label);
* the XML of ASSIN and ASSIN 2 (Portuguese), which uses the same layout with
  ``entailment="Entailment" | "Paraphrase" | "None"``;
* JSON Lines and tab-separated files with ``t``, ``h`` and ``label`` fields.
"""

from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

__all__ = ["Pair", "load_rte_xml", "load_jsonl", "save_jsonl", "load_tsv"]

_POSITIVE = {"yes", "true", "entailment", "paraphrase", "1"}
_NEGATIVE = {"no", "false", "contradiction", "unknown", "none", "0"}


@dataclass
class Pair:
    """One text-hypothesis pair.

    ``label`` is ``True`` for entailment, ``False`` for no entailment and
    ``None`` when the source gives no gold label. ``raw_label`` keeps the
    original string (useful for three-way data).
    """

    id: str
    t: str
    h: str
    label: Optional[bool] = None
    raw_label: Optional[str] = None
    source: str = ""
    extra: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_label(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    v = value.strip().lower()
    if v in _POSITIVE:
        return True
    if v in _NEGATIVE:
        return False
    raise ValueError(f"unrecognised entailment label {value!r}")


def load_rte_xml(path: Union[str, Path], source: Optional[str] = None) -> List[Pair]:
    """Read an RTE or ASSIN XML file."""
    path = Path(path)
    source = source or path.stem
    tree = ET.parse(path)
    pairs: List[Pair] = []
    for node in tree.getroot().iter("pair"):
        attrs = dict(node.attrib)
        raw = attrs.pop("entailment", None) or attrs.pop("value", None)
        t_node, h_node = node.find("t"), node.find("h")
        if t_node is None or h_node is None:
            raise ValueError(f"pair {attrs.get('id')} in {path} has no <t> or <h>")
        pairs.append(
            Pair(
                id=f"{source}:{attrs.pop('id', len(pairs) + 1)}",
                t=" ".join((t_node.text or "").split()),
                h=" ".join((h_node.text or "").split()),
                label=_parse_label(raw),
                raw_label=raw,
                source=source,
                extra={k: str(v) for k, v in attrs.items()},
            )
        )
    return pairs


def save_jsonl(pairs: Iterable[Pair], path: Union[str, Path]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for p in pairs:
            handle.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")


def load_jsonl(path: Union[str, Path]) -> List[Pair]:
    pairs: List[Pair] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                pairs.append(Pair(**json.loads(line)))
    return pairs


def load_tsv(path: Union[str, Path], source: Optional[str] = None) -> List[Pair]:
    """Read a tab-separated file with a header containing ``t`` and ``h``.

    Optional columns: ``id`` and ``label``.
    """
    path = Path(path)
    source = source or path.stem
    pairs: List[Pair] = []
    with open(path, encoding="utf-8", newline="") as handle:
        for i, row in enumerate(csv.DictReader(handle, delimiter="\t"), start=1):
            raw = row.get("label")
            pairs.append(
                Pair(
                    id=f"{source}:{row.get('id') or i}",
                    t=row["t"],
                    h=row["h"],
                    label=_parse_label(raw) if raw not in (None, "") else None,
                    raw_label=raw or None,
                    source=source,
                )
            )
    return pairs
