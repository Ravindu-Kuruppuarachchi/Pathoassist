"""Nottingham (Elston-Ellis) histological grade from its three components.

Each component is 1-3: tubule formation, nuclear pleomorphism, mitotic count. Total 3-5 -> grade 1,
6-7 -> grade 2, 8-9 -> grade 3. A component can come from the assistant ("auto") or be entered /
overridden by the pathologist ("manual"); every change is written to the audit log.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

COMPONENTS = ("tubules", "pleomorphism", "mitoses")
LABELS = {"tubules": "Tubule formation", "pleomorphism": "Nuclear pleomorphism", "mitoses": "Mitotic count"}


def grade_from_total(total: int) -> int:
    if not 3 <= total <= 9:
        raise ValueError(f"Nottingham total must be 3-9, got {total}")
    return 1 if total <= 5 else (2 if total <= 7 else 3)


@dataclass
class Component:
    name: str
    score: int | None = None          # final score used for the grade
    auto_score: int | None = None     # what the assistant proposed
    source: str = "missing"           # "auto", "manual", "edited" or "missing"
    value: str = ""                   # human-readable measurement, e.g. "tubules 12% of tumour"
    details: dict = field(default_factory=dict)
    validated: bool = False           # was the automatic method validated locally?

    @property
    def edited(self):
        return self.source == "edited"


class NottinghamGrade:
    def __init__(self, audit=None):
        self.c = {k: Component(k) for k in COMPONENTS}
        self.audit = audit

    def set_auto(self, name, score: int, value: str = "", validated: bool = False, details: dict | None = None):
        self._check(name, score)
        comp = self.c[name]
        comp.auto_score, comp.score, comp.source, comp.value, comp.details, comp.validated = score, score, "auto", value, details or {}, validated
        if self.audit:
            self.audit.log("component_auto", component=name, score=score, value=value, validated=validated)

    def set_manual(self, name, score: int, reason: str = ""):
        """Pathologist enters a component directly, or overrides the automatic score."""
        self._check(name, score)
        comp = self.c[name]
        old = comp.score
        comp.score = score
        comp.source = "edited" if comp.auto_score is not None and score != comp.auto_score else ("auto" if comp.auto_score == score else "manual")
        if self.audit:
            self.audit.log("component_manual", component=name, old=old, new=score, auto=comp.auto_score, reason=reason)

    @staticmethod
    def _check(name, score):
        if name not in COMPONENTS:
            raise KeyError(name)
        if score not in (1, 2, 3):
            raise ValueError(f"{name} score must be 1, 2 or 3")

    @property
    def complete(self):
        return all(c.score is not None for c in self.c.values())

    @property
    def total(self):
        return sum(c.score for c in self.c.values()) if self.complete else None

    @property
    def grade(self):
        return grade_from_total(self.total) if self.complete else None

    def as_dict(self):
        return dict(components={k: asdict(v) for k, v in self.c.items()}, total=self.total, grade=self.grade)
