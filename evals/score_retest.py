"""Score Open Care Buddy care plan drafts against the retest document's expected care.

Copy a draft from the Care Plan screen (its "action:", "Outcome:" and "source:" blocks, exactly
as shown) into a text file named <resident id>_<pathway>.txt, for example 81362_falls.txt.
Pathways: falls, must, bladder_bowel, oral, choking, waterlow. Then run:

    python evals/score_retest.py <folder or files...>            # a report per draft + summary
    python evals/score_retest.py --list                          # the file names expected

Standard library only. The expected care is in retest_18-11-2026.json next to this file; it
says how a match works. The score is a guide for the reviewer, not a clinical sign-off: a missed
item means "check the draft", and a must-not hit means the draft repeats out-of-date care.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

EXPECTED = Path(__file__).with_name("retest_18-11-2026.json")
_NAME = re.compile(r"^(\d+)_([a-z_]+)$")
_SECTION = re.compile(r"^\s*(actions?|outcomes?|sources?)\s*:?\s*$", re.IGNORECASE)
_NEGATION = re.compile(
    r"\b(?:no|not|never|no\s+longer|avoid|instead\s+of|stop|don't|do\s+not|rather\s+than)\b"
    # ... in the same clause: "do not give a normal diet or thin fluids", not "do not rush her;
    # she walks independently".
    r"[^,.;:]{0,60}$"
)


def normalise(text: str) -> str:
    text = re.sub("[‐‑‒–—]", "-", text)
    return text.replace("’", "'").casefold()


def _term(alternatives: str) -> re.Pattern[str]:
    words = [a.strip() for a in alternatives.split("/") if a.strip()]
    body = "|".join(r"\s+".join(re.escape(w) for w in a.split()) for a in words)
    return re.compile(r"(?<![a-z0-9])(?:" + body + ")")


def find(pattern: str, line: str) -> int | None:
    """Where `pattern` ("a + b/c") matches in `line` (normalised), or None."""
    starts = []
    for part in pattern.split(" + "):
        m = _term(normalise(part)).search(line)
        if m is None:
            return None
        starts.append(m.start())
    return min(starts)


def matches(patterns: list[str], lines: list[str]) -> str | None:
    """The first line one of `patterns` matches."""
    for line in lines:
        if any(find(p, line) is not None for p in patterns):
            return line
    return None


@dataclass
class Draft:
    actions: list[str] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)
    sources: str = ""


def parse_draft(text: str) -> Draft:
    """The action, outcome and source blocks as copied from the screen."""
    draft = Draft()
    section = None
    sources: list[str] = []
    for raw in text.splitlines():
        heading = _SECTION.match(raw)
        if heading:
            section = heading.group(1).casefold()[0]
            continue
        line = raw.strip()
        if not line:
            continue
        if section == "a":
            draft.actions.append(normalise(line))
        elif section == "o":
            draft.outcomes.append(normalise(line))
        elif section == "s":
            sources.append(normalise(line))
    draft.sources = "\n".join(sources)
    return draft


@dataclass
class Score:
    key: str
    title: str
    actions_found: list[str]
    actions_missed: list[str]
    outcomes_found: list[str]
    outcomes_missed: list[str]
    must_not_hits: list[str]
    sources_missed: list[str]
    too_many_actions: str | None

    @property
    def found(self) -> int:
        return len(self.actions_found) + len(self.outcomes_found)

    @property
    def total(self) -> int:
        return self.found + len(self.actions_missed) + len(self.outcomes_missed)

    @property
    def passed(self) -> bool:
        return (not self.actions_missed and not self.outcomes_missed and not self.must_not_hits
                and not self.sources_missed and self.too_many_actions is None)


def _must_not_hit(rule: dict, draft: Draft) -> str | None:
    scope = rule.get("scope", "both")
    lines = (draft.actions if scope != "outcomes" else []) + (
        draft.outcomes if scope != "actions" else [])
    unless = rule.get("unless")
    if unless and matches([unless], draft.actions + draft.outcomes):
        return None
    for line in lines:
        for pattern in rule["any"]:
            at = find(pattern, line)
            if at is not None and not _NEGATION.search(line[:at]):
                return line
    return None


def score(expected: dict, draft: Draft, key: str = "") -> Score:
    def split(items: list[dict], lines: list[str]) -> tuple[list[str], list[str]]:
        found, missed = [], []
        for item in items:
            (found if matches(item["any"], lines) else missed).append(item["name"])
        return found, missed

    a_found, a_missed = split(expected["actions"], draft.actions)
    o_found, o_missed = split(expected["outcomes"], draft.outcomes)
    hits = []
    for rule in expected.get("must_not", []):
        line = _must_not_hit(rule, draft)
        if line is not None:
            hits.append(f"{rule['name']}: \"{line}\"")
    limit = expected.get("max_actions")
    too_many = (f"{len(draft.actions)} actions for a low-risk pathway (expected {limit} or fewer)"
                if limit and len(draft.actions) > limit else None)
    sources_missed = [s for s in expected.get("sources", []) if s not in draft.sources]
    if not draft.sources and expected.get("sources"):
        sources_missed = ["(no source block in the file)"]
    return Score(key, expected["title"], a_found, a_missed, o_found, o_missed, hits,
                 sources_missed, too_many)


def load_expected(path: Path = EXPECTED) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_for(data: dict, resident: str, pathway: str) -> dict | None:
    return data["residents"].get(resident, {}).get("pathways", {}).get(pathway)


def report(s: Score, name: str) -> str:
    out = [f"== {name}: {s.title} - {s.found}/{s.total} expected items"
           + (" - PASS" if s.passed else "")]
    out.append(f"  Actions  {len(s.actions_found)}/{len(s.actions_found) + len(s.actions_missed)}")
    out += [f"    missing: {m}" for m in s.actions_missed]
    out.append(
        f"  Outcomes {len(s.outcomes_found)}/{len(s.outcomes_found) + len(s.outcomes_missed)}")
    out += [f"    missing: {m}" for m in s.outcomes_missed]
    out += [f"  MUST NOT: {h}" for h in s.must_not_hits]
    if s.too_many_actions:
        out.append(f"  Light only: {s.too_many_actions}")
    if s.sources_missed:
        out.append(f"  Sources missing: {', '.join(s.sources_missed)}")
    return "\n".join(out)


def _files(args: list[str]) -> list[Path]:
    paths: list[Path] = []
    for arg in args:
        p = Path(arg)
        paths += sorted(p.glob("*.txt")) if p.is_dir() else [p]
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", help="draft files or folders of them")
    parser.add_argument("--list", action="store_true", help="list the draft file names expected")
    parser.add_argument("--expected", default=str(EXPECTED), help="expected care JSON")
    args = parser.parse_args(argv)
    data = load_expected(Path(args.expected))
    if args.list:
        for rid, resident in data["residents"].items():
            for key, p in resident["pathways"].items():
                print(f"{rid}_{key}.txt  {resident['name']} - {p['title']}")
        return 0
    scores: list[tuple[str, Score]] = []
    for path in _files(args.paths):
        m = _NAME.match(path.stem)
        expected = expected_for(data, m.group(1), m.group(2)) if m else None
        if expected is None:
            print(f"?? {path.name}: name it <resident id>_<pathway>.txt (see --list)")
            continue
        s = score(expected, parse_draft(path.read_text(encoding="utf-8")), path.stem)
        scores.append((path.stem, s))
        print(report(s, path.stem) + "\n")
    if scores:
        found = sum(s.found for _, s in scores)
        total = sum(s.total for _, s in scores)
        passed = sum(s.passed for _, s in scores)
        hits = sum(len(s.must_not_hits) for _, s in scores)
        print(f"Summary: {len(scores)} drafts, {passed} pass, {found}/{total} expected items "
              f"({100 * found // max(total, 1)}%), {hits} must-not hits")
        print("\nPathway            Score")
        for name, s in sorted(scores, key=lambda x: x[1].found / max(x[1].total, 1)):
            print(f"  {name:<18} {s.found}/{s.total}"
                  + ("  must-not!" if s.must_not_hits else ""))
    return 0 if all(s.passed for _, s in scores) else 1


if __name__ == "__main__":
    sys.exit(main())
