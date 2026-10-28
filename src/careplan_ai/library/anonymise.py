"""Replaces identifiers with placeholders before a care plan is indexed.

Deterministic, no model: names the backend supplies, plus pattern-matched
room numbers, dates, CHI/NHS numbers, phone numbers, emails and postcodes.
Numbers that aren't identifiers (doses, frequencies) are left alone.
"""

import re

_ROOM = re.compile(
    r"\b(?:room|rm|bedroom|flat|bed)\s*(?:no\.?|number|num)?\s*#?\s*\d+[a-z]?\b", re.IGNORECASE
)
_MONTHS = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
_DATE_NUMERIC = re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")
_DATE_WORDS = re.compile(
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{_MONTHS})\.?,?\s+\d{{4}}\b"
    rf"|\b(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b",
    re.IGNORECASE,
)
_DOB = re.compile(r"\b(?:dob|d\.o\.b\.?|date of birth)\s*:?\s*\[Date\]", re.IGNORECASE)
# CHI (Scotland) and NHS numbers are both 10 digits, sometimes spaced 3-3-4 or 6-4.
_HEALTH_ID = re.compile(r"\b\d{3}\s?\d{3}\s?\d{4}\b|\b\d{6}\s?\d{4}\b")
_PHONE = re.compile(r"(?:\+44\s?\(?0?\)?\s?|\b0)\d{2,4}[\s-]?\d{3,4}[\s-]?\d{3,4}\b")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_POSTCODE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.IGNORECASE)


def _name_variants(names: list[str]) -> list[str]:
    """Full names and each part of them (first name, surname), longest first,
    so "Cathy Smith" is replaced whole before "Cathy" alone."""
    variants: set[str] = set()
    for name in names:
        clean = " ".join(name.split())
        if len(clean) < 2:
            continue
        variants.add(clean)
        variants.update(part for part in clean.split(" ") if len(part) >= 2)
    return sorted(variants, key=len, reverse=True)


def _replace_names(text: str, names: list[str], placeholder: str) -> str:
    for name in _name_variants(names):
        text = re.sub(rf"\b{re.escape(name)}\b", placeholder, text, flags=re.IGNORECASE)
    return text


def anonymise(
    text: str, *, resident_names: list[str] | None = None, other_names: list[str] | None = None
) -> str:
    text = _EMAIL.sub("[Email]", text)
    text = _DATE_WORDS.sub("[Date]", text)
    text = _DATE_NUMERIC.sub("[Date]", text)
    text = _DOB.sub("[Date of birth]", text)
    text = _HEALTH_ID.sub("[ID]", text)
    text = _PHONE.sub("[Phone]", text)
    text = _POSTCODE.sub("[Postcode]", text)
    text = _ROOM.sub("[Room]", text)
    text = _replace_names(text, resident_names or [], "[Resident]")
    text = _replace_names(text, other_names or [], "[Person]")
    return text
