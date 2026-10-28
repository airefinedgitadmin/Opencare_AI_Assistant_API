from careplan_ai.library.anonymise import anonymise


def test_story_scenario_name_and_room_become_placeholders() -> None:
    # OPENCARE-51: "Given a care plan mentions 'Cathy' and 'Room 12'..."
    text = "Cathy Smith in Room 12 needs prompting. Cathy's daughter Jane visits on 03/10/2026."

    result = anonymise(text, resident_names=["Cathy Smith"], other_names=["Jane"])

    assert "Cathy" not in result and "Smith" not in result and "Jane" not in result
    assert "12" not in result and "2026" not in result
    assert result == (
        "[Resident] in [Room] needs prompting. [Resident]'s daughter [Person] visits on [Date]."
    )


def test_identifiers_are_replaced_but_care_figures_are_kept() -> None:
    text = (
        "DOB: 01/02/1940, CHI 0102401234, tel 0141 555 1234, j.doe@example.com, G51 4TF. "
        "Offer 200ml fluids 3 times a day."
    )

    result = anonymise(text)

    for secret in ("1940", "0102401234", "555", "example.com", "G51"):
        assert secret not in result
    assert "200ml fluids 3 times a day" in result
