from convert.expr import alteryx_expr_to_spark


def test_translates_single_field_ref() -> None:
    assert alteryx_expr_to_spark("[Amount] > 0") == "`Amount` > 0"


def test_translates_multiple_field_refs() -> None:
    assert alteryx_expr_to_spark("[Amount] * [Quantity]") == "`Amount` * `Quantity`"


def test_field_names_with_spaces() -> None:
    assert alteryx_expr_to_spark("[Total Amount] > 0") == "`Total Amount` > 0"


def test_no_field_refs_passthrough() -> None:
    assert alteryx_expr_to_spark("1 = 1") == "1 = 1"
