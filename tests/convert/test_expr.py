from convert.expr import alteryx_expr_to_spark, unknown_functions


def test_translates_single_field_ref() -> None:
    assert alteryx_expr_to_spark("[Amount] > 0") == "`Amount` > 0"


def test_translates_multiple_field_refs() -> None:
    assert alteryx_expr_to_spark("[Amount] * [Quantity]") == "`Amount` * `Quantity`"


def test_field_names_with_spaces() -> None:
    assert alteryx_expr_to_spark("[Total Amount] > 0") == "`Total Amount` > 0"


def test_no_field_refs_passthrough() -> None:
    assert alteryx_expr_to_spark("1 = 1") == "1 = 1"


def test_iif_renamed_to_if() -> None:
    assert alteryx_expr_to_spark("IIF([A]>0,1,0)") == "if(`A`>0,1,0)"


def test_block_if_becomes_case_when() -> None:
    result = alteryx_expr_to_spark("IF [A]>0 THEN 1 ELSEIF [A]<0 THEN -1 ELSE 0 ENDIF")
    assert result == "CASE WHEN `A`>0 THEN 1 WHEN `A`<0 THEN -1 ELSE 0 END"


def test_substring_offset_shifts_from_zero_to_one_based() -> None:
    assert alteryx_expr_to_spark("Substring([A],0,3)") == "substring(`A`, 1, 3)"


def test_string_concat_plus_becomes_double_pipe() -> None:
    assert alteryx_expr_to_spark("'x' + [A]") == "'x' || `A`"


def test_eq_null_becomes_is_null() -> None:
    assert alteryx_expr_to_spark("[A] = Null()") == "`A` IS NULL"


def test_datetimeadd_rewritten_to_databricks_dateadd() -> None:
    result = alteryx_expr_to_spark('DateTimeAdd(current_date(),3,"YEAR")')
    assert result == "dateadd(YEAR, 3, current_date())"


def test_datetimeadd_with_unrecognized_unit_is_left_untouched() -> None:
    original = 'DateTimeAdd([A],3,"FORTNIGHT")'
    result = alteryx_expr_to_spark(original)
    assert result == "DateTimeAdd(`A`,3,\"FORTNIGHT\")"
    assert "DateTimeAdd" in unknown_functions(result)


def test_unknown_functions_empty_for_translated_known_expression() -> None:
    result = alteryx_expr_to_spark("IIF(Contains([A],'x'),Length([A]),0)")
    assert unknown_functions(result) == set()


def test_unknown_functions_flags_untranslated_call() -> None:
    assert unknown_functions("SomeWeirdAlteryxFunction(`A`)") == {"SomeWeirdAlteryxFunction"}


def test_unknown_functions_ignores_field_refs_and_keywords() -> None:
    result = alteryx_expr_to_spark("CASE WHEN [A] IN ('x','y') THEN 1 ELSE 0 END")
    assert unknown_functions(result) == set()


def test_datetimetoutc_one_arg_defaults_to_utc() -> None:
    assert alteryx_expr_to_spark("DateTimetoUTC(current_date())") == (
        "to_utc_timestamp(current_date(), 'UTC')"
    )
    assert unknown_functions(alteryx_expr_to_spark("DateTimetoUTC([t])")) == set()


def test_datetimetoutc_two_arg_keeps_timezone() -> None:
    assert alteryx_expr_to_spark('DateTimeToUTC([ts], "Europe/Paris")') == (
        'to_utc_timestamp(`ts`, "Europe/Paris")'
    )


def test_datetimetolocal_maps_to_from_utc_timestamp() -> None:
    assert alteryx_expr_to_spark("DateTimeToLocal([ts])") == (
        "from_utc_timestamp(`ts`, 'UTC')"
    )


def test_padleft_padright_and_ceiling_renames() -> None:
    assert alteryx_expr_to_spark("PadLeft([x], 10, '0')") == "lpad(`x`, 10, '0')"
    assert alteryx_expr_to_spark("PadRight([x], 5, ' ')") == "rpad(`x`, 5, ' ')"
    assert alteryx_expr_to_spark("Ceiling([n])") == "ceil(`n`)"


def test_replacechar_maps_to_translate_and_md5() -> None:
    assert alteryx_expr_to_spark("ReplaceChar([s], 'ab', 'xy')") == "translate(`s`, 'ab', 'xy')"
    assert alteryx_expr_to_spark("MD5_ASCII([s])") == "md5(`s`)"


def test_datetimediff_days_maps_to_datediff() -> None:
    assert alteryx_expr_to_spark('DateTimeDiff([a], [b], "days")') == "datediff(`a`, `b`)"


def test_datetimediff_nonday_unit_left_flagged() -> None:
    out = alteryx_expr_to_spark('DateTimeDiff([a], [b], "months")')
    assert out.startswith("DateTimeDiff(")
    assert "DateTimeDiff" in unknown_functions(out)
