"""Translates Alteryx expressions into Spark SQL expression text.

Two layers:
1. `[Field Name]` references become backtick-quoted Spark SQL identifiers.
2. Alteryx-specific functions and syntax are rewritten to Spark SQL
   equivalents via a deterministic mapping (IIF -> if, ToString -> string,
   block IF/THEN/ELSEIF/ELSE/ENDIF -> CASE WHEN, 0-based Substring ->
   1-based substring, string `+` concatenation -> `||`, `= Null()` ->
   IS NULL, IsEmpty(x) -> (x IS NULL OR x = ''), and friends).

Functions without an entry pass through unchanged to Spark's SQL parser —
unknown ones will surface as an AnalysisException at run time rather than
being silently mistranslated.
"""

from __future__ import annotations

import re

# Field names may themselves contain one level of brackets (cube-style names
# like "ProjectCube_Data[Redbox Customer]"), so match the outer reference and
# keep inner brackets as part of the quoted identifier.
_FIELD_REF = re.compile(r"\[((?:[^\[\]]|\[[^\]]*\])+)\]")

# Case-insensitive 1:1 function renames (call shape unchanged).
_FUNC_RENAMES: dict[str, str] = {
    "iif": "if",
    "tostring": "string",
    "tonumber": "double",
    "datetimemonth": "month",
    "datetimeyear": "year",
    "datetimeday": "day",
    "datetimehour": "hour",
    "datetimenow": "current_timestamp",
    "datetimetoday": "current_date",
    "uppercase": "upper",
    "lowercase": "lower",
    "titlecase": "initcap",
    "trimleft": "ltrim",
    "trimright": "rtrim",
    "regex_replace": "regexp_replace",
    "regex_match": "rlike",
    # same call shape and semantics, different name in Spark SQL
    "padleft": "lpad",
    "padright": "rpad",
    "ceiling": "ceil",
    "replacechar": "translate",  # char-for-char replacement
    "md5_ascii": "md5",
    "md5_unicode": "md5",
}

_NULL_CALL = re.compile(r"\bnull\s*\(\s*\)", re.IGNORECASE)
_EQ_NULL = re.compile(r"(=|==)\s*NULL\b")
_NE_NULL = re.compile(r"(!=|<>)\s*NULL\b")
_BLOCK_IF = re.compile(r"\bIF\b", re.IGNORECASE)
_BLOCK_ELSEIF = re.compile(r"\bELSEIF\b", re.IGNORECASE)
_BLOCK_ENDIF = re.compile(r"\bENDIF\b", re.IGNORECASE)

# Functions verified to exist in Spark/Databricks SQL with the same call
# shape and semantics as their Alteryx formula-language namesake (case
# differences aside), so leaving them untouched is safe. Anything NOT in
# this set that still looks like a function call after translation is
# either a genuine gap in this module or a name that happens to collide
# with something else in Spark — either way, `unknown_functions` flags it
# rather than betting on a silent guess.
_KNOWN_SAFE_FUNCTIONS: frozenset[str] = frozenset({
    # rename targets, already Spark-native once _rename_functions runs
    "if", "string", "double", "month", "year", "day", "hour",
    "current_timestamp", "current_date", "upper", "lower", "initcap",
    "ltrim", "rtrim", "regexp_replace", "rlike",
    # Alteryx names that pass through unchanged and match a Spark/Databricks
    # SQL builtin of the same name and call shape
    "trim", "length", "left", "right", "substring", "replace", "isnull",
    "contains", "startswith", "endswith", "coalesce", "concat", "isnan",
    "round", "abs", "ceil", "floor", "sqrt", "power", "mod", "nullif",
    "cast", "greatest", "least", "now", "date", "to_date", "to_timestamp",
    "date_format", "datediff", "date_add", "date_sub", "dateadd",
    "to_utc_timestamp", "from_utc_timestamp",
    "lpad", "rpad", "translate", "md5", "ln", "log", "log10", "exp",
    "regexp_extract", "split", "array", "size", "sum", "count", "avg",
    "min", "max", "distinct", "in", "like", "between", "exists",
    # SQL keywords that can appear immediately before "(" in generated text
    "case", "when", "then", "else", "end", "and", "or", "not", "is",
})

_CALL_NAME = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")

_DATE_UNITS = {
    "year", "quarter", "month", "week", "day", "hour", "minute", "second",
}


def _rename_functions(expr: str) -> str:
    for alteryx_name, spark_name in _FUNC_RENAMES.items():
        expr = re.sub(rf"\b{alteryx_name}\s*\(", f"{spark_name}(", expr, flags=re.IGNORECASE)
    return expr


def _find_call(expr: str, func_lower: str, start: int = 0) -> tuple[int, int, list[str]] | None:
    """Locate `func(...)` (case-insensitive) with balanced parens.

    Returns (start_index, end_index_exclusive, top_level_args) or None.
    """
    pattern = re.compile(rf"\b{func_lower}\s*\(", re.IGNORECASE)
    m = pattern.search(expr, start)
    if m is None:
        return None
    depth = 1
    args: list[str] = []
    current = ""
    i = m.end()
    in_quote: str | None = None
    while i < len(expr):
        ch = expr[i]
        if in_quote:
            if ch == in_quote:
                in_quote = None
            current += ch
        elif ch in "'\"":
            in_quote = ch
            current += ch
        elif ch == "(":
            depth += 1
            current += ch
        elif ch == ")":
            depth -= 1
            if depth == 0:
                args.append(current.strip())
                return m.start(), i + 1, args
            current += ch
        elif ch == "," and depth == 1:
            args.append(current.strip())
            current = ""
        else:
            current += ch
        i += 1
    return None


def _rewrite_datetimeadd(expr: str) -> str:
    """Alteryx DateTimeAdd(date, n, "unit") -> Databricks SQL dateadd(unit, n, date).

    Databricks's `dateadd` takes the unit as a bare keyword (not a string
    literal) and in the opposite argument order. Only rewritten when the
    unit is a literal we recognize (YEAR/MONTH/DAY/...); anything else is
    left as-is so `unknown_functions` flags it for review instead of this
    guessing at an unfamiliar shape.
    """
    search_from = 0
    while (found := _find_call(expr, "datetimeadd", search_from)) is not None:
        start, end, args = found
        if len(args) == 3:
            unit_match = re.fullmatch(r"""['"]\s*(\w+)\s*['"]""", args[2].strip())
            unit = unit_match.group(1).lower() if unit_match else ""
        else:
            unit = ""
        if unit in _DATE_UNITS:
            replacement = f"dateadd({unit.upper()}, {args[1]}, {args[0]})"
            expr = f"{expr[:start]}{replacement}{expr[end:]}"
            search_from = start + len(replacement)
        else:
            search_from = end
    return expr


def _rewrite_datetimediff(expr: str) -> str:
    """Alteryx DateTimeDiff(dt1, dt2, "days") -> Spark datediff(dt1, dt2).

    Alteryx returns dt1 - dt2 in the given unit; Spark's datediff is exactly
    that in days. Only the day unit maps to a single clean builtin, so other
    units are left untouched (and flagged) rather than mistranslated.
    """
    search_from = 0
    while (found := _find_call(expr, "datetimediff", search_from)) is not None:
        start, end, args = found
        unit = ""
        if len(args) == 3:
            unit_match = re.fullmatch(r"""['"]\s*(\w+)\s*['"]""", args[2].strip())
            unit = unit_match.group(1).lower() if unit_match else ""
        if unit in ("day", "days"):
            replacement = f"datediff({args[0]}, {args[1]})"
            expr = f"{expr[:start]}{replacement}{expr[end:]}"
            search_from = start + len(replacement)
        else:
            search_from = end
    return expr


def _rewrite_tz_convert(expr: str, alteryx_name: str, spark_name: str) -> str:
    """Alteryx DateTimeToUTC/DateTimeToLocal(dt[, tz]) -> Spark to/from_utc_timestamp.

    Alteryx's DateTimeToUTC(dt, tz) reads dt as being in tz and converts to
    UTC — exactly Spark's to_utc_timestamp(dt, tz) (and DateTimeToLocal ->
    from_utc_timestamp). Databricks has no DateTimeToUTC, so this was a hard
    runtime failure before. When no timezone is given, default to 'UTC'.
    """
    search_from = 0
    while (found := _find_call(expr, alteryx_name, search_from)) is not None:
        start, end, args = found
        if len(args) == 1:
            replacement = f"{spark_name}({args[0]}, 'UTC')"
        elif len(args) >= 2:
            replacement = f"{spark_name}({args[0]}, {args[1]})"
        else:
            search_from = end
            continue
        expr = f"{expr[:start]}{replacement}{expr[end:]}"
        search_from = start + len(replacement)
    return expr


def _rewrite_isempty(expr: str) -> str:
    while (found := _find_call(expr, "isempty")) is not None:
        start, end, args = found
        arg = args[0] if args else "NULL"
        expr = f"{expr[:start]}({arg} IS NULL OR {arg} = ''){expr[end:]}"
    return expr


def _rewrite_substring(expr: str) -> str:
    """Alteryx Substring(s, start, len) is 0-based; Spark substring is 1-based."""
    search_from = 0
    while (found := _find_call(expr, "substring", search_from)) is not None:
        start, end, args = found
        if len(args) >= 2 and re.fullmatch(r"\d+", args[1]):
            args[1] = str(int(args[1]) + 1)
        replacement = f"substring({', '.join(args)})"
        expr = f"{expr[:start]}{replacement}{expr[end:]}"
        search_from = start + len(replacement)
    return expr


def _rewrite_block_if(expr: str) -> str:
    """Alteryx block IF cond THEN a [ELSEIF ...] ELSE b ENDIF -> CASE WHEN ... END."""
    if not _BLOCK_ENDIF.search(expr):
        return expr
    expr = _BLOCK_ELSEIF.sub("WHEN", expr)
    expr = _BLOCK_IF.sub("CASE WHEN", expr)
    expr = _BLOCK_ENDIF.sub("END", expr)
    return expr


def _rewrite_string_concat(expr: str) -> str:
    """Replace `+` with `||` where either operand is a quoted string literal.

    Alteryx overloads + for concatenation; Spark SQL does not. Only the
    unambiguous cases (a string literal on one side) are rewritten.
    """
    result: list[str] = []
    i = 0
    in_quote: str | None = None
    while i < len(expr):
        ch = expr[i]
        if in_quote:
            if ch == in_quote:
                in_quote = None
            result.append(ch)
        elif ch in "'\"":
            in_quote = ch
            result.append(ch)
        elif ch == "+":
            prev = "".join(result).rstrip()
            nxt = expr[i + 1 :].lstrip()
            if (prev.endswith("'") or prev.endswith('"')) or (
                nxt.startswith("'") or nxt.startswith('"')
            ):
                result.append("||")
            else:
                result.append(ch)
        else:
            result.append(ch)
        i += 1
    return "".join(result)


def alteryx_expr_to_spark(expression: str) -> str:
    """Rewrite an Alteryx expression into Spark SQL expression text."""
    expr = _FIELD_REF.sub(lambda m: f"`{m.group(1)}`", expression.strip())
    expr = _NULL_CALL.sub("NULL", expr)
    expr = _rewrite_block_if(expr)
    expr = _rename_functions(expr)
    expr = _rewrite_datetimeadd(expr)
    expr = _rewrite_datetimediff(expr)
    expr = _rewrite_tz_convert(expr, "datetimetoutc", "to_utc_timestamp")
    expr = _rewrite_tz_convert(expr, "datetimetolocal", "from_utc_timestamp")
    expr = _rewrite_isempty(expr)
    expr = _rewrite_substring(expr)
    expr = _rewrite_string_concat(expr)
    expr = _EQ_NULL.sub("IS NULL", expr)
    expr = _NE_NULL.sub("IS NOT NULL", expr)
    return expr


def unknown_functions(expr: str) -> set[str]:
    """Function calls left in `expr` that aren't a verified-safe builtin.

    An unrecognized name either fails loudly at pipeline run time (the good
    case) or, worse, happens to collide with an unrelated Spark builtin and
    silently computes the wrong thing. Both are worth a human's eyes before
    the generated code is trusted, so callers surface this as a REVIEW
    comment rather than a translation guess.
    """
    return {name for name in _CALL_NAME.findall(expr) if name.lower() not in _KNOWN_SAFE_FUNCTIONS}
