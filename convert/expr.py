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

_FIELD_REF = re.compile(r"\[([^\[\]]+)]")

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
}

_NULL_CALL = re.compile(r"\bnull\s*\(\s*\)", re.IGNORECASE)
_EQ_NULL = re.compile(r"(=|==)\s*NULL\b")
_NE_NULL = re.compile(r"(!=|<>)\s*NULL\b")
_BLOCK_IF = re.compile(r"\bIF\b", re.IGNORECASE)
_BLOCK_ELSEIF = re.compile(r"\bELSEIF\b", re.IGNORECASE)
_BLOCK_ENDIF = re.compile(r"\bENDIF\b", re.IGNORECASE)


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
    expr = _rewrite_isempty(expr)
    expr = _rewrite_substring(expr)
    expr = _rewrite_string_concat(expr)
    expr = _EQ_NULL.sub("IS NULL", expr)
    expr = _NE_NULL.sub("IS NOT NULL", expr)
    return expr
