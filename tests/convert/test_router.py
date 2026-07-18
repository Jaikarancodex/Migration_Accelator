import pytest

from convert.router import UnknownSourceSystemError, route_language


@pytest.mark.parametrize("source", ["alteryx", "Alteryx", "pentaho"])
def test_procedural_sources_route_to_pyspark(source: str) -> None:
    assert route_language(source) == "pyspark"


@pytest.mark.parametrize("source", ["teradata", "oracle", "synapse", "redshift", "bigquery"])
def test_set_based_sources_route_to_sql(source: str) -> None:
    assert route_language(source) == "sql"


def test_unknown_source_raises() -> None:
    with pytest.raises(UnknownSourceSystemError):
        route_language("sap_hana")
