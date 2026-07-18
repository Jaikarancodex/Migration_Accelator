"""Schema models used to generate synthetic data and drive parity checks."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ColumnType = Literal["int", "float", "string", "bool", "date", "timestamp"]


class ColumnSchema(BaseModel):
    name: str
    data_type: ColumnType
    nullable: bool = True
    key: bool = False


class TableSchema(BaseModel):
    name: str
    columns: list[ColumnSchema] = Field(default_factory=list)

    def key_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.key]

    def value_columns(self) -> list[str]:
        return [c.name for c in self.columns if not c.key]
