-- Sample source data for the W3 pipeline.
-- Run this in a Databricks SQL editor / notebook (SQL cell) BEFORE running the
-- W3 pipeline, so its bronze read has a table to land.
--
-- The W3 source is Migration_Accelator.testing.todo_source_28. Column names
-- keep the original Alteryx bracket-style identifiers, so every reference is
-- backtick-quoted.

CREATE CATALOG IF NOT EXISTS Migration_Accelator;
CREATE SCHEMA  IF NOT EXISTS Migration_Accelator.testing;

CREATE OR REPLACE TABLE Migration_Accelator.testing.todo_source_28 (
  `ProjectCube_Data[Scenario]`        STRING,
  `ProjectCube_Data[NokiaPeriod]`     STRING,
  `ProjectCube_Data[Account_ID]`      STRING,
  `ProjectCube_Data[Source_ID]`       STRING,
  `ProjectCube_Data[Redbox Customer]` STRING,
  `ProjectCube_Data[GIC]`             STRING,
  `ProjectCube_Data[TopWBS]`          STRING,
  `ProjectCube_Data[Legal Entity]`    STRING,
  `[Base_Measure]`                    DOUBLE
);

INSERT INTO Migration_Accelator.testing.todo_source_28 VALUES
  ('Actuals',  '2026-M01', 'ACC1001', 'SRC-001', 'Vodafone Group',        'GIC-EUR-001', 'WBS-1000-ACCESS', 'Nokia Solutions and Networks', 12345.67),
  ('Actuals',  '2026-M01', 'ACC1002', 'SRC-002', 'Deutsche Telekom AG',   'GIC-EUR-002', 'WBS-1000-CORE',   'Nokia Oyj',                    98765.43),
  ('Budget',   '2026-M02', 'ACC1003', 'SRC-003', 'Orange S.A.',           'GIC-EUR-003', 'WBS-2000-OPTICS', 'Nokia of America Corp',        45678.90),
  ('Forecast', '2026-M02', 'ACC1004', 'SRC-004', 'Telefonica',            'GIC-EUR-004', 'WBS-2000-FIXED',  'Nokia Solutions and Networks', 23456.78),
  ('Actuals',  '2026-M03', 'ACC1005', 'SRC-005', 'BT Group plc',          'GIC-GBP-001', 'WBS-3000-MOBILE', 'Nokia UK Ltd',                 34567.89),
  -- a row with an over-length value to exercise the LEFT(...) truncations
  ('Budget',   '2026-M03', 'ACC1006-THIS-ACCOUNT-ID-IS-LONGER-THAN-THIRTY-CHARS', 'SRC-006-LONGER-THAN-15', 'A Very Long Redbox Customer Name Exceeding Thirty', 'GIC-USD-001', 'WBS-4000-SUBSEA', 'A Very Long Legal Entity Name Over 30', 87654.32),
  -- a row with NULLs to confirm nothing blows up on missing values
  ('Actuals',  '2026-M04', 'ACC1007', 'SRC-007', NULL,                    NULL,          'WBS-5000-CLOUD',  'Nokia Bell Labs',              NULL);

SELECT * FROM Migration_Accelator.testing.todo_source_28;
