-- Chip forecasting: per-week value of Triple Captain / Bench Boost across each chip's window.
ALTER TABLE optimization_runs ADD COLUMN chip_forecast JSONB NOT NULL DEFAULT '[]';
