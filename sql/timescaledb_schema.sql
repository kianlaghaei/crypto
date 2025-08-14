-- sql/timescaledb_schema.sql
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS ohlcv (
  ts timestamptz NOT NULL,
  symbol text NOT NULL,
  timeframe text NOT NULL,
  open double precision NOT NULL,
  high double precision NOT NULL,
  low double precision NOT NULL,
  close double precision NOT NULL,
  volume double precision NOT NULL,
  PRIMARY KEY (symbol, timeframe, ts)
);

SELECT create_hypertable('ohlcv','ts', if_not_exists => TRUE, migrate_data => TRUE, chunk_time_interval => INTERVAL '7 days');

CREATE INDEX IF NOT EXISTS ohlcv_symbol_tf_ts_idx ON ohlcv (symbol, timeframe, ts DESC);

-- نگهداری / سیاست Retention (نمونه)
-- SELECT add_retention_policy('ohlcv', INTERVAL '365 days');
