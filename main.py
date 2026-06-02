"""
Cross-database data pipeline: ClickHouse ↔ SQL Server, using pandas as the intermediate data type.

Usage:
    1. Copy .env.example to .env and fill in your connection details.
    2. Run: python main.py
"""

import os
from contextlib import contextmanager
from dataclasses import dataclass

import pandas as pd
import clickhouse_connect
import clickhouse_connect.driver
import pyodbc
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DbConfig:
    """Connection parameters for a single database."""

    host: str
    port: int
    user: str
    password: str
    database: str
    # ClickHouse extras
    secure: bool = False
    # SQL Server extras (empty = use system default ODBC driver)
    driver: str = ""


def clickhouse_config_from_env(prefix: str = "CH_") -> DbConfig:
    """Build a ClickHouse DbConfig from environment variables."""
    return DbConfig(
        host=os.getenv(f"{prefix}HOST", "localhost"),
        port=int(os.getenv(f"{prefix}PORT", "8123")),
        user=os.getenv(f"{prefix}USER", "default"),
        password=os.getenv(f"{prefix}PASSWORD", ""),
        database=os.getenv(f"{prefix}DATABASE", "default"),
        secure=os.getenv(f"{prefix}SECURE", "false").lower() == "true",
    )


def sqlserver_config_from_env(prefix: str = "MSSQL_") -> DbConfig:
    """Build a SQL Server DbConfig from environment variables."""
    return DbConfig(
        host=os.getenv(f"{prefix}HOST", "localhost"),
        port=int(os.getenv(f"{prefix}PORT", "1433")),
        user=os.getenv(f"{prefix}USER", "sa"),
        password=os.getenv(f"{prefix}PASSWORD", ""),
        database=os.getenv(f"{prefix}DATABASE", "master"),
        driver=os.getenv(f"{prefix}DRIVER", ""),
    )


# ---------------------------------------------------------------------------
# ClickHouse client
# ---------------------------------------------------------------------------


class ClickHouseClient:
    """ClickHouse client with pandas read/write support."""

    def __init__(self, config: DbConfig) -> None:
        self._config = config
        self._client: clickhouse_connect.driver.Client | None = None

    def connect(self) -> "ClickHouseClient":
        self._client = clickhouse_connect.get_client(
            host=self._config.host,
            port=self._config.port,
            username=self._config.user,
            password=self._config.password,
            database=self._config.database,
            secure=self._config.secure,
        )
        return self

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    # ---- Query ----

    def query(self, sql: str, **params: object) -> pd.DataFrame:
        """Execute a SELECT query and return results as a pandas DataFrame."""
        self._ensure_connected()
        return self._client.query_df(sql, parameters=params)  # type: ignore[union-attr]

    # ---- Write ----

    def insert(self, table: str, df: pd.DataFrame) -> int:
        """Insert a DataFrame into a ClickHouse table. Returns row count written."""
        self._ensure_connected()
        rows = df.shape[0]
        self._client.insert_df(table, df)  # type: ignore[union-attr]
        return rows

    def execute(self, sql: str, **params: object) -> None:
        """Execute a DDL / DML statement (CREATE, ALTER, TRUNCATE, etc.)."""
        self._ensure_connected()
        self._client.command(sql, parameters=params)  # type: ignore[union-attr]

    def _ensure_connected(self) -> None:
        if self._client is None:
            raise RuntimeError("ClickHouseClient is not connected. Call .connect() first.")


# ---------------------------------------------------------------------------
# SQL Server client
# ---------------------------------------------------------------------------


class SQLServerClient:
    """SQL Server client with pandas read/write support via pyodbc."""

    def __init__(self, config: DbConfig) -> None:
        self._config = config
        self._conn: pyodbc.Connection | None = None

    def connect(self) -> "SQLServerClient":
        conn_str = self._build_connection_string()
        self._conn = pyodbc.connect(conn_str)
        return self

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ---- Query ----

    def query(self, sql: str, **params: object) -> pd.DataFrame:
        """Execute a SELECT query and return results as a pandas DataFrame."""
        self._ensure_connected()
        return pd.read_sql(sql, self._conn, params=params)  # type: ignore[arg-type]

    # ---- Write ----

    def insert(self, table: str, df: pd.DataFrame) -> int:
        """Insert a DataFrame into a SQL Server table (row-by-row via executemany).

        For bulk inserts consider using the `fast_executemany` ODBC option or
        calling `to_sql` with SQLAlchemy for larger datasets.
        """
        self._ensure_connected()
        if df.empty:
            return 0

        columns = list(df.columns)
        placeholders = ", ".join(["?"] * len(columns))
        col_names = ", ".join([_quote_identifier(c) for c in columns])
        sql = f"INSERT INTO {_quote_identifier(table)} ({col_names}) VALUES ({placeholders})"

        cursor = self._conn.cursor()  # type: ignore[union-attr]
        try:
            cursor.fast_executemany = True
            cursor.executemany(sql, df.values.tolist())
        finally:
            cursor.close()
        self._conn.commit()  # type: ignore[union-attr]
        return df.shape[0]

    def execute(self, sql: str) -> None:
        """Execute a DDL / DML statement."""
        self._ensure_connected()
        cursor = self._conn.cursor()  # type: ignore[union-attr]
        try:
            cursor.execute(sql)
        finally:
            cursor.close()
        self._conn.commit()  # type: ignore[union-attr]

    def _build_connection_string(self) -> str:
        parts = [
            f"DRIVER={{{self._config.driver}}}" if self._config.driver else "",
            f"SERVER={self._config.host},{self._config.port}",
            f"DATABASE={self._config.database}",
            f"UID={self._config.user}",
            f"PWD={self._config.password}" if self._config.password else "Trusted_Connection=yes",
        ]
        return ";".join(p for p in parts if p)

    def _ensure_connected(self) -> None:
        if self._conn is None:
            raise RuntimeError("SQLServerClient is not connected. Call .connect() first.")


# ---------------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------------


class DataPipeline:
    """Orchestrates data movement between ClickHouse and SQL Server via pandas."""

    def __init__(self, ch: ClickHouseClient, mssql: SQLServerClient) -> None:
        self.ch = ch
        self.mssql = mssql

    # -- Read from one, write to the other --

    def ch_to_mssql(self, query: str, table: str) -> int:
        """Query ClickHouse, write results to a SQL Server table."""
        df = self.ch.query(query)
        return self.mssql.insert(table, df)

    def mssql_to_ch(self, query: str, table: str) -> int:
        """Query SQL Server, write results to a ClickHouse table."""
        df = self.mssql.query(query)
        return self.ch.insert(table, df)

    # -- Compare two DataFrames --

    def compare(self, ch_query: str, mssql_query: str) -> pd.DataFrame:
        """Return rows present in ClickHouse but absent in SQL Server (by all columns)."""
        df_ch = self.ch.query(ch_query)
        df_mssql = self.mssql.query(mssql_query)
        merged = df_ch.merge(df_mssql, how="left", indicator=True)
        return merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_reserved_keywords: set[str] = {
    "TABLE", "SELECT", "INSERT", "UPDATE", "DELETE", "FROM", "WHERE", "ORDER",
    "GROUP", "BY", "HAVING", "JOIN", "ON", "INTO", "VALUES", "SET", "CREATE",
    "ALTER", "DROP", "INDEX", "KEY", "PRIMARY", "FOREIGN", "REFERENCES",
    "DEFAULT", "NULL", "NOT", "AND", "OR", "AS", "DISTINCT", "TOP", "LIMIT",
    "OFFSET", "UNION", "ALL", "EXCEPT", "INTERSECT", "CASE", "WHEN", "THEN",
    "ELSE", "END", "EXISTS", "BETWEEN", "IN", "LIKE", "IS", "COUNT", "SUM",
    "AVG", "MIN", "MAX",
}


def _quote_identifier(name: str) -> str:
    """Wrap a SQL identifier in brackets if it contains special characters or is a reserved word."""
    if name.upper() in _reserved_keywords or " " in name or "-" in name:
        return f"[{name}]"
    return name


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------


@contextmanager
def open_ch(**overrides: object):
    """Context manager for a ClickHouse connection."""
    cfg = clickhouse_config_from_env()
    for k, v in overrides.items():
        if hasattr(cfg, k) and v is not None:
            setattr(cfg, k, v)
    client = ClickHouseClient(cfg).connect()
    try:
        yield client
    finally:
        client.close()


@contextmanager
def open_mssql(**overrides: object):
    """Context manager for a SQL Server connection."""
    cfg = sqlserver_config_from_env()
    for k, v in overrides.items():
        if hasattr(cfg, k) and v is not None:
            setattr(cfg, k, v)
    client = SQLServerClient(cfg).connect()
    try:
        yield client
    finally:
        client.close()


def main() -> None:
    """Demonstrate the pipeline: ClickHouse ↔ SQL Server via pandas."""

    print("=" * 60)
    print("Cross-DB Pipeline Demo (ClickHouse ↔ SQL Server via pandas)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Example 1: Query both databases into pandas DataFrames
    # ------------------------------------------------------------------
    print("\n[1] Querying ClickHouse → pandas DataFrame ...")
    with open_ch() as ch:
        df_ch = ch.query("SELECT 1 AS id, 'hello' AS msg")
        print(f"    ClickHouse result:\n{df_ch}")

    print("\n[2] Querying SQL Server → pandas DataFrame ...")
    with open_mssql() as mssql:
        df_mssql = mssql.query("SELECT 1 AS id, 'hello' AS msg")
        print(f"    SQL Server result:\n{df_mssql}")

    # ------------------------------------------------------------------
    # Example 2: Move data from ClickHouse → SQL Server
    # ------------------------------------------------------------------
    print("\n[3] Moving data: ClickHouse → SQL Server ...")
    with open_ch() as ch, open_mssql() as mssql:
        pipeline = DataPipeline(ch, mssql)
        try:
            rows = pipeline.ch_to_mssql(
                "SELECT 1 AS id, 'transferred' AS msg",
                "test_pipeline",
            )
            print(f"    Wrote {rows} row(s) to SQL Server table [test_pipeline]")
        except Exception as e:
            print(f"    Skipped (table may not exist or permissions issue): {e}")

    # ------------------------------------------------------------------
    # Example 3: Move data from SQL Server → ClickHouse
    # ------------------------------------------------------------------
    print("\n[4] Moving data: SQL Server → ClickHouse ...")
    with open_ch() as ch, open_mssql() as mssql:
        pipeline = DataPipeline(ch, mssql)
        try:
            rows = pipeline.mssql_to_ch(
                "SELECT 1 AS id, 'transferred' AS msg",
                "test_pipeline",
            )
            print(f"    Wrote {rows} row(s) to ClickHouse table [test_pipeline]")
        except Exception as e:
            print(f"    Skipped (table may not exist or permissions issue): {e}")

    # ------------------------------------------------------------------
    # Example 4: Compare data between the two databases
    # ------------------------------------------------------------------
    print("\n[5] Comparing data between ClickHouse and SQL Server ...")
    with open_ch() as ch, open_mssql() as mssql:
        pipeline = DataPipeline(ch, mssql)
        try:
            diff = pipeline.compare(
                "SELECT 1 AS id",
                "SELECT 2 AS id",
            )
            print(f"    Rows only in ClickHouse: {diff.shape[0]}")
        except Exception as e:
            print(f"    Skipped: {e}")

    # ------------------------------------------------------------------
    # Example 5: Pure pandas processing (no DB write, just analysis)
    # ------------------------------------------------------------------
    print("\n[6] Pandas in-memory join of two database results ...")
    with open_ch() as ch, open_mssql() as mssql:
        df_a = ch.query("SELECT 1 AS key, 'clickhouse' AS source")
        df_b = mssql.query("SELECT 1 AS key, 'sqlserver' AS source")
        combined = pd.concat([df_a, df_b], ignore_index=True)
        print(f"    Combined DataFrame:\n{combined}")

    print("\nDone.")


if __name__ == "__main__":
    main()
