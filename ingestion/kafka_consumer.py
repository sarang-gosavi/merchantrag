"""
ingestion/kafka_consumer.py — Consumer + Snowpipe + batch loaders.

Consumes merchant transaction events from Kafka topic,
validates schema, then routes to:
  1. Snowflake via Snowpipe (streaming)
  2. SQLite fallback (local dev)
  3. File-based batch loader (Ab Initio-style domain bridge)
"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from config import settings
from models import KafkaEvent, MerchantTransaction, PipelineStats

logger = logging.getLogger(__name__)


# ── Snowpipe / Snowflake writer ───────────────────────────────────────────────

class SnowflakeWriter:
    """
    Writes merchant transactions to Snowflake.

    Real mode:  snowflake-connector-python → Snowpipe REST API
    Local mode: SQLite (identical schema, no external service needed)

    Schema mirrors the playbook spec:
      merchant_id, transaction_amount, MCC code, location, timestamp, acceptance_method
    """

    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS raw_merchant_transactions (
        transaction_id      TEXT PRIMARY KEY,
        merchant_id         TEXT NOT NULL,
        merchant_name       TEXT,
        transaction_amount  REAL,
        mcc_code            TEXT,
        mcc_description     TEXT,
        city                TEXT,
        state               TEXT,
        neighborhood        TEXT,
        latitude            REAL,
        longitude           REAL,
        timestamp           TEXT,
        acceptance_method   TEXT,
        stars               REAL,
        review_count        INTEGER,
        review_velocity_30d INTEGER,
        categories          TEXT,
        is_open             INTEGER,
        price_range         INTEGER,
        ingested_at         TEXT
    );
    """

    def __init__(self):
        self._use_snowflake = not settings.USE_LOCAL_DB
        self._conn = None

    async def start(self):
        if self._use_snowflake:
            await self._init_snowflake()
        else:
            await self._init_sqlite()
        logger.info(
            "SnowflakeWriter ready — backend: %s",
            "snowflake" if self._use_snowflake else "sqlite"
        )

    async def _init_snowflake(self):
        """Connect to Snowflake and ensure raw schema exists."""
        try:
            import snowflake.connector
            self._conn = snowflake.connector.connect(
                account=settings.SNOWFLAKE_ACCOUNT,
                user=settings.SNOWFLAKE_USER,
                password=settings.SNOWFLAKE_PASSWORD,
                database=settings.SNOWFLAKE_DATABASE,
                schema=settings.SNOWFLAKE_SCHEMA,
                warehouse=settings.SNOWFLAKE_WAREHOUSE,
                role=settings.SNOWFLAKE_ROLE,
            )
            cursor = self._conn.cursor()
            cursor.execute(self.CREATE_TABLE_SQL.replace("TEXT", "VARCHAR")
                           .replace("REAL", "FLOAT"))
            cursor.close()
        except ImportError:
            logger.warning("snowflake-connector-python not installed — falling back to SQLite")
            self._use_snowflake = False
            await self._init_sqlite()

    async def _init_sqlite(self):
        """Create SQLite database with same schema as Snowflake raw table."""
        import aiosqlite
        db_path = Path(settings.LOCAL_DB_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(db_path) as db:
            await db.execute(self.CREATE_TABLE_SQL)
            await db.commit()

    async def upsert(self, txn: MerchantTransaction) -> None:
        row = (
            txn.transaction_id, txn.merchant_id, txn.merchant_name,
            txn.transaction_amount, txn.mcc_code, txn.mcc_description,
            txn.city, txn.state, txn.neighborhood,
            txn.latitude, txn.longitude,
            txn.timestamp.isoformat(),
            txn.acceptance_method.value,
            txn.stars, txn.review_count, txn.review_velocity_30d,
            ",".join(txn.categories), int(txn.is_open), txn.price_range,
            datetime.utcnow().isoformat(),
        )
        if self._use_snowflake:
            self._snowflake_upsert(row)
        else:
            await self._sqlite_upsert(row)

    def _snowflake_upsert(self, row):
        sql = """
        MERGE INTO raw_merchant_transactions t
        USING (SELECT %s AS transaction_id) src
        ON t.transaction_id = src.transaction_id
        WHEN NOT MATCHED THEN INSERT VALUES (%s)
        """ % ("%s", ",".join(["%s"] * len(row)))
        cursor = self._conn.cursor()
        cursor.execute(sql, (row[0],) + row)
        cursor.close()

    async def _sqlite_upsert(self, row):
        import aiosqlite
        sql = """
        INSERT OR REPLACE INTO raw_merchant_transactions VALUES
        (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        async with aiosqlite.connect(settings.LOCAL_DB_PATH) as db:
            await db.execute(sql, row)
            await db.commit()

    async def upsert_batch(self, transactions: List[MerchantTransaction]) -> int:
        ok = 0
        for txn in transactions:
            try:
                await self.upsert(txn)
                ok += 1
            except Exception as exc:
                logger.error("Failed to upsert %s: %s", txn.transaction_id, exc)
        return ok

    async def count(self) -> int:
        if self._use_snowflake:
            cursor = self._conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM raw_merchant_transactions")
            n = cursor.fetchone()[0]
            cursor.close()
            return n
        import aiosqlite
        async with aiosqlite.connect(settings.LOCAL_DB_PATH) as db:
            async with db.execute("SELECT COUNT(*) FROM raw_merchant_transactions") as cur:
                row = await cur.fetchone()
                return row[0] if row else 0


# ── Batch loader (Ab Initio-style file-based ingestion) ──────────────────────

class BatchLoader:
    """
    File-based batch ingestion — mirrors Ab Initio-style ETL patterns.

    Reads CSV/Parquet files from a staging directory,
    validates schema, and bulk-loads into the warehouse.
    This is the domain bridge pattern referenced in the playbook.
    """

    STAGING_DIR = Path("./data/staging")
    ARCHIVE_DIR = Path("./data/archive")

    def __init__(self, writer: SnowflakeWriter):
        self.writer = writer
        self.STAGING_DIR.mkdir(parents=True, exist_ok=True)
        self.ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    def write_staging_file(self, transactions: List[MerchantTransaction], filename: str = None) -> Path:
        """Write transactions to a CSV staging file (simulates Ab Initio output)."""
        if not filename:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"merchant_txn_{ts}.csv"
        path = self.STAGING_DIR / filename
        fieldnames = [
            "transaction_id", "merchant_id", "merchant_name",
            "transaction_amount", "mcc_code", "mcc_description",
            "city", "state", "neighborhood", "latitude", "longitude",
            "timestamp", "acceptance_method", "stars", "review_count",
            "review_velocity_30d", "categories", "is_open", "price_range",
        ]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for txn in transactions:
                writer.writerow({
                    "transaction_id": txn.transaction_id,
                    "merchant_id": txn.merchant_id,
                    "merchant_name": txn.merchant_name,
                    "transaction_amount": txn.transaction_amount,
                    "mcc_code": txn.mcc_code,
                    "mcc_description": txn.mcc_description,
                    "city": txn.city,
                    "state": txn.state,
                    "neighborhood": txn.neighborhood or "",
                    "latitude": txn.latitude,
                    "longitude": txn.longitude,
                    "timestamp": txn.timestamp.isoformat(),
                    "acceptance_method": txn.acceptance_method.value,
                    "stars": txn.stars or "",
                    "review_count": txn.review_count or 0,
                    "review_velocity_30d": txn.review_velocity_30d or 0,
                    "categories": "|".join(txn.categories),
                    "is_open": int(txn.is_open),
                    "price_range": txn.price_range or 2,
                })
        logger.info("Staged %d records → %s", len(transactions), path.name)
        return path

    async def process_staging_files(self) -> int:
        """Process all CSV files in staging directory → warehouse."""
        total = 0
        for csv_path in self.STAGING_DIR.glob("*.csv"):
            loaded = await self._load_file(csv_path)
            total += loaded
            # Archive processed file
            csv_path.rename(self.ARCHIVE_DIR / csv_path.name)
            logger.info("Archived %s (%d records)", csv_path.name, loaded)
        return total

    async def _load_file(self, path: Path) -> int:
        transactions: List[MerchantTransaction] = []
        with open(path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    from models import AcceptanceMethod
                    txn = MerchantTransaction(
                        transaction_id=row["transaction_id"],
                        merchant_id=row["merchant_id"],
                        merchant_name=row["merchant_name"],
                        transaction_amount=float(row["transaction_amount"]),
                        mcc_code=row["mcc_code"],
                        mcc_description=row["mcc_description"],
                        city=row["city"],
                        state=row["state"],
                        neighborhood=row.get("neighborhood") or None,
                        latitude=float(row["latitude"]),
                        longitude=float(row["longitude"]),
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        acceptance_method=AcceptanceMethod(row["acceptance_method"]),
                        stars=float(row["stars"]) if row.get("stars") else None,
                        review_count=int(row.get("review_count") or 0),
                        review_velocity_30d=int(row.get("review_velocity_30d") or 0),
                        categories=[c for c in row.get("categories", "").split("|") if c],
                        is_open=bool(int(row.get("is_open", 1))),
                        price_range=int(row.get("price_range") or 2),
                    )
                    transactions.append(txn)
                except Exception as exc:
                    logger.warning("Skipping malformed row in %s: %s", path.name, exc)

        return await self.writer.upsert_batch(transactions)


# ── Kafka Consumer ─────────────────────────────────────────────────────────────

class MerchantConsumer:
    """
    Consumes KafkaEvents from the merchant.transactions topic,
    validates them as MerchantTransaction objects, and yields them
    for downstream routing (vector store indexing, warehouse write).
    """

    def __init__(self):
        self.topic = settings.KAFKA_TOPIC_TRANSACTIONS
        self._mock = settings.KAFKA_MOCK_MODE
        self._consumer = None

    async def start(self):
        if self._mock:
            logger.info("[Mock Kafka] Consumer ready — topic '%s'", self.topic)
            return
        try:
            from aiokafka import AIOKafkaConsumer
            self._consumer = AIOKafkaConsumer(
                self.topic,
                bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
                group_id=settings.KAFKA_GROUP_ID,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                auto_offset_reset="earliest",
                enable_auto_commit=True,
            )
            await self._consumer.start()
        except ImportError:
            logger.warning("aiokafka not installed — using mock mode")
            self._mock = True

    async def stop(self):
        if self._consumer:
            await self._consumer.stop()

    async def consume(self, max_records: int = 0) -> AsyncGenerator[MerchantTransaction, None]:
        """Yield validated MerchantTransaction objects from the topic."""
        consumed = 0
        q = _MockBroker().get().topic(self.topic)

        while True:
            if self._mock:
                try:
                    raw = await asyncio.wait_for(q.get(), timeout=2.0)
                except asyncio.TimeoutError:
                    return
            else:
                msg = await self._consumer.__anext__()
                raw = msg.value

            try:
                event = KafkaEvent(**raw)
                txn = MerchantTransaction(**event.payload)
                yield txn
                consumed += 1
                if max_records and consumed >= max_records:
                    return
            except Exception as exc:
                logger.warning("Failed to parse Kafka message: %s", exc)


# ── Import mock broker from producer module ───────────────────────────────────
from ingestion.kafka_producer import _MockBroker  # noqa: E402
