"""
ingestion/kafka_producer.py — Merchant transaction event producer.

Streams synthetic merchant events (Yelp → merchant schema mapping)
to a Kafka topic. Supports real Kafka broker and in-process mock mode.

Production: Set KAFKA_MOCK_MODE=false + KAFKA_BOOTSTRAP_SERVERS
Demo/CI:    KAFKA_MOCK_MODE=true (asyncio.Queue, zero infrastructure)

MCC code reference: ISO 18245 Merchant Category Codes
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Dict, List, Optional

from config import settings
from models import (
    AcceptanceMethod, KafkaEvent, MerchantTransaction, PipelineStats
)

logger = logging.getLogger(__name__)

# ── MCC Code mapping (ISO 18245 subset — restaurant/retail domain) ────────────
MCC_MAP = {
    "5812": "Eating Places, Restaurants",
    "5813": "Drinking Places (Alcoholic Beverages) — Bars, Taverns, Cocktail Lounges",
    "5814": "Fast Food Restaurants",
    "5411": "Grocery Stores, Supermarkets",
    "5912": "Drug Stores, Pharmacies",
    "7011": "Lodging — Hotels, Motels, Resorts",
    "5941": "Sporting Goods Stores",
    "5921": "Package Stores — Beer, Wine, Liquor",
    "7011": "Hotels and Motels",
    "5999": "Miscellaneous and Specialty Retail Stores",
    "7299": "Services — Not Elsewhere Classified",
    "5734": "Computer and Computer Software Stores",
    "5651": "Family Clothing Stores",
    "5661": "Shoe Stores",
    "5732": "Electronics Stores",
    "7996": "Amusement Parks, Circuses, Carnivals",
    "7011": "Hotels, Motels, and Resorts",
    "4111": "Local/Suburban Commuter Transportation",
    "7011": "Hotels",
    "5045": "Computers, Peripherals, and Software",
}

CATEGORY_TO_MCC = {
    "Restaurants":        "5812",
    "Pizza":              "5812",
    "Italian":            "5812",
    "Burgers":            "5812",
    "Mexican":            "5812",
    "Japanese":           "5812",
    "Sushi Bars":         "5812",
    "Bars":               "5813",
    "Cocktail Bars":      "5813",
    "Breweries":          "5813",
    "Wine Bars":          "5813",
    "Fast Food":          "5814",
    "Hot Dogs":           "5814",
    "Grocery":            "5411",
    "Hotels":             "7011",
    "Gyms":               "7299",
    "Coffee & Tea":       "5812",
    "Cafes":              "5812",
    "Bakeries":           "5812",
    "Music Venues":       "7996",
}

ACCEPTANCE_WEIGHTS = {
    AcceptanceMethod.chip:        0.45,
    AcceptanceMethod.contactless: 0.25,
    AcceptanceMethod.online:      0.15,
    AcceptanceMethod.swipe:       0.10,
    AcceptanceMethod.mobile_pay:  0.04,
    AcceptanceMethod.keyed:       0.01,
}

PRICE_TO_AVG_TRANSACTION = {1: 15, 2: 35, 3: 75, 4: 150}


def yelp_to_transaction(raw: Dict[str, Any]) -> MerchantTransaction:
    """
    Map a Yelp business record to a MerchantTransaction.
    This is the Yelp→merchant domain bridge.
    """
    cats = raw.get("categories") or []
    if isinstance(cats, str):
        cats = [c.strip() for c in cats.split(",")]

    # Assign MCC from first matching category
    mcc = "5812"   # default: restaurants
    for cat in cats:
        if cat in CATEGORY_TO_MCC:
            mcc = CATEGORY_TO_MCC[cat]
            break

    attrs = raw.get("attributes") or {}
    price_range = attrs.get("RestaurantsPriceRange2") or 2
    try:
        price_range = int(price_range)
    except (TypeError, ValueError):
        price_range = 2

    base_amount = PRICE_TO_AVG_TRANSACTION.get(price_range, 35)
    # Add Gaussian noise to simulate real transaction distribution
    amount = max(1.0, random.gauss(base_amount, base_amount * 0.35))

    # Random acceptance method, weighted by real-world distribution
    methods = list(ACCEPTANCE_WEIGHTS.keys())
    weights = list(ACCEPTANCE_WEIGHTS.values())
    acceptance = random.choices(methods, weights=weights, k=1)[0]

    # Backdate transaction randomly within last 90 days
    days_ago = random.randint(0, 90)
    ts = datetime.utcnow() - timedelta(days=days_ago, hours=random.randint(0, 23))

    stars = raw.get("stars")
    review_count = raw.get("review_count", 0)
    velocity = raw.get("review_velocity_30d", 0)

    return MerchantTransaction(
        transaction_id=str(uuid.uuid4()),
        merchant_id=raw["business_id"],
        merchant_name=raw.get("name", "Unknown"),
        transaction_amount=round(amount, 2),
        mcc_code=mcc,
        mcc_description=MCC_MAP.get(mcc, "General Retail"),
        city=raw.get("city", ""),
        state=raw.get("state", ""),
        neighborhood=raw.get("neighborhood"),
        latitude=float(raw.get("latitude") or 0),
        longitude=float(raw.get("longitude") or 0),
        timestamp=ts,
        acceptance_method=acceptance,
        stars=float(stars) if stars is not None else None,
        review_count=int(review_count),
        review_velocity_30d=int(velocity),
        categories=cats[:5],
        is_open=bool(raw.get("is_open", 1)),
        price_range=price_range,
    )


# ── In-process mock broker ────────────────────────────────────────────────────

class _MockBroker:
    _instance: Optional["_MockBroker"] = None
    _queues: Dict[str, asyncio.Queue] = {}

    @classmethod
    def get(cls) -> "_MockBroker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def topic(self, name: str) -> asyncio.Queue:
        if name not in self._queues:
            self._queues[name] = asyncio.Queue(maxsize=50_000)
        return self._queues[name]


# ── Kafka Producer ─────────────────────────────────────────────────────────────

class MerchantProducer:
    """
    Produces merchant transaction events to Kafka.

    Real mode:  AIOKafkaProducer → Confluent / MSK / self-hosted
    Mock mode:  asyncio.Queue — identical interface, zero infra
    """

    def __init__(self):
        self.topic = settings.KAFKA_TOPIC_TRANSACTIONS
        self._mock = settings.KAFKA_MOCK_MODE
        self._producer = None

    async def start(self):
        if self._mock:
            logger.info("[Mock Kafka] Producer ready — topic '%s'", self.topic)
            return
        try:
            from aiokafka import AIOKafkaProducer
            self._producer = AIOKafkaProducer(
                bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                compression_type="gzip",
                max_batch_size=settings.KAFKA_BATCH_SIZE * 1024,
            )
            await self._producer.start()
            logger.info("Kafka producer connected: %s", settings.KAFKA_BOOTSTRAP_SERVERS)
        except ImportError:
            logger.warning("aiokafka not installed — falling back to mock mode")
            self._mock = True

    async def stop(self):
        if self._producer:
            await self._producer.stop()

    async def send(self, transaction: MerchantTransaction) -> None:
        event = KafkaEvent(
            event_type="merchant_transaction",
            payload=transaction.dict(),
        )
        raw = event.dict()
        if self._mock:
            await _MockBroker.get().topic(self.topic).put(raw)
        else:
            await self._producer.send_and_wait(self.topic, raw)

    async def send_batch(self, transactions: List[MerchantTransaction]) -> int:
        sent = 0
        for txn in transactions:
            await self.send(txn)
            sent += 1
        if sent:
            logger.debug("Produced %d transactions to '%s'", sent, self.topic)
        return sent

    async def produce_from_yelp(
        self,
        records: List[Dict],
        batch_size: int = 200,
    ) -> int:
        """
        Stream Yelp business records through the transaction producer.
        Converts each business → MerchantTransaction and sends to Kafka.
        """
        total = 0
        batch: List[MerchantTransaction] = []
        for raw in records:
            try:
                txn = yelp_to_transaction(raw)
                batch.append(txn)
            except Exception as exc:
                logger.warning("Failed to convert record %s: %s", raw.get("business_id"), exc)

            if len(batch) >= batch_size:
                sent = await self.send_batch(batch)
                total += sent
                batch = []

        if batch:
            total += await self.send_batch(batch)

        logger.info("Produced %d merchant transactions from %d Yelp records", total, len(records))
        return total
