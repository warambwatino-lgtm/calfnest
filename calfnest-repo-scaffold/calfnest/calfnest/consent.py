"""Data Protection Act [Chapter 12:07] consent gate.

Until a farmer's consent record is `granted`, the sync layer must not let
any PII or audio leave the device. Withdrawal deletes opted-in audio and
writes to an audit log. This module is the single source of truth for that
decision so it can be unit-tested and cannot be bypassed by callers.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ConsentRecord:
    farmer_id: str
    purpose: str
    version: str
    channel: str                     # "app" | "ussd" | "paper"
    granted: bool = False
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def may_sync_personal_data(record: ConsentRecord | None) -> bool:
    """Return True only if valid, granted consent exists."""
    return bool(record) and record.granted and bool(record.farmer_id)


def withdraw(record: ConsentRecord) -> ConsentRecord:
    """Flip consent off; caller must then delete opted-in audio + audit."""
    record.granted = False
    record.timestamp = datetime.now(timezone.utc).isoformat()
    return record
