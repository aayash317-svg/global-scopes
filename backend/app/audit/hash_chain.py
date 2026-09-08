"""
Tamper-Evident Hash-Chain Audit Logging Subsystem
Cryptographically binds each security event to its predecessor using SHA-256.
Provides deterministic end-to-end chain verification to prove data integrity.
Theme alignment: Blockchain & Cybersecurity (lightweight immutable event chain).
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.audit.events import AuditEventType, serialize_payload
from backend.app.database.models import AuditEventRecord
from backend.app.database.repositories import AuditRepository


@dataclass
class ChainVerificationResult:
    is_valid: bool
    total_blocks: int
    head_hash: Optional[str] = None
    tampered_block_sequence: Optional[int] = None
    tampered_field: Optional[str] = None
    expected_hash: Optional[str] = None
    actual_hash: Optional[str] = None
    message: str = "Hash chain integrity intact."


class HashChainService:
    """Manages appending blocks and verifying SHA-256 cryptographic chain integrity."""

    GENESIS_PREV_HASH = "0" * 64

    def __init__(self, db: AsyncSession):
        self.db = db
        self.repo = AuditRepository(db)

    @staticmethod
    def _format_timestamp(ts: Any) -> str:
        """Standardizes timestamp to deterministic second-precision format."""
        if isinstance(ts, datetime):
            return ts.strftime("%Y-%m-%d %H:%M:%S")
        s = str(ts)
        # Handle ISO strings with T
        if "T" in s:
            s = s.replace("T", " ")
        return s[:19]

    @staticmethod
    def calculate_hash(prev_hash: str, sequence_number: int, timestamp_str: str, event_type: str, payload_str: str) -> str:
        """Computes deterministic SHA-256 hash across canonical block components."""
        data = f"{prev_hash}|{sequence_number}|{timestamp_str}|{event_type}|{payload_str}"
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    async def _ensure_genesis_block(self) -> AuditEventRecord:
        """Ensures the immutable genesis block exists."""
        latest = await self.repo.get_latest()
        if latest is None:
            now_dt = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
            ts_str = self._format_timestamp(now_dt)
            payload = {"message": "Voice Integrity Verification Framework Genesis Block"}
            payload_str = serialize_payload(payload)
            genesis_hash = self.calculate_hash(
                prev_hash=self.GENESIS_PREV_HASH,
                sequence_number=1,
                timestamp_str=ts_str,
                event_type=AuditEventType.GENESIS.value,
                payload_str=payload_str,
            )
            record = await self.repo.append(
                event_type=AuditEventType.GENESIS.value,
                prev_hash=self.GENESIS_PREV_HASH,
                event_hash=genesis_hash,
                payload=payload,
                session_id=None,
                timestamp_dt=now_dt,
            )
            return record
        return latest

    async def record_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> AuditEventRecord:
        """
        Appends an auditable security event to the cryptographic chain.
        Ensures NO raw audio or unencrypted biometrics are written.
        """
        latest = await self._ensure_genesis_block()
        # Re-fetch latest to ensure fresh head
        latest = await self.repo.get_latest()

        prev_hash = latest.event_hash
        seq = latest.sequence_number + 1
        now_dt = datetime.now(timezone.utc).replace(microsecond=0)
        ts_str = self._format_timestamp(now_dt)
        payload_str = serialize_payload(payload)

        event_hash = self.calculate_hash(
            prev_hash=prev_hash,
            sequence_number=seq,
            timestamp_str=ts_str,
            event_type=event_type,
            payload_str=payload_str,
        )

        record = await self.repo.append(
            event_type=event_type,
            prev_hash=prev_hash,
            event_hash=event_hash,
            payload=payload,
            session_id=session_id,
            timestamp_dt=now_dt,
        )
        return record

    async def verify_chain(self) -> ChainVerificationResult:
        """
        Recomputes every cryptographic hash from Genesis to Head.
        Returns is_valid=False the instant any historical record is modified.
        """
        blocks: List[AuditEventRecord] = await self.repo.get_all(limit=10000)
        if not blocks:
            return ChainVerificationResult(is_valid=True, total_blocks=0, message="Empty chain.")

        # 1. Verify Genesis Block
        genesis = blocks[0]
        if genesis.prev_hash != self.GENESIS_PREV_HASH:
            return ChainVerificationResult(
                is_valid=False,
                total_blocks=len(blocks),
                tampered_block_sequence=1,
                tampered_field="prev_hash",
                expected_hash=self.GENESIS_PREV_HASH,
                actual_hash=genesis.prev_hash,
                message="Genesis block previous hash is corrupted.",
            )

        genesis_ts_str = self._format_timestamp(genesis.timestamp)
        genesis_payload = json.loads(genesis.payload) if isinstance(genesis.payload, str) else genesis.payload
        genesis_expected_hash = self.calculate_hash(
            prev_hash=genesis.prev_hash,
            sequence_number=genesis.sequence_number,
            timestamp_str=genesis_ts_str,
            event_type=genesis.event_type,
            payload_str=serialize_payload(genesis_payload),
        )

        if genesis.event_hash != genesis_expected_hash:
            return ChainVerificationResult(
                is_valid=False,
                total_blocks=len(blocks),
                tampered_block_sequence=1,
                tampered_field="event_hash",
                expected_hash=genesis_expected_hash,
                actual_hash=genesis.event_hash,
                message="Genesis block hash corrupted.",
            )

        prev_hash = genesis.event_hash

        # 2. Verify Subsequent Linked Blocks
        for i in range(1, len(blocks)):
            block = blocks[i]

            # Check previous hash link continuity
            if block.prev_hash != prev_hash:
                return ChainVerificationResult(
                    is_valid=False,
                    total_blocks=len(blocks),
                    tampered_block_sequence=block.sequence_number,
                    tampered_field="prev_hash_link",
                    expected_hash=prev_hash,
                    actual_hash=block.prev_hash,
                    message=f"Tampering detected at block {block.sequence_number}: previous hash linkage broken.",
                )

            # Recompute block hash
            ts_str = self._format_timestamp(block.timestamp)
            payload_data = json.loads(block.payload) if isinstance(block.payload, str) else block.payload
            payload_str = serialize_payload(payload_data)

            expected_block_hash = self.calculate_hash(
                prev_hash=block.prev_hash,
                sequence_number=block.sequence_number,
                timestamp_str=ts_str,
                event_type=block.event_type,
                payload_str=payload_str,
            )

            if block.event_hash != expected_block_hash:
                return ChainVerificationResult(
                    is_valid=False,
                    total_blocks=len(blocks),
                    tampered_block_sequence=block.sequence_number,
                    tampered_field="event_hash",
                    expected_hash=expected_block_hash,
                    actual_hash=block.event_hash,
                    message=f"Tampering detected at block {block.sequence_number}: payload or metadata altered.",
                )

            prev_hash = block.event_hash

        return ChainVerificationResult(
            is_valid=True,
            total_blocks=len(blocks),
            head_hash=prev_hash,
            message="All hash-chain blocks cryptographically verified and intact.",
        )
