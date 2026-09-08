"""
Audit API Routes
Endpoints for inspecting the tamper-evident hash-chain audit log and verifying cryptographic integrity.
"""

import json
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.database.session import get_db
from backend.app.database.repositories import AuditRepository
from backend.app.api.schemas import AuditChainResponse, AuditVerifyResponse, AuditBlockItem
from backend.app.audit.hash_chain import HashChainService

router = APIRouter()


@router.get("/chain", response_model=AuditChainResponse, summary="Inspect tamper-evident hash chain")
async def get_audit_chain(limit: int = 50, db: AsyncSession = Depends(get_db)):
    """Returns recent hash-chain blocks containing metadata only (zero audio persisted)."""
    repo = AuditRepository(db)
    blocks = await repo.get_all(limit=limit)

    items: List[AuditBlockItem] = []
    for b in blocks:
        payload_dict = json.loads(b.payload) if isinstance(b.payload, str) else b.payload
        items.append(
            AuditBlockItem(
                sequence_number=b.sequence_number,
                timestamp=b.timestamp.isoformat() if hasattr(b.timestamp, "isoformat") else str(b.timestamp),
                event_type=b.event_type,
                session_id=b.session_id,
                prev_hash=b.prev_hash,
                event_hash=b.event_hash,
                payload=payload_dict,
            )
        )

    return AuditChainResponse(total_blocks=len(items), blocks=items)


@router.get("/verify", response_model=AuditVerifyResponse, summary="Cryptographically verify hash-chain integrity")
async def verify_chain_integrity(db: AsyncSession = Depends(get_db)):
    """
    Recomputes SHA-256 hashes from Genesis to Head.
    Detects any retroactive tampering, modification, or deletion deterministically.
    """
    svc = HashChainService(db)
    res = await svc.verify_chain()

    return AuditVerifyResponse(
        is_valid=res.is_valid,
        total_blocks=res.total_blocks,
        head_hash=res.head_hash,
        message=res.message,
        tampered_block_sequence=res.tampered_block_sequence,
        tampered_field=res.tampered_field,
    )
