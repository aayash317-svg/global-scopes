# Voice Integrity Verification Framework
**Problem Statement ID:** 26104  
**Organization:** AICTE Cyber Security Cell  
**Category:** Software | **Theme:** Blockchain & Cybersecurity  

An end-to-end, near-real-time voice authenticity and impersonation defense framework that analyzes live voice streams, computes dynamic fraud risk scores, and issues automated step-up verification alerts before sensitive actions occur.

---

## Architecture Overview (5 Layers, 4 Explicit Layer-1 Signals)

```
┌────────────────────────────────────────────────────────────────────────┐
│              Layer 5: Edge Resilience & Platform API                   │
│      REST Endpoints, WebSocket Streaming (/stream), Offline Edge Queue │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│           Layer 4: DPDP Act 2023 Compliance & Audit Integrity          │
│   • Data Minimisation (0 raw audio persisted, in-memory zeroization)   │
│   • Biometric Fernet Encryption (At-rest AES-128-CBC + HMAC-SHA256)    │
│   • Tamper-Evident Hash Chain (SHA-256 block linking, instant audit)   │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│             Layer 3: Alerting & Step-Up Verification Layer             │
│   • Dynamic Voice Challenge Prompts (prevents replay/static clones)     │
│   • Multi-Channel MFA OTP Dispatch (SMS, Email, Push)                  │
│   • Supervisor Takeover Escalation & Out-of-Band Callbacks             │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                Layer 2: Real-Time Risk Scoring Engine                  │
│   • Contextual Modulation (Call origin, known caller, transaction val) │
│   • Scenario Decision Boundaries (Routine, High-Value, Privileged)     │
│   • Mandatory Safety Floor Rule (>=90% -> MEDIUM, >=97% -> HIGH)       │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│            Layer 1: Multi-Layer Voice Authenticity Analysis            │
│   1. Acoustic: Phase jitter, instantaneous phase variance, kurtosis    │
│   2. Spectral: Spectral flatness, centroid, rolloff, HF ratio          │
│   3. Prosody: Pitch (F0) contour std, micro-jitter, amplitude shimmer  │
│   4. Speaker Consistency: ECAPA/MFCC embedding vs KYC voiceprint       │
│   + Voice Activity Gate (VAD) & Replay Attack Transducer Analysis      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## DPDP Act 2023 Compliance Mapping

| DPDP Principle | Framework Guarantee |
| :--- | :--- |
| **Data Minimisation** | Raw audio is processed transiently in-memory per 3.5s window and zeroed out immediately; zero audio files/recordings ever hit disk or database. |
| **Storage Limitation** | Transient audio discarded immediately; session metadata subject to automated 30-day retention policies (`/privacy/retention/apply`). |
| **Purpose Limitation** | Mathematical embeddings are used strictly for real-time fraud defense; never repurposed. |
| **Consent (Sec. 6)** | Mandatory explicit consent notice required and logged before voiceprint enrollment at KYC stage. |
| **Security Safeguards** | Voiceprints stored using Fernet symmetric authenticated encryption; security events protected by cryptographic hash chain. |
| **Right to Erasure (Sec. 12)** | `DELETE /api/v1/privacy/user/{id}` executes immediate permanent cryptographic purge of all biometric data. |

---

## Blockchain Theme: Tamper-Evident Hash Chain

Every security event (`VOICEPRINT_ENROLLED`, `RISK_EVALUATED`, `ALERT_TRIGGERED`, `CHALLENGE_ISSUED`) is hashed and cryptographically linked to its predecessor via SHA-256:
$$\text{Event Hash} = \text{SHA256}(\text{Prev Hash} \parallel \text{Seq} \parallel \text{Timestamp} \parallel \text{Event Type} \parallel \text{Payload JSON})$$

- **Deterministic Verification:** `GET /audit/verify` recomputes the entire chain from Genesis to Head in milliseconds.
- **Tamper Detection Proof:** Modifying even a single field in a historical database record breaks the cryptographic chain deterministically and pinpoints the exact tampered sequence number and field.

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | System health, database connectivity, and runtime metrics |
| `POST` | `/enroll` | Enroll genuine KYC voiceprint (extracts embedding, encrypts, discards raw audio) |
| `POST` | `/analyze` | Full 5-layer pipeline: VAD -> 4 Layer-1 signals -> Risk scoring -> Alert -> Audit log |
| `WS` | `/stream` | Real-time WebSocket audio streaming for live telephony/VoIP streams |
| `GET` | `/audit/chain` | Retrieve recent tamper-evident hash-chain blocks |
| `GET` | `/audit/verify` | End-to-end cryptographic verification of audit trail integrity |
| `GET` | `/compliance` | Active DPDP Act 2023 compliance status and privacy mapping |
| `POST` | `/api/v1/audio/validate` | Audio SNR, quality metrics, and input validation |
| `POST` | `/api/v1/audio/telephony-simulate` | Emulates G.711 A-law/μ-law 8kHz PSTN channel degradation |
| `POST` | `/api/v1/risk/assess` | Standalone risk calculation with scenario thresholds & safety floor |
| `GET` | `/api/v1/risk/thresholds` | Active decision thresholds across operational scenarios |
| `POST` | `/api/v1/verification/challenge` | Issues dynamic unpredictable voice challenge prompt |
| `POST` | `/api/v1/verification/challenge/verify` | Verifies spoken challenge response |
| `POST` | `/api/v1/verification/mfa/dispatch` | Dispatches secondary OTP step-up authentication alert |
| `POST` | `/api/v1/verification/mfa/verify` | Verifies claimant submitted OTP |
| `POST` | `/api/v1/verification/callback` | Handles out-of-band resolution callback |
| `DELETE` | `/api/v1/privacy/user/{id}` | Permanently erases user's encrypted biometric profile |
| `GET` | `/edge/status` | Edge node status and pending sync queue |
| `POST` | `/edge/sync` | Flushes queued offline events to persistent hash chain |

---

## Quickstart & Installation

### 1. Local Setup
```bash
# Clone and enter project root
cd "global scopes"

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run database setup and startup
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Docker Deployment
```bash
docker-compose up --build -d
```

### 3. Run Automated Test Suite
```bash
pytest backend/tests -v
```

---

## ML Models & Production Upgrade Path

| Subsystem | Current State (Built-in) | Production Upgrade Path |
| :--- | :--- | :--- |
| **Spoof Classifier** | Algorithmic Acoustic & Spectral Analyzers (Phase Jitter, Kurtosis, Spectral Flatness, Rolloff, HF ratio) | Fine-tune AASIST / RawNet2 / WavLM weights on ASVspoof 2019/2021 + IndicSynth |
| **Speaker Consistency** | 78-dimensional Mel-Frequency Cepstral Coefficients + Delta-Delta Statistical Vector | Train ECAPA-TDNN / x-vector on VoxCeleb 1/2 |
| **Replay Detection** | Acoustic reverberation decay, spectral flux, and loudspeaker band-cutoff analyzer | Deep ResNet replay detector trained on ASVspoof 2017/2019 Physical Access (PA) |
| **Biometric Encryption** | Real Fernet symmetric authenticated encryption | KMS / HSM envelope encryption with key rotation policy |
| **Tamper Evidence** | Real SHA-256 cryptographic hash-chain with deterministic verify | Durable persistent backend; periodic root anchoring to public blockchain/SIEM |
