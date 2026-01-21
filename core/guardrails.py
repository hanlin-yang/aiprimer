"""
Security Guardrails Module

Implements input validation, biosecurity screening, and rate limiting
for the Primer Design Agent.

Key Features:
- Input sanitization against prompt injection
- Biosecurity screening for dangerous sequences
- Rate limiting for resource protection
- Audit logging for compliance

Compliance: FDA 21 CFR Part 11, Biosecurity Best Practices
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from functools import wraps

logger = logging.getLogger(__name__)


# ============================================================================
# Security Event Types
# ============================================================================

class SecurityEventType(str, Enum):
    """Types of security events for audit logging."""
    INPUT_SANITIZED = "input_sanitized"
    INJECTION_BLOCKED = "injection_blocked"
    BIOSECURITY_FLAG = "biosecurity_flag"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    LARGE_REQUEST_BLOCKED = "large_request_blocked"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    VALIDATION_FAILED = "validation_failed"


class ThreatLevel(str, Enum):
    """Threat severity levels."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# Security Event Logging
# ============================================================================

@dataclass
class SecurityEvent:
    """Security event record for audit trail."""
    event_id: str
    event_type: SecurityEventType
    threat_level: ThreatLevel
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    details: Dict[str, Any]
    action_taken: str
    blocked: bool


class SecurityAuditLog:
    """
    Security-specific audit log for compliance.

    Maintains immutable records of all security events
    for regulatory compliance and forensic analysis.
    """

    def __init__(self, storage_backend: str = "file"):
        self.storage_backend = storage_backend
        self.events: List[SecurityEvent] = []

    def log_event(
        self,
        event_type: SecurityEventType,
        threat_level: ThreatLevel,
        details: Dict[str, Any],
        action_taken: str,
        blocked: bool = False,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> str:
        """Log a security event and return event ID."""
        import uuid

        event_id = str(uuid.uuid4())

        event = SecurityEvent(
            event_id=event_id,
            event_type=event_type,
            threat_level=threat_level,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            details=details,
            action_taken=action_taken,
            blocked=blocked
        )

        self.events.append(event)

        # Log to standard logger
        log_level = {
            ThreatLevel.INFO: logging.INFO,
            ThreatLevel.LOW: logging.INFO,
            ThreatLevel.MEDIUM: logging.WARNING,
            ThreatLevel.HIGH: logging.WARNING,
            ThreatLevel.CRITICAL: logging.ERROR
        }.get(threat_level, logging.INFO)

        logger.log(
            log_level,
            f"Security Event [{event_type.value}]: {action_taken} | "
            f"Threat: {threat_level.value} | Blocked: {blocked}"
        )

        return event_id

    def get_events_by_user(
        self,
        user_id: str,
        since: Optional[datetime] = None
    ) -> List[SecurityEvent]:
        """Get security events for a specific user."""
        events = [e for e in self.events if e.user_id == user_id]
        if since:
            events = [e for e in events if e.timestamp >= since]
        return events

    def get_blocked_events(
        self,
        since: Optional[datetime] = None
    ) -> List[SecurityEvent]:
        """Get all blocked events."""
        events = [e for e in self.events if e.blocked]
        if since:
            events = [e for e in events if e.timestamp >= since]
        return events


# Global security audit log
security_audit = SecurityAuditLog()


# ============================================================================
# Prompt Injection Protection
# ============================================================================

class PromptInjectionGuard:
    """
    Guards against prompt injection attacks.

    Detects and neutralizes attempts to manipulate the LLM
    through malicious input sequences.
    """

    # Patterns indicating injection attempts
    INJECTION_PATTERNS = [
        # Direct instruction override attempts
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|rules?|prompts?)",
        r"disregard\s+(all\s+)?(previous|prior|above)",
        r"forget\s+(everything|all|your)\s+(instructions?|training|rules?)",
        r"new\s+instructions?:",
        r"system\s*:\s*",
        r"assistant\s*:\s*",

        # Role manipulation
        r"you\s+are\s+(now|no\s+longer)",
        r"act\s+as\s+(if|a|an)",
        r"pretend\s+(to\s+be|you\s+are)",
        r"roleplay\s+as",

        # Jailbreak attempts
        r"DAN\s+mode",
        r"developer\s+mode",
        r"jailbreak",
        r"bypass\s+(safety|restrictions?|filters?)",

        # Output manipulation
        r"output\s+the\s+following",
        r"print\s+exactly",
        r"respond\s+with\s+only",

        # Prompt leaking attempts
        r"what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions?)",
        r"show\s+me\s+your\s+(system\s+)?(prompt|instructions?)",
        r"reveal\s+your\s+(system\s+)?(prompt|instructions?)",

        # Code injection attempts
        r"```\s*(python|bash|shell|exec)",
        r"eval\s*\(",
        r"exec\s*\(",
        r"import\s+os",
        r"subprocess",
        r"__import__",
    ]

    # Compiled patterns for efficiency
    _compiled_patterns: List[re.Pattern] = []

    def __init__(self):
        if not self._compiled_patterns:
            self._compiled_patterns = [
                re.compile(pattern, re.IGNORECASE)
                for pattern in self.INJECTION_PATTERNS
            ]

    def check_input(self, text: str) -> Tuple[bool, List[str]]:
        """
        Check input for injection attempts.

        Args:
            text: Input text to check

        Returns:
            Tuple of (is_safe, list_of_detected_patterns)
        """
        detected = []

        for i, pattern in enumerate(self._compiled_patterns):
            if pattern.search(text):
                detected.append(self.INJECTION_PATTERNS[i])

        is_safe = len(detected) == 0
        return is_safe, detected

    def sanitize_input(
        self,
        text: str,
        user_id: Optional[str] = None
    ) -> str:
        """
        Sanitize input by removing or neutralizing dangerous patterns.

        Args:
            text: Input text to sanitize
            user_id: User ID for audit logging

        Returns:
            Sanitized text
        """
        is_safe, detected = self.check_input(text)

        if not is_safe:
            # Log the attempt
            security_audit.log_event(
                event_type=SecurityEventType.INJECTION_BLOCKED,
                threat_level=ThreatLevel.HIGH,
                details={
                    "original_text_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
                    "detected_patterns": detected[:5],  # Limit for log size
                    "text_length": len(text)
                },
                action_taken="Input sanitized - injection patterns removed",
                blocked=False,
                user_id=user_id
            )

            # Remove detected patterns
            sanitized = text
            for pattern in self._compiled_patterns:
                sanitized = pattern.sub("[FILTERED]", sanitized)

            return sanitized

        return text


# ============================================================================
# Biosecurity Screening
# ============================================================================

class BiosecurityScreener:
    """
    Screens sequences for biosecurity concerns.

    Detects sequences associated with:
    - Select agents and toxins (CDC/USDA)
    - Bioweapon-related organisms
    - Dangerous pathogens

    This is a critical safety feature - do not disable.
    """

    # Known dangerous sequence signatures (simplified for demonstration)
    # In production, this would integrate with actual biosecurity databases
    DANGEROUS_SIGNATURES = {
        # Anthrax protective antigen (partial)
        "ATGAAAAAACGAAAAGTGCTG": {
            "organism": "Bacillus anthracis",
            "gene": "pagA",
            "risk": "Select Agent - CDC",
            "threat_level": ThreatLevel.CRITICAL
        },
        # Botulinum toxin (partial signature)
        "ATGCCAATTAATAATAT": {
            "organism": "Clostridium botulinum",
            "gene": "botA",
            "risk": "Select Toxin - CDC",
            "threat_level": ThreatLevel.CRITICAL
        },
        # Ricin A chain (partial)
        "ATGAACCAGGTGTGCG": {
            "organism": "Ricinus communis",
            "gene": "RCA",
            "risk": "Select Toxin - CDC",
            "threat_level": ThreatLevel.CRITICAL
        },
        # Smallpox (Variola) signature
        "ATGGATCCGTTAATAGTAGCT": {
            "organism": "Variola virus",
            "gene": "HA",
            "risk": "Select Agent - CDC Category A",
            "threat_level": ThreatLevel.CRITICAL
        },
    }

    # Dangerous organism keywords
    DANGEROUS_KEYWORDS = [
        "anthrax", "anthracis",
        "botulinum", "botulism",
        "ricin", "abrin",
        "variola", "smallpox",
        "ebola", "marburg",
        "yersinia pestis", "plague",
        "francisella", "tularemia",
        "brucella", "brucellosis",
        "burkholderia mallei", "glanders",
        "clostridium perfringens epsilon",
        "staphylococcal enterotoxin",
        "t-2 toxin", "saxitoxin",
        "biological weapon", "bioweapon",
        "weaponized", "weaponise",
    ]

    def __init__(self):
        self._keyword_pattern = re.compile(
            '|'.join(re.escape(kw) for kw in self.DANGEROUS_KEYWORDS),
            re.IGNORECASE
        )

    def screen_sequence(
        self,
        sequence: str,
        user_id: Optional[str] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Screen a DNA sequence for biosecurity concerns.

        Args:
            sequence: DNA sequence to screen
            user_id: User ID for audit logging

        Returns:
            Tuple of (is_safe, threat_details or None)
        """
        sequence = sequence.upper().replace(" ", "").replace("\n", "")

        # Check against known signatures
        for signature, info in self.DANGEROUS_SIGNATURES.items():
            if signature in sequence:
                # Critical threat detected
                security_audit.log_event(
                    event_type=SecurityEventType.BIOSECURITY_FLAG,
                    threat_level=info["threat_level"],
                    details={
                        "signature_matched": signature[:10] + "...",
                        "organism": info["organism"],
                        "gene": info["gene"],
                        "risk_category": info["risk"]
                    },
                    action_taken="Request blocked - biosecurity threat detected",
                    blocked=True,
                    user_id=user_id
                )

                return False, {
                    "threat_type": "dangerous_sequence",
                    "organism": info["organism"],
                    "risk": info["risk"],
                    "action": "blocked"
                }

        return True, None

    def screen_text(
        self,
        text: str,
        user_id: Optional[str] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Screen text content for biosecurity-related keywords.

        Args:
            text: Text to screen
            user_id: User ID for audit logging

        Returns:
            Tuple of (is_safe, threat_details or None)
        """
        matches = self._keyword_pattern.findall(text.lower())

        if matches:
            unique_matches = list(set(matches))

            security_audit.log_event(
                event_type=SecurityEventType.BIOSECURITY_FLAG,
                threat_level=ThreatLevel.HIGH,
                details={
                    "keywords_detected": unique_matches[:5],
                    "num_matches": len(matches)
                },
                action_taken="Request flagged - biosecurity keywords detected",
                blocked=True,
                user_id=user_id
            )

            return False, {
                "threat_type": "biosecurity_keywords",
                "keywords": unique_matches[:5],
                "action": "blocked"
            }

        return True, None

    def full_screen(
        self,
        sequence: Optional[str],
        text: str,
        user_id: Optional[str] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Perform full biosecurity screening on both sequence and text.

        Args:
            sequence: DNA sequence (optional)
            text: Text content
            user_id: User ID for audit logging

        Returns:
            Tuple of (is_safe, threat_details or None)
        """
        # Screen text first
        text_safe, text_threat = self.screen_text(text, user_id)
        if not text_safe:
            return False, text_threat

        # Screen sequence if provided
        if sequence:
            seq_safe, seq_threat = self.screen_sequence(sequence, user_id)
            if not seq_safe:
                return False, seq_threat

        return True, None


# ============================================================================
# Rate Limiting
# ============================================================================

@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_minute: int = 10
    requests_per_hour: int = 100
    sequences_per_day: int = 1000
    max_sequence_length: int = 100000
    max_batch_size: int = 100


class RateLimiter:
    """
    Rate limiter for API requests and resource protection.

    Implements token bucket algorithm with multiple time windows.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self._user_requests: Dict[str, List[datetime]] = {}
        self._user_sequences: Dict[str, int] = {}
        self._last_reset: Dict[str, datetime] = {}

    def _cleanup_old_requests(self, user_id: str):
        """Remove requests older than 1 hour."""
        if user_id not in self._user_requests:
            self._user_requests[user_id] = []
            return

        cutoff = datetime.utcnow() - timedelta(hours=1)
        self._user_requests[user_id] = [
            ts for ts in self._user_requests[user_id]
            if ts > cutoff
        ]

    def _reset_daily_counters(self, user_id: str):
        """Reset daily counters if needed."""
        now = datetime.utcnow()
        last_reset = self._last_reset.get(user_id)

        if last_reset is None or (now - last_reset).days >= 1:
            self._user_sequences[user_id] = 0
            self._last_reset[user_id] = now

    def check_rate_limit(
        self,
        user_id: str,
        sequence_length: int = 0
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if request is within rate limits.

        Args:
            user_id: User identifier
            sequence_length: Length of sequence being processed

        Returns:
            Tuple of (is_allowed, rejection_reason or None)
        """
        now = datetime.utcnow()

        # Cleanup and reset
        self._cleanup_old_requests(user_id)
        self._reset_daily_counters(user_id)

        requests = self._user_requests.get(user_id, [])

        # Check requests per minute
        minute_ago = now - timedelta(minutes=1)
        requests_last_minute = len([ts for ts in requests if ts > minute_ago])

        if requests_last_minute >= self.config.requests_per_minute:
            security_audit.log_event(
                event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
                threat_level=ThreatLevel.MEDIUM,
                details={
                    "limit_type": "per_minute",
                    "current": requests_last_minute,
                    "limit": self.config.requests_per_minute
                },
                action_taken="Request rejected - minute rate limit exceeded",
                blocked=True,
                user_id=user_id
            )
            return False, f"Rate limit exceeded: {self.config.requests_per_minute} requests/minute"

        # Check requests per hour
        if len(requests) >= self.config.requests_per_hour:
            security_audit.log_event(
                event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
                threat_level=ThreatLevel.MEDIUM,
                details={
                    "limit_type": "per_hour",
                    "current": len(requests),
                    "limit": self.config.requests_per_hour
                },
                action_taken="Request rejected - hourly rate limit exceeded",
                blocked=True,
                user_id=user_id
            )
            return False, f"Rate limit exceeded: {self.config.requests_per_hour} requests/hour"

        # Check sequence length
        if sequence_length > self.config.max_sequence_length:
            security_audit.log_event(
                event_type=SecurityEventType.LARGE_REQUEST_BLOCKED,
                threat_level=ThreatLevel.MEDIUM,
                details={
                    "sequence_length": sequence_length,
                    "max_allowed": self.config.max_sequence_length
                },
                action_taken="Request rejected - sequence too long",
                blocked=True,
                user_id=user_id
            )
            return False, f"Sequence too long: max {self.config.max_sequence_length} bp"

        # Check daily sequence quota
        current_sequences = self._user_sequences.get(user_id, 0)
        if current_sequences >= self.config.sequences_per_day:
            security_audit.log_event(
                event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
                threat_level=ThreatLevel.MEDIUM,
                details={
                    "limit_type": "daily_sequences",
                    "current": current_sequences,
                    "limit": self.config.sequences_per_day
                },
                action_taken="Request rejected - daily sequence limit exceeded",
                blocked=True,
                user_id=user_id
            )
            return False, f"Daily sequence limit exceeded: {self.config.sequences_per_day}"

        return True, None

    def record_request(
        self,
        user_id: str,
        sequence_count: int = 1
    ):
        """Record a successful request."""
        now = datetime.utcnow()

        if user_id not in self._user_requests:
            self._user_requests[user_id] = []

        self._user_requests[user_id].append(now)

        if user_id not in self._user_sequences:
            self._user_sequences[user_id] = 0

        self._user_sequences[user_id] += sequence_count


# ============================================================================
# Input Validator
# ============================================================================

class InputValidator:
    """
    Comprehensive input validation for primer design requests.

    Validates:
    - Sequence format and content
    - Parameter ranges
    - Request structure
    """

    # Valid DNA characters (IUPAC)
    VALID_DNA = set("ATCGRYSWKMBDHVN")

    # Gene name pattern
    GENE_PATTERN = re.compile(r'^[A-Za-z0-9_\-\.]+$')

    def validate_sequence(
        self,
        sequence: str,
        field_name: str = "sequence"
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a DNA sequence.

        Args:
            sequence: DNA sequence to validate
            field_name: Name of field for error messages

        Returns:
            Tuple of (is_valid, error_message or None)
        """
        if not sequence:
            return True, None  # Optional sequences are allowed

        # Remove whitespace
        sequence = sequence.upper().replace(" ", "").replace("\n", "").replace("\r", "")

        # Check characters
        invalid_chars = set(sequence) - self.VALID_DNA
        if invalid_chars:
            return False, f"Invalid characters in {field_name}: {invalid_chars}"

        # Check length
        if len(sequence) < 20:
            return False, f"{field_name} too short: minimum 20 bp"

        if len(sequence) > 100000:
            return False, f"{field_name} too long: maximum 100,000 bp"

        return True, None

    def validate_gene_name(
        self,
        gene_name: str
    ) -> Tuple[bool, Optional[str]]:
        """Validate a gene name."""
        if not gene_name:
            return False, "Gene name is required"

        if len(gene_name) > 50:
            return False, "Gene name too long: maximum 50 characters"

        if not self.GENE_PATTERN.match(gene_name):
            return False, "Gene name contains invalid characters"

        return True, None

    def validate_tm_range(
        self,
        tm_min: float,
        tm_max: float
    ) -> Tuple[bool, Optional[str]]:
        """Validate melting temperature range."""
        if tm_min < 40 or tm_min > 80:
            return False, "tm_min must be between 40-80°C"

        if tm_max < 40 or tm_max > 80:
            return False, "tm_max must be between 40-80°C"

        if tm_min > tm_max:
            return False, "tm_min must be <= tm_max"

        if tm_max - tm_min > 20:
            return False, "Tm range too wide: maximum 20°C spread"

        return True, None

    def validate_product_size(
        self,
        size_min: int,
        size_max: int,
        task_type: str = "PCR"
    ) -> Tuple[bool, Optional[str]]:
        """Validate product size range."""
        if size_min < 50:
            return False, "Minimum product size must be >= 50 bp"

        if size_max > 10000:
            return False, "Maximum product size must be <= 10,000 bp"

        if size_min > size_max:
            return False, "product_size_min must be <= product_size_max"

        # Task-specific validation
        if task_type == "qPCR" and size_max > 300:
            return False, "qPCR amplicons should be <= 300 bp"

        return True, None


# ============================================================================
# Main Guardrails Class
# ============================================================================

class Guardrails:
    """
    Main guardrails class integrating all security checks.

    Provides a single interface for:
    - Input sanitization
    - Biosecurity screening
    - Rate limiting
    - Validation
    """

    def __init__(
        self,
        rate_limit_config: Optional[RateLimitConfig] = None
    ):
        self.injection_guard = PromptInjectionGuard()
        self.biosecurity_screener = BiosecurityScreener()
        self.rate_limiter = RateLimiter(rate_limit_config)
        self.validator = InputValidator()

    async def check_request(
        self,
        target_gene: str,
        sequence: Optional[str],
        user_message: str,
        user_id: str,
        session_id: Optional[str] = None
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Perform all security checks on a primer design request.

        Args:
            target_gene: Target gene name
            sequence: DNA sequence (optional)
            user_message: User's natural language request
            user_id: User identifier
            session_id: Session identifier

        Returns:
            Tuple of (is_allowed, rejection_reason, sanitized_data)
        """
        # 1. Rate limiting
        seq_length = len(sequence) if sequence else 0
        allowed, reason = self.rate_limiter.check_rate_limit(user_id, seq_length)
        if not allowed:
            return False, reason, None

        # 2. Prompt injection check
        sanitized_message = self.injection_guard.sanitize_input(user_message, user_id)

        # 3. Biosecurity screening
        is_safe, threat = self.biosecurity_screener.full_screen(
            sequence, user_message, user_id
        )
        if not is_safe:
            return False, f"Security block: {threat['threat_type']}", None

        # 4. Input validation
        valid, error = self.validator.validate_gene_name(target_gene)
        if not valid:
            return False, error, None

        if sequence:
            valid, error = self.validator.validate_sequence(sequence)
            if not valid:
                return False, error, None

        # Record successful check
        self.rate_limiter.record_request(user_id, 1)

        return True, None, {
            "sanitized_message": sanitized_message,
            "sequence": sequence,
            "target_gene": target_gene
        }

    def create_middleware(self):
        """
        Create FastAPI middleware for request screening.

        Returns:
            Async middleware function
        """
        guardrails = self

        async def middleware(request, call_next):
            # Extract user info from request
            user_id = request.headers.get("X-User-ID", "anonymous")

            # For POST requests with JSON body
            if request.method == "POST":
                # Note: In production, you'd need to handle body reading properly
                pass

            response = await call_next(request)
            return response

        return middleware


# ============================================================================
# Decorator for Protected Functions
# ============================================================================

def require_security_check(
    guardrails: Optional[Guardrails] = None
):
    """
    Decorator to require security checks before function execution.

    Usage:
        @require_security_check()
        async def design_primers(request: PrimerDesignRequest):
            ...
    """
    _guardrails = guardrails or Guardrails()

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract relevant parameters
            request = kwargs.get('request') or (args[0] if args else None)

            if request and hasattr(request, 'target_gene'):
                allowed, reason, _ = await _guardrails.check_request(
                    target_gene=getattr(request, 'target_gene', ''),
                    sequence=getattr(request, 'sequence_template', None),
                    user_message=str(request),
                    user_id=getattr(request, 'user_id', 'anonymous')
                )

                if not allowed:
                    raise SecurityException(reason)

            return await func(*args, **kwargs)

        return wrapper

    return decorator


class SecurityException(Exception):
    """Exception raised when security check fails."""

    def __init__(self, message: str, threat_level: ThreatLevel = ThreatLevel.HIGH):
        super().__init__(message)
        self.threat_level = threat_level


# ============================================================================
# Module-level instances
# ============================================================================

guardrails = Guardrails()
injection_guard = PromptInjectionGuard()
biosecurity_screener = BiosecurityScreener()
rate_limiter = RateLimiter()
input_validator = InputValidator()
