"""
Primer Design Pydantic Schemas

Enterprise-grade schema definitions for primer design requests and responses.
Compliant with FDA 21 CFR Part 11 audit trail requirements.
"""

from enum import Enum
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator
import uuid


class TaskType(str, Enum):
    """Supported primer design task types."""
    QPCR = "qPCR"
    PCR = "PCR"
    LAMP = "LAMP"
    NGS = "NGS"


class ErrorCode(str, Enum):
    """Standardized error codes for primer design failures.

    Error codes follow the pattern: ERR_{CATEGORY}_{SPECIFIC}
    These codes enable automated error handling and user guidance.
    """
    # Thermodynamic errors
    ERR_DIMER_RISK = "ERR_DIMER_RISK"  # Delta G < -9.0 kcal/mol
    ERR_HAIRPIN_RISK = "ERR_HAIRPIN_RISK"  # Significant hairpin structure detected
    ERR_TM_MISMATCH = "ERR_TM_MISMATCH"  # Tm difference > 5°C between primers
    ERR_TM_OUT_OF_RANGE = "ERR_TM_OUT_OF_RANGE"  # Tm outside acceptable range
    ERR_GC_OUT_OF_RANGE = "ERR_GC_OUT_OF_RANGE"  # GC% outside 40-60%

    # Variant/SNP errors
    ERR_SNP_CONFLICT = "ERR_SNP_CONFLICT"  # High-frequency SNP in primer binding site
    ERR_VARIANT_HOTSPOT = "ERR_VARIANT_HOTSPOT"  # Multiple variants in primer region

    # Specificity errors
    ERR_OFF_TARGET = "ERR_OFF_TARGET"  # BLAST detected off-target binding
    ERR_LOW_SPECIFICITY = "ERR_LOW_SPECIFICITY"  # Multiple genomic hits
    ERR_PSEUDOGENE_HIT = "ERR_PSEUDOGENE_HIT"  # Pseudogene cross-reactivity

    # Sequence errors
    ERR_INVALID_SEQUENCE = "ERR_INVALID_SEQUENCE"  # Non-ATCG characters
    ERR_SEQUENCE_TOO_SHORT = "ERR_SEQUENCE_TOO_SHORT"  # Template < 50bp
    ERR_SEQUENCE_TOO_LONG = "ERR_SEQUENCE_TOO_LONG"  # Template > 10kb
    ERR_HIGH_GC_REGION = "ERR_HIGH_GC_REGION"  # GC content > 70%
    ERR_LOW_COMPLEXITY = "ERR_LOW_COMPLEXITY"  # Repetitive sequence detected

    # System errors
    ERR_BLAST_TIMEOUT = "ERR_BLAST_TIMEOUT"  # BLAST service timeout
    ERR_PRIMER3_FAILURE = "ERR_PRIMER3_FAILURE"  # Primer3 computation failed
    ERR_NO_VALID_PRIMERS = "ERR_NO_VALID_PRIMERS"  # No primers met criteria

    # Security errors
    ERR_SECURITY_BLOCK = "ERR_SECURITY_BLOCK"  # Biosecurity flag triggered
    ERR_RATE_LIMIT = "ERR_RATE_LIMIT"  # Too many requests


class DesignConstraints(BaseModel):
    """Thermodynamic and structural constraints for primer design."""

    tm_min: float = Field(
        default=55.0,
        ge=40.0,
        le=75.0,
        description="Minimum melting temperature (°C)"
    )
    tm_max: float = Field(
        default=60.0,
        ge=40.0,
        le=75.0,
        description="Maximum melting temperature (°C)"
    )
    tm_optimal: Optional[float] = Field(
        default=57.5,
        description="Optimal melting temperature (°C)"
    )

    gc_min: float = Field(
        default=40.0,
        ge=20.0,
        le=80.0,
        description="Minimum GC content (%)"
    )
    gc_max: float = Field(
        default=60.0,
        ge=20.0,
        le=80.0,
        description="Maximum GC content (%)"
    )

    gc_clamp: bool = Field(
        default=True,
        description="Require G/C at 3' end (GC clamp)"
    )
    gc_clamp_length: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Number of G/C bases required at 3' end"
    )

    primer_length_min: int = Field(
        default=18,
        ge=15,
        le=35,
        description="Minimum primer length (bp)"
    )
    primer_length_max: int = Field(
        default=25,
        ge=15,
        le=35,
        description="Maximum primer length (bp)"
    )
    primer_length_optimal: int = Field(
        default=20,
        description="Optimal primer length (bp)"
    )

    product_size_min: int = Field(
        default=75,
        ge=50,
        le=5000,
        description="Minimum amplicon size (bp)"
    )
    product_size_max: int = Field(
        default=200,
        ge=50,
        le=5000,
        description="Maximum amplicon size (bp)"
    )

    max_poly_x: int = Field(
        default=4,
        ge=2,
        le=6,
        description="Maximum length of mononucleotide repeat"
    )

    max_self_complementarity: int = Field(
        default=8,
        description="Maximum self-complementarity score"
    )

    max_end_complementarity: int = Field(
        default=3,
        description="Maximum 3' end complementarity score"
    )

    # Sodium and magnesium concentrations for Tm calculation
    na_concentration: float = Field(
        default=50.0,
        ge=0.0,
        le=1000.0,
        description="Sodium concentration (mM)"
    )
    mg_concentration: float = Field(
        default=1.5,
        ge=0.0,
        le=10.0,
        description="Magnesium concentration (mM)"
    )
    dntp_concentration: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="dNTP concentration (mM)"
    )

    @model_validator(mode='after')
    def validate_ranges(self) -> 'DesignConstraints':
        """Ensure min <= max for all range constraints."""
        if self.tm_min > self.tm_max:
            raise ValueError(f"tm_min ({self.tm_min}) must be <= tm_max ({self.tm_max})")
        if self.gc_min > self.gc_max:
            raise ValueError(f"gc_min ({self.gc_min}) must be <= gc_max ({self.gc_max})")
        if self.primer_length_min > self.primer_length_max:
            raise ValueError(
                f"primer_length_min ({self.primer_length_min}) must be <= "
                f"primer_length_max ({self.primer_length_max})"
            )
        if self.product_size_min > self.product_size_max:
            raise ValueError(
                f"product_size_min ({self.product_size_min}) must be <= "
                f"product_size_max ({self.product_size_max})"
            )
        return self


class PrimerDesignRequest(BaseModel):
    """
    Input schema for primer design requests.

    Supports multiple task types (qPCR, PCR, LAMP, NGS) with automatic
    validation of constraints based on the selected task type.
    """

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier for audit trail"
    )

    target_gene: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Target gene symbol (e.g., GAPDH, ACE2)"
    )

    sequence_template: Optional[str] = Field(
        default=None,
        description="DNA template sequence (ATCG only). If not provided, will fetch from database."
    )

    target_region_start: Optional[int] = Field(
        default=None,
        ge=0,
        description="Start position of target region within template"
    )

    target_region_end: Optional[int] = Field(
        default=None,
        ge=0,
        description="End position of target region within template"
    )

    task_type: TaskType = Field(
        default=TaskType.PCR,
        description="Type of primer design task"
    )

    constraints: DesignConstraints = Field(
        default_factory=DesignConstraints,
        description="Design constraints and parameters"
    )

    num_primers_return: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of primer pairs to return"
    )

    check_specificity: bool = Field(
        default=True,
        description="Run BLAST specificity check"
    )

    genome_database: str = Field(
        default="hg38",
        description="Reference genome for specificity check"
    )

    check_variants: bool = Field(
        default=True,
        description="Check for SNPs/variants in primer binding sites"
    )

    variant_maf_threshold: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Minor allele frequency threshold for variant flagging"
    )

    include_probe: bool = Field(
        default=False,
        description="Design internal probe (for qPCR TaqMan assays)"
    )

    # Audit trail fields
    user_id: Optional[str] = Field(
        default=None,
        description="User identifier for audit trail"
    )

    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier for audit trail"
    )

    @field_validator('sequence_template')
    @classmethod
    def validate_sequence(cls, v: Optional[str]) -> Optional[str]:
        """Validate DNA sequence contains only valid nucleotides."""
        if v is None:
            return v

        # Convert to uppercase and remove whitespace
        v = v.upper().replace(" ", "").replace("\n", "").replace("\r", "")

        # Check for valid nucleotides (ATCG + IUPAC ambiguity codes)
        valid_chars = set("ATCGRYSWKMBDHVN")
        invalid_chars = set(v) - valid_chars

        if invalid_chars:
            raise ValueError(
                f"Invalid characters in sequence: {invalid_chars}. "
                "Only ATCG and IUPAC ambiguity codes are allowed."
            )

        # Length validation
        if len(v) < 50:
            raise ValueError(
                f"Sequence too short ({len(v)} bp). Minimum length is 50 bp."
            )

        if len(v) > 100000:
            raise ValueError(
                f"Sequence too long ({len(v)} bp). Maximum length is 100,000 bp. "
                "For larger sequences, please specify a target region."
            )

        return v

    @field_validator('target_gene')
    @classmethod
    def validate_gene_name(cls, v: str) -> str:
        """Sanitize gene name input."""
        # Remove potentially dangerous characters
        sanitized = ''.join(c for c in v if c.isalnum() or c in '-_.')
        if len(sanitized) != len(v):
            raise ValueError(
                "Gene name contains invalid characters. "
                "Only alphanumeric, hyphen, underscore, and period are allowed."
            )
        return sanitized.upper()

    @model_validator(mode='after')
    def validate_qpcr_constraints(self) -> 'PrimerDesignRequest':
        """
        Validate task-specific constraints.

        qPCR amplicons should be 70-300 bp for optimal efficiency.
        LAMP requires specific primer set configurations.
        """
        if self.task_type == TaskType.QPCR:
            if self.constraints.product_size_max > 300:
                raise ValueError(
                    f"qPCR amplicon size ({self.constraints.product_size_max} bp) exceeds "
                    "recommended maximum of 300 bp. For optimal qPCR efficiency, "
                    "amplicons should be 70-300 bp. Consider reducing product_size_max."
                )
            if self.constraints.product_size_min < 70:
                raise ValueError(
                    f"qPCR amplicon size ({self.constraints.product_size_min} bp) below "
                    "recommended minimum of 70 bp. This may affect quantification accuracy."
                )

        if self.task_type == TaskType.LAMP:
            # LAMP has different requirements
            if self.constraints.primer_length_min < 18:
                raise ValueError(
                    "LAMP primers require minimum length of 18 bp for stability."
                )

        if self.task_type == TaskType.NGS:
            # NGS typically needs longer products
            if self.constraints.product_size_max < 150:
                raise ValueError(
                    "NGS library prep typically requires amplicons >= 150 bp "
                    "to accommodate adapter ligation."
                )

        # Validate target region
        if self.target_region_start is not None and self.target_region_end is not None:
            if self.target_region_start >= self.target_region_end:
                raise ValueError(
                    f"target_region_start ({self.target_region_start}) must be < "
                    f"target_region_end ({self.target_region_end})"
                )

            region_size = self.target_region_end - self.target_region_start
            if region_size < self.constraints.product_size_min:
                raise ValueError(
                    f"Target region ({region_size} bp) is smaller than "
                    f"minimum product size ({self.constraints.product_size_min} bp)"
                )

        return self


class PrimerCandidate(BaseModel):
    """Individual primer sequence with calculated properties."""

    sequence: str = Field(
        ...,
        description="Primer sequence (5' to 3')"
    )

    length: int = Field(
        ...,
        description="Primer length (bp)"
    )

    start_position: int = Field(
        ...,
        description="Start position on template (0-indexed)"
    )

    end_position: int = Field(
        ...,
        description="End position on template (0-indexed)"
    )

    tm: float = Field(
        ...,
        description="Melting temperature (°C) calculated using Nearest-Neighbor method"
    )

    gc_percent: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="GC content (%)"
    )

    self_complementarity: float = Field(
        default=0.0,
        description="Self-complementarity score (Primer3)"
    )

    end_complementarity: float = Field(
        default=0.0,
        description="3' end self-complementarity score"
    )

    hairpin_tm: Optional[float] = Field(
        default=None,
        description="Hairpin structure melting temperature (°C)"
    )

    delta_g: Optional[float] = Field(
        default=None,
        description="Gibbs free energy (kcal/mol) at 37°C"
    )


class PrimerPair(BaseModel):
    """A pair of forward and reverse primers with combined analysis."""

    pair_id: int = Field(
        ...,
        description="Primer pair ranking (0 = best)"
    )

    forward: PrimerCandidate = Field(
        ...,
        description="Forward (left) primer"
    )

    reverse: PrimerCandidate = Field(
        ...,
        description="Reverse (right) primer"
    )

    probe: Optional[PrimerCandidate] = Field(
        default=None,
        description="Internal probe (for TaqMan qPCR)"
    )

    product_size: int = Field(
        ...,
        description="Expected amplicon size (bp)"
    )

    tm_difference: float = Field(
        ...,
        description="Tm difference between forward and reverse primers (°C)"
    )

    pair_complementarity: float = Field(
        default=0.0,
        description="Inter-primer complementarity score"
    )

    end_complementarity: float = Field(
        default=0.0,
        description="3' end inter-primer complementarity"
    )

    dimer_risk_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Primer dimer formation risk (0=low, 1=high)"
    )

    dimer_delta_g: Optional[float] = Field(
        default=None,
        description="Most stable dimer structure Delta G (kcal/mol)"
    )

    requires_optimization: bool = Field(
        default=False,
        description="Flag indicating primer pair needs further optimization"
    )

    penalty_score: float = Field(
        default=0.0,
        description="Combined penalty score from Primer3"
    )


class SpecificityResult(BaseModel):
    """BLAST specificity check results."""

    is_specific: bool = Field(
        ...,
        description="True if no significant off-target hits detected"
    )

    num_perfect_matches: int = Field(
        default=1,
        description="Number of perfect match sites in genome"
    )

    off_target_hits: List[dict] = Field(
        default_factory=list,
        description="List of off-target binding sites with details"
    )

    pseudogene_hits: List[str] = Field(
        default_factory=list,
        description="List of pseudogene accessions with cross-reactivity"
    )

    specificity_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Overall specificity score (1=highly specific)"
    )


class StructuralWarning(BaseModel):
    """Structural or thermodynamic warning for a primer pair."""

    code: ErrorCode = Field(
        ...,
        description="Warning/error code"
    )

    severity: str = Field(
        default="warning",
        description="Severity level: info, warning, error, critical"
    )

    message: str = Field(
        ...,
        description="Human-readable warning message"
    )

    affected_primer: Optional[str] = Field(
        default=None,
        description="Which primer is affected: forward, reverse, probe, or both"
    )

    suggestion: Optional[str] = Field(
        default=None,
        description="Suggested remediation action"
    )

    details: Optional[dict] = Field(
        default=None,
        description="Additional technical details"
    )


class PrimerDesignResponse(BaseModel):
    """
    Output schema for primer design results.

    Contains structured output with computed primers, risk assessments,
    and AI-generated suggestions for optimization.
    """

    request_id: str = Field(
        ...,
        description="Original request ID for traceability"
    )

    status: str = Field(
        default="success",
        description="Response status: success, partial, failed"
    )

    task_type: TaskType = Field(
        ...,
        description="Task type from original request"
    )

    target_gene: str = Field(
        ...,
        description="Target gene from original request"
    )

    primer_pairs: List[PrimerPair] = Field(
        default_factory=list,
        description="List of designed primer pairs, ranked by quality"
    )

    best_pair: Optional[PrimerPair] = Field(
        default=None,
        description="Recommended best primer pair"
    )

    specificity_results: Optional[SpecificityResult] = Field(
        default=None,
        description="BLAST specificity analysis results"
    )

    structural_warnings: List[StructuralWarning] = Field(
        default_factory=list,
        description="List of structural and thermodynamic warnings"
    )

    variant_warnings: List[dict] = Field(
        default_factory=list,
        description="SNP/variant conflicts in primer binding sites"
    )

    ai_suggestion: Optional[str] = Field(
        default=None,
        description="AI-generated optimization suggestions based on analysis"
    )

    optimization_history: List[dict] = Field(
        default_factory=list,
        description="History of optimization iterations (for ReAct loop tracing)"
    )

    error_codes: List[ErrorCode] = Field(
        default_factory=list,
        description="List of error codes encountered during design"
    )

    error_message: Optional[str] = Field(
        default=None,
        description="Detailed error message if design failed"
    )

    # Audit trail fields
    computation_time_ms: Optional[int] = Field(
        default=None,
        description="Total computation time in milliseconds"
    )

    primer3_version: Optional[str] = Field(
        default=None,
        description="Primer3 library version used"
    )

    blast_database: Optional[str] = Field(
        default=None,
        description="BLAST database used for specificity check"
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response creation timestamp (UTC)"
    )

    audit_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of results for integrity verification"
    )

    def has_critical_warnings(self) -> bool:
        """Check if any critical warnings exist."""
        return any(w.severity == "critical" for w in self.structural_warnings)

    def get_warnings_by_severity(self, severity: str) -> List[StructuralWarning]:
        """Filter warnings by severity level."""
        return [w for w in self.structural_warnings if w.severity == severity]


class BatchPrimerDesignRequest(BaseModel):
    """Batch primer design request for multiple targets."""

    batch_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique batch identifier"
    )

    requests: List[PrimerDesignRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of individual primer design requests"
    )

    priority: str = Field(
        default="normal",
        description="Processing priority: low, normal, high"
    )

    callback_url: Optional[str] = Field(
        default=None,
        description="Webhook URL for completion notification"
    )


class BatchPrimerDesignResponse(BaseModel):
    """Batch primer design response."""

    batch_id: str = Field(
        ...,
        description="Batch identifier from request"
    )

    status: str = Field(
        default="processing",
        description="Batch status: queued, processing, completed, failed"
    )

    total_requests: int = Field(
        ...,
        description="Total number of requests in batch"
    )

    completed_requests: int = Field(
        default=0,
        description="Number of completed requests"
    )

    failed_requests: int = Field(
        default=0,
        description="Number of failed requests"
    )

    results: List[PrimerDesignResponse] = Field(
        default_factory=list,
        description="List of individual design results"
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Batch creation timestamp"
    )

    completed_at: Optional[datetime] = Field(
        default=None,
        description="Batch completion timestamp"
    )
