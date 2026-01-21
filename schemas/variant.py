"""
Variant/SNP Pydantic Schemas

Schema definitions for variant checking and SNP conflict detection.
Integrates with dbSNP and GISAID data sources.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import uuid


class VariantSource(str, Enum):
    """Data sources for variant information."""
    DBSNP = "dbSNP"
    CLINVAR = "ClinVar"
    GISAID = "GISAID"
    GNOMAD = "gnomAD"
    COSMIC = "COSMIC"
    CUSTOM = "custom"


class VariantType(str, Enum):
    """Types of genetic variants."""
    SNP = "SNP"  # Single nucleotide polymorphism
    INDEL = "INDEL"  # Insertion or deletion
    MNP = "MNP"  # Multi-nucleotide polymorphism
    CNV = "CNV"  # Copy number variant
    SV = "SV"  # Structural variant


class ClinicalSignificance(str, Enum):
    """Clinical significance classifications."""
    BENIGN = "benign"
    LIKELY_BENIGN = "likely_benign"
    UNCERTAIN = "uncertain_significance"
    LIKELY_PATHOGENIC = "likely_pathogenic"
    PATHOGENIC = "pathogenic"
    DRUG_RESPONSE = "drug_response"
    NOT_PROVIDED = "not_provided"


class PopulationFrequency(BaseModel):
    """Allele frequency data across populations."""

    population: str = Field(
        ...,
        description="Population identifier (e.g., AFR, EUR, EAS, SAS, AMR)"
    )

    allele_frequency: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Minor allele frequency in this population"
    )

    allele_count: Optional[int] = Field(
        default=None,
        description="Number of alleles observed"
    )

    total_alleles: Optional[int] = Field(
        default=None,
        description="Total number of alleles in population"
    )


class Variant(BaseModel):
    """
    Individual variant record.

    Contains variant position, alleles, frequency data, and clinical annotations.
    """

    variant_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Internal variant identifier"
    )

    rsid: Optional[str] = Field(
        default=None,
        description="dbSNP rsID (e.g., rs12345)"
    )

    chromosome: str = Field(
        ...,
        description="Chromosome (e.g., chr1, chrX)"
    )

    position: int = Field(
        ...,
        ge=1,
        description="Genomic position (1-based)"
    )

    reference_allele: str = Field(
        ...,
        min_length=1,
        description="Reference allele sequence"
    )

    alternate_allele: str = Field(
        ...,
        min_length=1,
        description="Alternate allele sequence"
    )

    variant_type: VariantType = Field(
        default=VariantType.SNP,
        description="Type of variant"
    )

    source: VariantSource = Field(
        default=VariantSource.DBSNP,
        description="Data source for this variant"
    )

    global_maf: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Global minor allele frequency"
    )

    population_frequencies: List[PopulationFrequency] = Field(
        default_factory=list,
        description="Frequency data per population"
    )

    clinical_significance: Optional[ClinicalSignificance] = Field(
        default=None,
        description="Clinical significance from ClinVar"
    )

    gene_symbol: Optional[str] = Field(
        default=None,
        description="Associated gene symbol"
    )

    transcript_id: Optional[str] = Field(
        default=None,
        description="Affected transcript ID"
    )

    consequence: Optional[str] = Field(
        default=None,
        description="Variant consequence (e.g., missense, synonymous)"
    )

    hgvs_c: Optional[str] = Field(
        default=None,
        description="HGVS coding notation"
    )

    hgvs_p: Optional[str] = Field(
        default=None,
        description="HGVS protein notation"
    )

    @field_validator('rsid')
    @classmethod
    def validate_rsid(cls, v: Optional[str]) -> Optional[str]:
        """Validate rsID format."""
        if v is None:
            return v
        if not v.startswith('rs'):
            raise ValueError(f"Invalid rsID format: {v}. Must start with 'rs'")
        try:
            int(v[2:])
        except ValueError:
            raise ValueError(f"Invalid rsID format: {v}. Must be 'rs' followed by numbers")
        return v

    @field_validator('chromosome')
    @classmethod
    def normalize_chromosome(cls, v: str) -> str:
        """Normalize chromosome notation."""
        v = v.upper().replace('CHR', '')
        if v in [str(i) for i in range(1, 23)] + ['X', 'Y', 'M', 'MT']:
            return f"chr{v.replace('MT', 'M')}"
        raise ValueError(f"Invalid chromosome: {v}")


class VariantCheckRequest(BaseModel):
    """Request for variant checking in a genomic region."""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier"
    )

    chromosome: str = Field(
        ...,
        description="Chromosome of the region"
    )

    start_position: int = Field(
        ...,
        ge=1,
        description="Start position (1-based, inclusive)"
    )

    end_position: int = Field(
        ...,
        ge=1,
        description="End position (1-based, inclusive)"
    )

    reference_genome: str = Field(
        default="GRCh38",
        description="Reference genome build (GRCh37, GRCh38)"
    )

    maf_threshold: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Minimum minor allele frequency to report"
    )

    sources: List[VariantSource] = Field(
        default=[VariantSource.DBSNP, VariantSource.GNOMAD],
        description="Variant sources to query"
    )

    include_clinical: bool = Field(
        default=True,
        description="Include ClinVar clinical annotations"
    )

    populations: Optional[List[str]] = Field(
        default=None,
        description="Specific populations to check (None = all)"
    )


class PrimerVariantConflict(BaseModel):
    """
    Variant conflict detected in primer binding site.

    This represents a potentially problematic variant that could
    affect primer binding efficiency or specificity.
    """

    primer_type: str = Field(
        ...,
        description="Primer type: forward, reverse, or probe"
    )

    variant: Variant = Field(
        ...,
        description="The conflicting variant"
    )

    position_in_primer: int = Field(
        ...,
        description="Position of variant within primer sequence (1-based)"
    )

    distance_from_3_prime: int = Field(
        ...,
        description="Distance from 3' end of primer (critical for binding)"
    )

    impact_severity: str = Field(
        default="medium",
        description="Estimated impact: low, medium, high, critical"
    )

    affects_binding: bool = Field(
        default=True,
        description="Whether variant affects primer binding"
    )

    mismatch_type: str = Field(
        default="terminal",
        description="Type of mismatch: terminal, internal, wobble"
    )

    predicted_tm_shift: Optional[float] = Field(
        default=None,
        description="Predicted Tm change due to mismatch (°C)"
    )

    recommendation: str = Field(
        default="",
        description="Suggested action for this conflict"
    )

    def calculate_severity(self) -> str:
        """Calculate impact severity based on position and frequency."""
        # 3' end variants are critical
        if self.distance_from_3_prime <= 3:
            if self.variant.global_maf and self.variant.global_maf > 0.05:
                return "critical"
            return "high"

        # High frequency variants anywhere are concerning
        if self.variant.global_maf and self.variant.global_maf > 0.1:
            return "high"

        # Internal variants with lower frequency
        if self.distance_from_3_prime <= 6:
            return "medium"

        return "low"


class VariantCheckResponse(BaseModel):
    """Response from variant checking service."""

    request_id: str = Field(
        ...,
        description="Original request ID"
    )

    status: str = Field(
        default="success",
        description="Response status: success, partial, failed"
    )

    region: Dict[str, Any] = Field(
        ...,
        description="Queried region details"
    )

    total_variants: int = Field(
        default=0,
        description="Total variants found in region"
    )

    high_frequency_variants: int = Field(
        default=0,
        description="Variants with MAF > threshold"
    )

    variants: List[Variant] = Field(
        default_factory=list,
        description="List of variants found"
    )

    has_clinical_variants: bool = Field(
        default=False,
        description="Whether clinically significant variants exist"
    )

    primer_conflicts: List[PrimerVariantConflict] = Field(
        default_factory=list,
        description="Conflicts with primer binding sites"
    )

    risk_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall variant risk score for primer design"
    )

    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for handling variants"
    )

    queried_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Query timestamp"
    )

    sources_queried: List[VariantSource] = Field(
        default_factory=list,
        description="Successfully queried sources"
    )


class HighRiskVariantException(Exception):
    """
    Exception raised when high-risk variants are detected in primer binding sites.

    This exception should trigger a design retry with adjusted parameters
    or user notification for manual review.
    """

    def __init__(
        self,
        message: str,
        conflicts: List[PrimerVariantConflict],
        risk_score: float,
        recommendations: List[str]
    ):
        super().__init__(message)
        self.conflicts = conflicts
        self.risk_score = risk_score
        self.recommendations = recommendations

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        return {
            "error_type": "HighRiskVariantException",
            "message": str(self),
            "risk_score": self.risk_score,
            "num_conflicts": len(self.conflicts),
            "conflicts": [
                {
                    "primer_type": c.primer_type,
                    "variant_id": c.variant.rsid or c.variant.variant_id,
                    "position": c.variant.position,
                    "maf": c.variant.global_maf,
                    "impact": c.impact_severity
                }
                for c in self.conflicts
            ],
            "recommendations": self.recommendations
        }


class VariantDatabase(BaseModel):
    """Configuration for variant database connection."""

    source: VariantSource = Field(
        ...,
        description="Database source type"
    )

    version: str = Field(
        ...,
        description="Database version (e.g., dbSNP b156)"
    )

    last_updated: datetime = Field(
        ...,
        description="Last update timestamp"
    )

    connection_string: Optional[str] = Field(
        default=None,
        description="Database connection string"
    )

    api_endpoint: Optional[str] = Field(
        default=None,
        description="REST API endpoint"
    )

    api_key: Optional[str] = Field(
        default=None,
        description="API key (if required)"
    )

    is_local: bool = Field(
        default=False,
        description="Whether database is local or remote"
    )


class AlleleDropoutRisk(BaseModel):
    """
    Allele dropout (ADO) risk assessment.

    Allele dropout occurs when one allele fails to amplify due to
    primer binding site variants, leading to false homozygous calls.
    """

    primer_pair_id: int = Field(
        ...,
        description="Associated primer pair ID"
    )

    forward_ado_risk: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="ADO risk from forward primer variants"
    )

    reverse_ado_risk: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="ADO risk from reverse primer variants"
    )

    combined_ado_risk: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Combined ADO risk for the primer pair"
    )

    affected_populations: List[str] = Field(
        default_factory=list,
        description="Populations with elevated ADO risk"
    )

    variants_contributing: List[str] = Field(
        default_factory=list,
        description="Variant rsIDs contributing to ADO risk"
    )

    recommendation: str = Field(
        default="",
        description="Recommendation for mitigating ADO risk"
    )


class MutationHotspot(BaseModel):
    """
    Mutation hotspot region definition.

    Used to flag regions with high mutation rates that may be
    unsuitable for primer binding.
    """

    chromosome: str = Field(
        ...,
        description="Chromosome"
    )

    start: int = Field(
        ...,
        description="Start position"
    )

    end: int = Field(
        ...,
        description="End position"
    )

    mutation_rate: float = Field(
        ...,
        description="Relative mutation rate (compared to genome average)"
    )

    gene: Optional[str] = Field(
        default=None,
        description="Associated gene"
    )

    hotspot_type: str = Field(
        default="general",
        description="Type: general, cpg, microsatellite, oncogene"
    )

    avoid_for_primer: bool = Field(
        default=True,
        description="Whether to avoid this region for primer design"
    )
