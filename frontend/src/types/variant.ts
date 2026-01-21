// Variant Types
export type VariantSource = 'dbSNP' | 'ClinVar' | 'GISAID' | 'gnomAD' | 'COSMIC' | 'custom';
export type VariantType = 'SNP' | 'INDEL' | 'MNP' | 'CNV' | 'SV';
export type ClinicalSignificance =
  | 'benign'
  | 'likely_benign'
  | 'uncertain_significance'
  | 'likely_pathogenic'
  | 'pathogenic'
  | 'drug_response'
  | 'not_provided';

// Population Frequency
export interface PopulationFrequency {
  population: string;
  allele_frequency: number;
  allele_count?: number;
  total_alleles?: number;
}

// Variant
export interface Variant {
  variant_id: string;
  rsid?: string;
  chromosome: string;
  position: number;
  reference_allele: string;
  alternate_allele: string;
  variant_type: VariantType;
  source: VariantSource;
  global_maf?: number;
  population_frequencies: PopulationFrequency[];
  clinical_significance?: ClinicalSignificance;
  gene_symbol?: string;
  transcript_id?: string;
  consequence?: string;
  hgvs_c?: string;
  hgvs_p?: string;
}

// Variant Check Request
export interface VariantCheckRequest {
  request_id?: string;
  chromosome: string;
  start_position: number;
  end_position: number;
  reference_genome: string;
  maf_threshold: number;
  sources: VariantSource[];
  include_clinical: boolean;
  populations?: string[];
}

// Primer Variant Conflict
export interface PrimerVariantConflict {
  primer_type: 'forward' | 'reverse' | 'probe';
  variant: Variant;
  position_in_primer: number;
  distance_from_3_prime: number;
  impact_severity: 'low' | 'medium' | 'high' | 'critical';
  affects_binding: boolean;
  mismatch_type: 'terminal' | 'internal' | 'wobble';
  predicted_tm_shift?: number;
  recommendation: string;
}

// Variant Check Response
export interface VariantCheckResponse {
  request_id: string;
  status: 'success' | 'partial' | 'failed';
  region: {
    chromosome: string;
    start: number;
    end: number;
  };
  total_variants: number;
  high_frequency_variants: number;
  variants: Variant[];
  has_clinical_variants: boolean;
  primer_conflicts: PrimerVariantConflict[];
  risk_score: number;
  recommendations: string[];
  queried_at: string;
  sources_queried: VariantSource[];
}

// Allele Dropout Risk
export interface AlleleDropoutRisk {
  primer_pair_id: number;
  forward_ado_risk: number;
  reverse_ado_risk: number;
  combined_ado_risk: number;
  affected_populations: string[];
  variants_contributing: string[];
  recommendation: string;
}

// Mutation Hotspot
export interface MutationHotspot {
  chromosome: string;
  start: number;
  end: number;
  mutation_rate: number;
  gene?: string;
  hotspot_type: 'general' | 'cpg' | 'microsatellite' | 'oncogene';
  avoid_for_primer: boolean;
}
