// Task Types
export type TaskType = 'qPCR' | 'PCR' | 'LAMP' | 'NGS';

// Error Codes
export type ErrorCode =
  | 'ERR_DIMER_RISK'
  | 'ERR_HAIRPIN_RISK'
  | 'ERR_TM_MISMATCH'
  | 'ERR_TM_OUT_OF_RANGE'
  | 'ERR_GC_OUT_OF_RANGE'
  | 'ERR_SNP_CONFLICT'
  | 'ERR_VARIANT_HOTSPOT'
  | 'ERR_OFF_TARGET'
  | 'ERR_LOW_SPECIFICITY'
  | 'ERR_PSEUDOGENE_HIT'
  | 'ERR_INVALID_SEQUENCE'
  | 'ERR_SEQUENCE_TOO_SHORT'
  | 'ERR_SEQUENCE_TOO_LONG'
  | 'ERR_HIGH_GC_REGION'
  | 'ERR_LOW_COMPLEXITY'
  | 'ERR_BLAST_TIMEOUT'
  | 'ERR_PRIMER3_FAILURE'
  | 'ERR_NO_VALID_PRIMERS'
  | 'ERR_SECURITY_BLOCK'
  | 'ERR_RATE_LIMIT';

// Design Constraints
export interface DesignConstraints {
  tm_min: number;
  tm_max: number;
  tm_optimal?: number;
  gc_min: number;
  gc_max: number;
  gc_clamp: boolean;
  gc_clamp_length: number;
  primer_length_min: number;
  primer_length_max: number;
  primer_length_optimal: number;
  product_size_min: number;
  product_size_max: number;
  max_poly_x: number;
  max_self_complementarity: number;
  max_end_complementarity: number;
  na_concentration: number;
  mg_concentration: number;
  dntp_concentration: number;
}

// Default constraints
export const DEFAULT_CONSTRAINTS: DesignConstraints = {
  tm_min: 55.0,
  tm_max: 60.0,
  tm_optimal: 57.5,
  gc_min: 40.0,
  gc_max: 60.0,
  gc_clamp: true,
  gc_clamp_length: 2,
  primer_length_min: 18,
  primer_length_max: 25,
  primer_length_optimal: 20,
  product_size_min: 75,
  product_size_max: 200,
  max_poly_x: 4,
  max_self_complementarity: 8,
  max_end_complementarity: 3,
  na_concentration: 50.0,
  mg_concentration: 1.5,
  dntp_concentration: 0.2,
};

// Primer Design Request
export interface PrimerDesignRequest {
  request_id?: string;
  target_gene: string;
  sequence_template?: string;
  target_region_start?: number;
  target_region_end?: number;
  task_type: TaskType;
  constraints: DesignConstraints;
  num_primers_return: number;
  check_specificity: boolean;
  genome_database: string;
  check_variants: boolean;
  variant_maf_threshold: number;
  include_probe: boolean;
  user_id?: string;
  session_id?: string;
}

// Primer Candidate
export interface PrimerCandidate {
  sequence: string;
  length: number;
  start_position: number;
  end_position: number;
  tm: number;
  gc_percent: number;
  self_complementarity: number;
  end_complementarity: number;
  hairpin_tm?: number;
  delta_g?: number;
}

// Primer Pair
export interface PrimerPair {
  pair_id: number;
  forward: PrimerCandidate;
  reverse: PrimerCandidate;
  probe?: PrimerCandidate;
  product_size: number;
  tm_difference: number;
  pair_complementarity: number;
  end_complementarity: number;
  dimer_risk_score: number;
  dimer_delta_g?: number;
  requires_optimization: boolean;
  penalty_score: number;
}

// Specificity Result
export interface SpecificityResult {
  is_specific: boolean;
  num_perfect_matches: number;
  off_target_hits: OffTargetHit[];
  pseudogene_hits: string[];
  specificity_score: number;
}

export interface OffTargetHit {
  chromosome: string;
  position: number;
  gene?: string;
  mismatch_count: number;
}

// Structural Warning
export interface StructuralWarning {
  code: ErrorCode;
  severity: 'info' | 'warning' | 'error' | 'critical';
  message: string;
  affected_primer?: 'forward' | 'reverse' | 'probe' | 'both';
  suggestion?: string;
  details?: Record<string, unknown>;
}

// Variant Warning
export interface VariantWarning {
  rsid?: string;
  position: number;
  alleles: string;
  maf: number;
  primer_type: 'forward' | 'reverse' | 'probe';
  impact: 'low' | 'medium' | 'high' | 'critical';
}

// Primer Design Response
export interface PrimerDesignResponse {
  request_id: string;
  status: 'success' | 'partial' | 'failed';
  task_type: TaskType;
  target_gene: string;
  primer_pairs: PrimerPair[];
  best_pair?: PrimerPair;
  specificity_results?: SpecificityResult;
  structural_warnings: StructuralWarning[];
  variant_warnings: VariantWarning[];
  ai_suggestion?: string;
  optimization_history: OptimizationStep[];
  error_codes: ErrorCode[];
  error_message?: string;
  computation_time_ms?: number;
  primer3_version?: string;
  blast_database?: string;
  created_at: string;
  audit_hash?: string;
}

export interface OptimizationStep {
  iteration: number;
  action: string;
  result: string;
}

// Batch Design
export interface BatchPrimerDesignRequest {
  batch_id?: string;
  requests: PrimerDesignRequest[];
  priority: 'low' | 'normal' | 'high';
  callback_url?: string;
}

export interface BatchPrimerDesignResponse {
  batch_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  total_requests: number;
  completed_requests: number;
  failed_requests: number;
  results: PrimerDesignResponse[];
  created_at: string;
  completed_at?: string;
}

// Health Check
export interface HealthResponse {
  status: string;
  timestamp: string;
  version: string;
  services: Record<string, string>;
}

// Genome Database
export interface GenomeDatabase {
  id: string;
  name: string;
  species: string;
}

// Sequence Validation
export interface SequenceValidationResult {
  valid: boolean;
  error?: string;
  length: number;
  gc_content?: number;
}

// Audit Event
export interface AuditEvent {
  timestamp: string;
  operation: string;
  user_id: string;
  success: boolean;
  computation_time_ms: number;
}

// Security Event
export interface SecurityEvent {
  event_id: string;
  event_type: string;
  threat_level: string;
  timestamp: string;
  action_taken: string;
  blocked: boolean;
}

// Report Request
export interface ReportRequest {
  response_data: PrimerDesignResponse;
  format: 'html' | 'pdf';
}
