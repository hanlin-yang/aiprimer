import axios, { AxiosInstance, AxiosError } from 'axios';
import type {
  PrimerDesignRequest,
  PrimerDesignResponse,
  BatchPrimerDesignRequest,
  BatchPrimerDesignResponse,
  HealthResponse,
  GenomeDatabase,
  SequenceValidationResult,
  AuditEvent,
  SecurityEvent,
} from '@/types';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1';

// Create axios instance
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 120000, // 2 minutes for long-running primer design
});

// Error handler
export class ApiError extends Error {
  public errorCode?: string;
  public details?: Record<string, unknown>;
  public statusCode?: number;

  constructor(message: string, errorCode?: string, statusCode?: number, details?: Record<string, unknown>) {
    super(message);
    this.name = 'ApiError';
    this.errorCode = errorCode;
    this.statusCode = statusCode;
    this.details = details;
  }
}

// Response interceptor
apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError<{ error_code?: string; message?: string; details?: Record<string, unknown> }>) => {
    const message = error.response?.data?.message || error.message || 'An error occurred';
    const errorCode = error.response?.data?.error_code;
    const statusCode = error.response?.status;
    const details = error.response?.data?.details;
    throw new ApiError(message, errorCode, statusCode, details);
  }
);

// API Functions

// Health Check
export async function getHealth(): Promise<HealthResponse> {
  const response = await axios.get<HealthResponse>('/health');
  return response.data;
}

// Primer Design
export async function designPrimers(request: PrimerDesignRequest): Promise<PrimerDesignResponse> {
  const response = await apiClient.post<PrimerDesignResponse>('/primers/design', request);
  return response.data;
}

// Batch Primer Design
export async function batchDesignPrimers(request: BatchPrimerDesignRequest): Promise<BatchPrimerDesignResponse> {
  const response = await apiClient.post<BatchPrimerDesignResponse>('/primers/batch', request);
  return response.data;
}

// Get Batch Status
export async function getBatchStatus(batchId: string): Promise<BatchPrimerDesignResponse> {
  const response = await apiClient.get<BatchPrimerDesignResponse>(`/primers/batch/${batchId}`);
  return response.data;
}

// Generate Report
export async function generateReport(
  responseData: PrimerDesignResponse,
  format: 'html' | 'pdf' = 'html'
): Promise<string | Blob> {
  const response = await apiClient.post('/reports/generate', responseData, {
    params: { format },
    responseType: format === 'pdf' ? 'blob' : 'json',
  });

  if (format === 'pdf') {
    return response.data as Blob;
  }
  return (response.data as { html: string }).html;
}

// Download PDF Report
export async function downloadPdfReport(responseData: PrimerDesignResponse): Promise<void> {
  const blob = await generateReport(responseData, 'pdf') as Blob;
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `primer_report_${responseData.request_id.slice(0, 8)}.pdf`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
}

// Validate Sequence
export async function validateSequence(sequence: string): Promise<SequenceValidationResult> {
  const response = await apiClient.post<SequenceValidationResult>('/utils/validate-sequence', null, {
    params: { sequence },
  });
  return response.data;
}

// Get Available Databases
export async function getDatabases(): Promise<GenomeDatabase[]> {
  const response = await apiClient.get<{ databases: GenomeDatabase[] }>('/utils/databases');
  return response.data.databases;
}

// Get Audit Events
export async function getAuditEvents(userId?: string, limit = 100): Promise<{ total: number; events: AuditEvent[] }> {
  const response = await apiClient.get<{ total: number; events: AuditEvent[] }>('/audit/events', {
    params: { user_id: userId, limit },
  });
  return response.data;
}

// Get Security Events
export async function getSecurityEvents(
  blockedOnly = false,
  limit = 100
): Promise<{ total: number; events: SecurityEvent[] }> {
  const response = await apiClient.get<{ total: number; events: SecurityEvent[] }>('/security/events', {
    params: { blocked_only: blockedOnly, limit },
  });
  return response.data;
}

export default apiClient;
