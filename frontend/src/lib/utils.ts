import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Calculate GC content percentage
export function calculateGC(sequence: string): number {
  if (!sequence) return 0;
  const upper = sequence.toUpperCase();
  const gc = (upper.match(/[GC]/g) || []).length;
  return (gc / sequence.length) * 100;
}

// Format temperature with unit
export function formatTemperature(temp: number, decimals = 1): string {
  return `${temp.toFixed(decimals)}°C`;
}

// Format percentage
export function formatPercent(value: number, decimals = 1): string {
  return `${value.toFixed(decimals)}%`;
}

// Format delta G
export function formatDeltaG(value: number | undefined): string {
  if (value === undefined) return 'N/A';
  return `${value.toFixed(2)} kcal/mol`;
}

// Get severity color
export function getSeverityColor(severity: string): string {
  switch (severity) {
    case 'critical':
      return 'text-red-600 bg-red-50 border-red-200';
    case 'error':
    case 'high':
      return 'text-orange-600 bg-orange-50 border-orange-200';
    case 'warning':
    case 'medium':
      return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    case 'info':
    case 'low':
      return 'text-blue-600 bg-blue-50 border-blue-200';
    default:
      return 'text-gray-600 bg-gray-50 border-gray-200';
  }
}

// Get nucleotide color for visualization
export function getNucleotideColor(base: string): string {
  switch (base.toUpperCase()) {
    case 'A':
      return '#22c55e'; // Green
    case 'T':
      return '#ef4444'; // Red
    case 'G':
      return '#3b82f6'; // Blue
    case 'C':
      return '#eab308'; // Yellow
    default:
      return '#9ca3af'; // Gray
  }
}

// Format sequence with spacing
export function formatSequence(sequence: string, groupSize = 10): string {
  const groups = [];
  for (let i = 0; i < sequence.length; i += groupSize) {
    groups.push(sequence.slice(i, i + groupSize));
  }
  return groups.join(' ');
}

// Validate DNA sequence
export function isValidDNASequence(sequence: string): boolean {
  const validChars = /^[ATCGRYSWKMBDHVNatcgryswkmbdhvn\s]*$/;
  return validChars.test(sequence);
}

// Calculate Tm score (0-100, higher is better around optimal)
export function calculateTmScore(tm: number, optimal = 57.5, range = 5): number {
  const diff = Math.abs(tm - optimal);
  if (diff <= range) {
    return 100 - (diff / range) * 50;
  }
  return Math.max(0, 50 - (diff - range) * 10);
}

// Calculate GC score (0-100, optimal is 50%)
export function calculateGCScore(gc: number): number {
  const optimal = 50;
  const diff = Math.abs(gc - optimal);
  if (diff <= 10) {
    return 100 - diff * 2;
  }
  return Math.max(0, 80 - (diff - 10) * 4);
}

// Calculate dimer risk score display
export function getDimerRiskLevel(score: number): { level: string; color: string } {
  if (score <= 0.2) return { level: 'Low', color: 'text-green-600' };
  if (score <= 0.5) return { level: 'Medium', color: 'text-yellow-600' };
  if (score <= 0.7) return { level: 'High', color: 'text-orange-600' };
  return { level: 'Critical', color: 'text-red-600' };
}

// Format time duration
export function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}min`;
}

// Generate unique ID
export function generateId(): string {
  return crypto.randomUUID();
}

// Debounce function
export function debounce<T extends (...args: Parameters<T>) => ReturnType<T>>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: ReturnType<typeof setTimeout> | null = null;
  return (...args: Parameters<T>) => {
    if (timeout) clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}

// Copy to clipboard
export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}
