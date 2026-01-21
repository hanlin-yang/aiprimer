import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Award,
  AlertTriangle,
  CheckCircle,
  ChevronDown,
  Copy,
  Thermometer,
  Percent,
  Zap,
  Target,
  AlertCircle,
  Sparkles,
  Clock,
  Shield
} from 'lucide-react';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Badge,
  Button,
} from '@/components/ui';
import { SequenceVisualization } from './SequenceVisualization';
import type {
  PrimerDesignResponse,
  PrimerPair,
  StructuralWarning,
  VariantWarning,
} from '@/types';
import {
  formatTemperature,
  formatPercent,
  formatDeltaG,
  formatDuration,
  getSeverityColor,
  getDimerRiskLevel,
  copyToClipboard,
} from '@/lib/utils';

interface PrimerResultsProps {
  result: PrimerDesignResponse;
}

export function PrimerResults({ result }: PrimerResultsProps) {
  const [expandedPair, setExpandedPair] = useState<number | null>(
    result.best_pair?.pair_id ?? result.primer_pairs[0]?.pair_id ?? null
  );
  const [copiedSequence, setCopiedSequence] = useState<string | null>(null);

  const handleCopy = async (sequence: string, id: string) => {
    const success = await copyToClipboard(sequence);
    if (success) {
      setCopiedSequence(id);
      setTimeout(() => setCopiedSequence(null), 2000);
    }
  };

  const statusConfig = {
    success: { icon: CheckCircle, color: 'text-green-400', bg: 'bg-green-400/10' },
    partial: { icon: AlertTriangle, color: 'text-yellow-400', bg: 'bg-yellow-400/10' },
    failed: { icon: AlertCircle, color: 'text-red-400', bg: 'bg-red-400/10' },
  };

  const StatusIcon = statusConfig[result.status].icon;

  return (
    <div className="space-y-6">
      {/* Summary Header */}
      <Card variant="gradient" className="overflow-hidden">
        <CardContent className="p-6">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className={`p-3 rounded-xl ${statusConfig[result.status].bg}`}>
                <StatusIcon className={`w-8 h-8 ${statusConfig[result.status].color}`} />
              </div>
              <div>
                <h2 className="text-2xl font-bold">
                  {result.target_gene}
                  <Badge variant="outline" className="ml-3 text-xs">
                    {result.task_type}
                  </Badge>
                </h2>
                <p className="text-muted-foreground mt-1">
                  {result.primer_pairs.length} primer pair{result.primer_pairs.length !== 1 ? 's' : ''} designed
                </p>
              </div>
            </div>

            <div className="flex items-center gap-6 text-sm">
              <div className="flex items-center gap-2">
                <Clock className="w-4 h-4 text-muted-foreground" />
                <span className="text-muted-foreground">
                  {formatDuration(result.computation_time_ms || 0)}
                </span>
              </div>
              {result.primer3_version && (
                <div className="flex items-center gap-2">
                  <Zap className="w-4 h-4 text-muted-foreground" />
                  <span className="text-muted-foreground text-xs">
                    Primer3 {result.primer3_version}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* AI Suggestion */}
          {result.ai_suggestion && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="mt-6 p-4 rounded-lg bg-primary/5 border border-primary/20"
            >
              <div className="flex items-start gap-3">
                <Sparkles className="w-5 h-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-primary mb-1">AI Recommendation</p>
                  <p className="text-sm text-muted-foreground">{result.ai_suggestion}</p>
                </div>
              </div>
            </motion.div>
          )}
        </CardContent>
      </Card>

      {/* Warnings Section */}
      {(result.structural_warnings.length > 0 || result.variant_warnings.length > 0) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {result.structural_warnings.length > 0 && (
            <WarningsCard
              title="Structural Warnings"
              icon={<AlertTriangle className="w-5 h-5" />}
              warnings={result.structural_warnings}
            />
          )}
          {result.variant_warnings.length > 0 && (
            <VariantWarningsCard warnings={result.variant_warnings} />
          )}
        </div>
      )}

      {/* Primer Pairs */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Target className="w-5 h-5 text-primary" />
          Primer Pairs
        </h3>

        <div className="space-y-3">
          {result.primer_pairs.map((pair, index) => (
            <PrimerPairCard
              key={pair.pair_id}
              pair={pair}
              index={index}
              isExpanded={expandedPair === pair.pair_id}
              isBest={result.best_pair?.pair_id === pair.pair_id}
              onToggle={() => setExpandedPair(
                expandedPair === pair.pair_id ? null : pair.pair_id
              )}
              onCopy={handleCopy}
              copiedSequence={copiedSequence}
            />
          ))}
        </div>
      </div>

      {/* Specificity Results */}
      {result.specificity_results && (
        <Card variant="glass">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Shield className="w-5 h-5 text-primary" />
              Specificity Analysis
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <MetricCard
                label="Specificity Score"
                value={`${(result.specificity_results.specificity_score * 100).toFixed(0)}%`}
                status={result.specificity_results.is_specific ? 'success' : 'warning'}
              />
              <MetricCard
                label="Perfect Matches"
                value={result.specificity_results.num_perfect_matches.toString()}
                status={result.specificity_results.num_perfect_matches === 1 ? 'success' : 'warning'}
              />
              <MetricCard
                label="Off-Target Hits"
                value={result.specificity_results.off_target_hits.length.toString()}
                status={result.specificity_results.off_target_hits.length === 0 ? 'success' : 'warning'}
              />
              <MetricCard
                label="Pseudogene Hits"
                value={result.specificity_results.pseudogene_hits.length.toString()}
                status={result.specificity_results.pseudogene_hits.length === 0 ? 'success' : 'warning'}
              />
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

// Primer Pair Card Component
interface PrimerPairCardProps {
  pair: PrimerPair;
  index: number;
  isExpanded: boolean;
  isBest: boolean;
  onToggle: () => void;
  onCopy: (sequence: string, id: string) => void;
  copiedSequence: string | null;
}

function PrimerPairCard({
  pair,
  index,
  isExpanded,
  isBest,
  onToggle,
  onCopy,
  copiedSequence,
}: PrimerPairCardProps) {
  const dimerRisk = getDimerRiskLevel(pair.dimer_risk_score);

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1 }}
    >
      <Card
        variant={isBest ? 'gradient' : 'glass'}
        className={`overflow-hidden transition-all duration-300 ${
          isBest ? 'ring-2 ring-primary/50' : ''
        }`}
      >
        {/* Header */}
        <button
          onClick={onToggle}
          className="w-full p-4 flex items-center justify-between hover:bg-muted/20 transition-colors"
        >
          <div className="flex items-center gap-4">
            <div className={`
              w-10 h-10 rounded-xl flex items-center justify-center font-bold
              ${isBest ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'}
            `}>
              {isBest ? <Award className="w-5 h-5" /> : `#${index + 1}`}
            </div>
            <div className="text-left">
              <div className="flex items-center gap-2">
                <span className="font-semibold">Pair {pair.pair_id}</span>
                {isBest && (
                  <Badge variant="success" className="text-[10px]">
                    Best Match
                  </Badge>
                )}
              </div>
              <div className="flex items-center gap-4 text-xs text-muted-foreground mt-1">
                <span>Product: {pair.product_size} bp</span>
                <span>ΔTm: {pair.tm_difference.toFixed(1)}°C</span>
                <span className={dimerRisk.color}>Dimer: {dimerRisk.level}</span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="text-right hidden md:block">
              <div className="text-xs text-muted-foreground">Penalty Score</div>
              <div className={`font-mono font-bold ${
                pair.penalty_score < 1 ? 'text-green-400' :
                pair.penalty_score < 3 ? 'text-yellow-400' : 'text-red-400'
              }`}>
                {pair.penalty_score.toFixed(2)}
              </div>
            </div>
            <motion.div
              animate={{ rotate: isExpanded ? 180 : 0 }}
              transition={{ duration: 0.2 }}
            >
              <ChevronDown className="w-5 h-5 text-muted-foreground" />
            </motion.div>
          </div>
        </button>

        {/* Expanded Content */}
        <AnimatePresence>
          {isExpanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.3 }}
              className="overflow-hidden"
            >
              <div className="px-4 pb-4 space-y-4 border-t border-border/50 pt-4">
                {/* Forward Primer */}
                <PrimerDetail
                  type="Forward"
                  primer={pair.forward}
                  onCopy={onCopy}
                  copiedSequence={copiedSequence}
                />

                {/* Reverse Primer */}
                <PrimerDetail
                  type="Reverse"
                  primer={pair.reverse}
                  onCopy={onCopy}
                  copiedSequence={copiedSequence}
                />

                {/* Probe (if present) */}
                {pair.probe && (
                  <PrimerDetail
                    type="Probe"
                    primer={pair.probe}
                    onCopy={onCopy}
                    copiedSequence={copiedSequence}
                  />
                )}

                {/* Sequence Visualization */}
                <SequenceVisualization pair={pair} />

                {/* Thermodynamic Details */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <div className="p-3 rounded-lg bg-muted/30">
                    <div className="text-xs text-muted-foreground">Pair Comp.</div>
                    <div className="font-mono font-semibold">{pair.pair_complementarity}</div>
                  </div>
                  <div className="p-3 rounded-lg bg-muted/30">
                    <div className="text-xs text-muted-foreground">End Comp.</div>
                    <div className="font-mono font-semibold">{pair.end_complementarity}</div>
                  </div>
                  <div className="p-3 rounded-lg bg-muted/30">
                    <div className="text-xs text-muted-foreground">Dimer ΔG</div>
                    <div className="font-mono font-semibold">{formatDeltaG(pair.dimer_delta_g)}</div>
                  </div>
                  <div className="p-3 rounded-lg bg-muted/30">
                    <div className="text-xs text-muted-foreground">Optimization</div>
                    <div className={`font-semibold ${pair.requires_optimization ? 'text-yellow-400' : 'text-green-400'}`}>
                      {pair.requires_optimization ? 'Recommended' : 'Not Required'}
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </Card>
    </motion.div>
  );
}

// Primer Detail Component
interface PrimerDetailProps {
  type: 'Forward' | 'Reverse' | 'Probe';
  primer: {
    sequence: string;
    length: number;
    tm: number;
    gc_percent: number;
    delta_g?: number;
  };
  onCopy: (sequence: string, id: string) => void;
  copiedSequence: string | null;
}

function PrimerDetail({ type, primer, onCopy, copiedSequence }: PrimerDetailProps) {
  const id = `${type}-${primer.sequence.slice(0, 8)}`;
  const isCopied = copiedSequence === id;

  const typeConfig = {
    Forward: { color: 'text-green-400', bg: 'bg-green-400/10', arrow: '→' },
    Reverse: { color: 'text-blue-400', bg: 'bg-blue-400/10', arrow: '←' },
    Probe: { color: 'text-purple-400', bg: 'bg-purple-400/10', arrow: '◆' },
  };

  const config = typeConfig[type];

  return (
    <div className="p-3 rounded-lg bg-muted/20 border border-border/50">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className={`text-lg ${config.color}`}>{config.arrow}</span>
          <span className="font-medium">{type} Primer</span>
          <Badge variant="outline" className="text-[10px]">
            {primer.length} bp
          </Badge>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => onCopy(primer.sequence, id)}
          className="gap-2 h-8"
        >
          {isCopied ? (
            <>
              <CheckCircle className="w-3 h-3 text-green-400" />
              Copied!
            </>
          ) : (
            <>
              <Copy className="w-3 h-3" />
              Copy
            </>
          )}
        </Button>
      </div>

      <div className="font-mono text-sm tracking-wider bg-background/50 p-2 rounded border border-border/30 overflow-x-auto">
        <span className="text-muted-foreground">5'-</span>
        {primer.sequence.split('').map((base, i) => (
          <span
            key={i}
            className={`
              ${base === 'A' ? 'text-dna-adenine' : ''}
              ${base === 'T' ? 'text-dna-thymine' : ''}
              ${base === 'G' ? 'text-dna-guanine' : ''}
              ${base === 'C' ? 'text-dna-cytosine' : ''}
            `}
          >
            {base}
          </span>
        ))}
        <span className="text-muted-foreground">-3'</span>
      </div>

      <div className="flex items-center gap-6 mt-2 text-xs text-muted-foreground">
        <span className="flex items-center gap-1">
          <Thermometer className="w-3 h-3" />
          Tm: {formatTemperature(primer.tm)}
        </span>
        <span className="flex items-center gap-1">
          <Percent className="w-3 h-3" />
          GC: {formatPercent(primer.gc_percent)}
        </span>
        {primer.delta_g !== undefined && (
          <span>ΔG: {formatDeltaG(primer.delta_g)}</span>
        )}
      </div>
    </div>
  );
}

// Warnings Card Component
function WarningsCard({
  title,
  icon,
  warnings,
}: {
  title: string;
  icon: React.ReactNode;
  warnings: StructuralWarning[];
}) {
  return (
    <Card variant="glass" className="border-yellow-500/30">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-base text-yellow-400">
          {icon}
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {warnings.map((warning, i) => (
            <div
              key={i}
              className={`p-2 rounded-lg border text-sm ${getSeverityColor(warning.severity)}`}
            >
              <div className="font-medium">{warning.message}</div>
              {warning.suggestion && (
                <div className="text-xs mt-1 opacity-80">{warning.suggestion}</div>
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// Variant Warnings Card
function VariantWarningsCard({ warnings }: { warnings: VariantWarning[] }) {
  return (
    <Card variant="glass" className="border-orange-500/30">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-base text-orange-400">
          <AlertCircle className="w-5 h-5" />
          Variant Warnings
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {warnings.map((warning, i) => (
            <div
              key={i}
              className={`p-2 rounded-lg border text-sm ${getSeverityColor(warning.impact)}`}
            >
              <div className="flex items-center justify-between">
                <span className="font-mono">{warning.rsid || `Pos ${warning.position}`}</span>
                <Badge variant="outline" className="text-[10px]">
                  {warning.primer_type}
                </Badge>
              </div>
              <div className="text-xs mt-1 opacity-80">
                {warning.alleles} (MAF: {(warning.maf * 100).toFixed(2)}%)
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// Metric Card Component
function MetricCard({
  label,
  value,
  status,
}: {
  label: string;
  value: string;
  status: 'success' | 'warning' | 'error';
}) {
  const statusColors = {
    success: 'text-green-400',
    warning: 'text-yellow-400',
    error: 'text-red-400',
  };

  return (
    <div className="p-4 rounded-lg bg-muted/30 text-center">
      <div className="text-xs text-muted-foreground mb-1">{label}</div>
      <div className={`text-xl font-bold ${statusColors[status]}`}>{value}</div>
    </div>
  );
}
