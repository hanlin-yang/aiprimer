import React, { useState, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Dna,
  AlertCircle,
  CheckCircle,
  Database
} from 'lucide-react';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
  Input,
  Textarea,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Badge,
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui';
import { isValidDNASequence, calculateGC } from '@/lib/utils';

interface SequenceInputProps {
  targetGene: string;
  sequence: string;
  onTargetGeneChange: (value: string) => void;
  onSequenceChange: (value: string) => void;
  genomeDatabase: string;
  onGenomeDatabaseChange: (value: string) => void;
}

const GENOME_DATABASES = [
  { id: 'hg38', name: 'GRCh38 (hg38)', species: 'Human' },
  { id: 'hg19', name: 'GRCh37 (hg19)', species: 'Human' },
  { id: 'mm10', name: 'GRCm38 (mm10)', species: 'Mouse' },
  { id: 'mm39', name: 'GRCm39 (mm39)', species: 'Mouse' },
];

const EXAMPLE_GENES = ['GAPDH', 'BRCA1', 'TP53', 'ACE2', 'EGFR', 'KRAS'];

export function SequenceInput({
  targetGene,
  sequence,
  onTargetGeneChange,
  onSequenceChange,
  genomeDatabase,
  onGenomeDatabaseChange,
}: SequenceInputProps) {
  const [inputMode, setInputMode] = useState<'gene' | 'sequence'>('gene');

  const sequenceStats = useMemo(() => {
    if (!sequence) return null;
    const cleanSeq = sequence.replace(/\s/g, '').toUpperCase();
    const isValid = isValidDNASequence(cleanSeq);
    const gc = calculateGC(cleanSeq);
    return {
      length: cleanSeq.length,
      gc: gc.toFixed(1),
      isValid,
      aCount: (cleanSeq.match(/A/g) || []).length,
      tCount: (cleanSeq.match(/T/g) || []).length,
      gCount: (cleanSeq.match(/G/g) || []).length,
      cCount: (cleanSeq.match(/C/g) || []).length,
    };
  }, [sequence]);

  const handleSequenceChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value.toUpperCase().replace(/[^ATCGRYSWKMBDHVN\s]/gi, '');
    onSequenceChange(value);
  }, [onSequenceChange]);

  const handleExampleGene = useCallback((gene: string) => {
    onTargetGeneChange(gene);
  }, [onTargetGeneChange]);

  const renderNucleotideComposition = () => {
    if (!sequenceStats) return null;

    const nucleotides = [
      { base: 'A', count: sequenceStats.aCount, color: 'bg-dna-adenine', label: 'Adenine' },
      { base: 'T', count: sequenceStats.tCount, color: 'bg-dna-thymine', label: 'Thymine' },
      { base: 'G', count: sequenceStats.gCount, color: 'bg-dna-guanine', label: 'Guanine' },
      { base: 'C', count: sequenceStats.cCount, color: 'bg-dna-cytosine', label: 'Cytosine' },
    ];

    const total = sequenceStats.length || 1;

    return (
      <div className="flex gap-1 h-2 rounded-full overflow-hidden bg-muted/50">
        {nucleotides.map(({ base, count, color }) => (
          <Tooltip key={base}>
            <TooltipTrigger asChild>
              <div
                className={`${color} transition-all duration-300`}
                style={{ width: `${(count / total) * 100}%` }}
              />
            </TooltipTrigger>
            <TooltipContent>
              <p className="font-mono">{base}: {count} ({((count / total) * 100).toFixed(1)}%)</p>
            </TooltipContent>
          </Tooltip>
        ))}
      </div>
    );
  };

  return (
    <Card variant="glass" className="h-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Dna className="w-5 h-5 text-primary" />
              Target Sequence
            </CardTitle>
            <CardDescription>
              Enter a gene symbol or paste a DNA sequence
            </CardDescription>
          </div>
          <div className="flex items-center gap-1 p-1 rounded-lg bg-muted/50">
            <button
              onClick={() => setInputMode('gene')}
              className={`
                px-3 py-1.5 rounded-md text-sm font-medium transition-all
                ${inputMode === 'gene'
                  ? 'bg-background text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
                }
              `}
            >
              Gene Symbol
            </button>
            <button
              onClick={() => setInputMode('sequence')}
              className={`
                px-3 py-1.5 rounded-md text-sm font-medium transition-all
                ${inputMode === 'sequence'
                  ? 'bg-background text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
                }
              `}
            >
              DNA Sequence
            </button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        <AnimatePresence mode="wait">
          {inputMode === 'gene' ? (
            <motion.div
              key="gene"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              className="space-y-4"
            >
              <Input
                label="Target Gene"
                placeholder="e.g., BRCA1, GAPDH, TP53"
                value={targetGene}
                onChange={(e) => onTargetGeneChange(e.target.value.toUpperCase())}
                hint="Enter a gene symbol to fetch sequence from reference genome"
              />

              <div className="space-y-2">
                <p className="text-xs text-muted-foreground">Popular genes:</p>
                <div className="flex flex-wrap gap-2">
                  {EXAMPLE_GENES.map((gene) => (
                    <motion.button
                      key={gene}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => handleExampleGene(gene)}
                      className={`
                        px-3 py-1 rounded-full text-xs font-mono transition-all
                        ${targetGene === gene
                          ? 'bg-primary text-primary-foreground'
                          : 'bg-muted hover:bg-muted/80 text-muted-foreground hover:text-foreground'
                        }
                      `}
                    >
                      {gene}
                    </motion.button>
                  ))}
                </div>
              </div>

              <Select value={genomeDatabase} onValueChange={onGenomeDatabaseChange}>
                <SelectTrigger label="Reference Genome">
                  <SelectValue placeholder="Select reference genome" />
                </SelectTrigger>
                <SelectContent>
                  {GENOME_DATABASES.map((db) => (
                    <SelectItem key={db.id} value={db.id}>
                      <div className="flex items-center gap-2">
                        <Database className="w-4 h-4 text-muted-foreground" />
                        <span>{db.name}</span>
                        <Badge variant="outline" className="ml-2 text-[10px]">
                          {db.species}
                        </Badge>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </motion.div>
          ) : (
            <motion.div
              key="sequence"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-4"
            >
              <div className="relative">
                <Textarea
                  label="DNA Sequence"
                  placeholder="Paste your DNA sequence here (FASTA format supported)...&#10;&#10;Example:&#10;ATGCGATCGATCGATCGATCG..."
                  value={sequence}
                  onChange={handleSequenceChange}
                  className="min-h-[200px] font-mono text-sm tracking-wider"
                  hint="Only A, T, C, G, and IUPAC ambiguity codes are accepted"
                />

                {sequence && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="absolute top-8 right-3"
                  >
                    {sequenceStats?.isValid ? (
                      <Badge variant="success" className="gap-1">
                        <CheckCircle className="w-3 h-3" />
                        Valid
                      </Badge>
                    ) : (
                      <Badge variant="error" className="gap-1">
                        <AlertCircle className="w-3 h-3" />
                        Invalid
                      </Badge>
                    )}
                  </motion.div>
                )}
              </div>

              {/* Sequence Statistics */}
              <AnimatePresence>
                {sequenceStats && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="overflow-hidden"
                  >
                    <div className="p-4 rounded-lg bg-muted/30 border border-border/50 space-y-4">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-muted-foreground">Sequence Statistics</span>
                        <div className="flex items-center gap-4">
                          <span className="font-mono">
                            <span className="text-muted-foreground">Length:</span>{' '}
                            <span className="text-foreground font-semibold">{sequenceStats.length} bp</span>
                          </span>
                          <span className="font-mono">
                            <span className="text-muted-foreground">GC:</span>{' '}
                            <span className={`font-semibold ${
                              parseFloat(sequenceStats.gc) >= 40 && parseFloat(sequenceStats.gc) <= 60
                                ? 'text-green-400'
                                : 'text-yellow-400'
                            }`}>
                              {sequenceStats.gc}%
                            </span>
                          </span>
                        </div>
                      </div>

                      {renderNucleotideComposition()}

                      <div className="flex items-center gap-6 text-xs">
                        <div className="flex items-center gap-2">
                          <span className="w-3 h-3 rounded bg-dna-adenine" />
                          <span className="text-muted-foreground">A: {sequenceStats.aCount}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="w-3 h-3 rounded bg-dna-thymine" />
                          <span className="text-muted-foreground">T: {sequenceStats.tCount}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="w-3 h-3 rounded bg-dna-guanine" />
                          <span className="text-muted-foreground">G: {sequenceStats.gCount}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="w-3 h-3 rounded bg-dna-cytosine" />
                          <span className="text-muted-foreground">C: {sequenceStats.cCount}</span>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          )}
        </AnimatePresence>
      </CardContent>
    </Card>
  );
}
