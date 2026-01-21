import { useMemo } from 'react';
import { motion } from 'framer-motion';
import type { PrimerPair } from '@/types';

interface SequenceVisualizationProps {
  pair: PrimerPair;
}

export function SequenceVisualization({ pair }: SequenceVisualizationProps) {
  const visualization = useMemo(() => {
    const fwdSeq = pair.forward.sequence;
    const revSeq = pair.reverse.sequence;
    const productSize = pair.product_size;

    // Create a simplified visual representation
    const totalWidth = 100;
    const fwdWidth = (fwdSeq.length / productSize) * totalWidth;
    const revWidth = (revSeq.length / productSize) * totalWidth;
    const gapWidth = totalWidth - fwdWidth - revWidth;

    return { fwdWidth, revWidth, gapWidth, fwdSeq, revSeq };
  }, [pair]);

  return (
    <div className="p-4 rounded-lg bg-background/50 border border-border/30 overflow-hidden">
      <div className="text-xs text-muted-foreground mb-3">Amplicon Visualization</div>

      {/* Visual representation */}
      <div className="relative h-16">
        {/* Template strand (top) */}
        <div className="absolute top-0 left-0 right-0 h-6">
          <div className="flex items-center h-full">
            {/* 5' label */}
            <span className="text-[10px] text-muted-foreground mr-1">5'</span>

            {/* Forward primer binding site */}
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${visualization.fwdWidth}%` }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="h-4 bg-green-500/30 border-2 border-green-500 rounded-l-full flex items-center justify-center relative"
            >
              <span className="text-[8px] text-green-400 font-mono truncate px-1">
                FWD →
              </span>
            </motion.div>

            {/* Intervening sequence */}
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${visualization.gapWidth}%` }}
              transition={{ duration: 0.5, delay: 0.3 }}
              className="h-2 bg-gradient-to-r from-muted/50 via-muted to-muted/50 flex items-center"
            >
              <div className="w-full border-t-2 border-dashed border-muted-foreground/30" />
            </motion.div>

            {/* Reverse primer binding site */}
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${visualization.revWidth}%` }}
              transition={{ duration: 0.5, delay: 0.5 }}
              className="h-4 bg-blue-500/30 border-2 border-blue-500 rounded-r-full flex items-center justify-center relative"
            >
              <span className="text-[8px] text-blue-400 font-mono truncate px-1">
                ← REV
              </span>
            </motion.div>

            {/* 3' label */}
            <span className="text-[10px] text-muted-foreground ml-1">3'</span>
          </div>
        </div>

        {/* Double helix connection */}
        <div className="absolute top-6 left-0 right-0 h-4 flex items-center justify-center">
          <svg className="w-full h-full" viewBox="0 0 100 16" preserveAspectRatio="none">
            {[...Array(20)].map((_, i) => (
              <motion.line
                key={i}
                x1={5 + i * 5}
                y1="4"
                x2={5 + i * 5}
                y2="12"
                stroke="currentColor"
                strokeWidth="0.5"
                className="text-muted-foreground/20"
                initial={{ pathLength: 0 }}
                animate={{ pathLength: 1 }}
                transition={{ duration: 0.3, delay: i * 0.02 }}
              />
            ))}
          </svg>
        </div>

        {/* Complementary strand (bottom) */}
        <div className="absolute bottom-0 left-0 right-0 h-6">
          <div className="flex items-center h-full">
            <span className="text-[10px] text-muted-foreground mr-1">3'</span>

            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${visualization.fwdWidth}%` }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="h-4 bg-green-500/10 border border-green-500/50 border-dashed rounded-l-full"
            />

            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${visualization.gapWidth}%` }}
              transition={{ duration: 0.5, delay: 0.4 }}
              className="h-2 bg-gradient-to-r from-muted/30 via-muted/50 to-muted/30"
            />

            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${visualization.revWidth}%` }}
              transition={{ duration: 0.5, delay: 0.6 }}
              className="h-4 bg-blue-500/10 border border-blue-500/50 border-dashed rounded-r-full"
            />

            <span className="text-[10px] text-muted-foreground ml-1">5'</span>
          </div>
        </div>
      </div>

      {/* Product size indicator */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7 }}
        className="mt-4 flex items-center justify-center gap-2"
      >
        <div className="h-px flex-1 bg-border" />
        <span className="text-xs font-mono text-muted-foreground px-2">
          {pair.product_size} bp amplicon
        </span>
        <div className="h-px flex-1 bg-border" />
      </motion.div>

      {/* Legend */}
      <div className="mt-3 flex items-center justify-center gap-6 text-[10px] text-muted-foreground">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded bg-green-500/30 border border-green-500" />
          <span>Forward Primer</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded bg-blue-500/30 border border-blue-500" />
          <span>Reverse Primer</span>
        </div>
        {pair.probe && (
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded bg-purple-500/30 border border-purple-500" />
            <span>Probe</span>
          </div>
        )}
      </div>
    </div>
  );
}

// DNA Helix Animation Component (for loading states)
export function DNAHelixLoader({ className = '' }: { className?: string }) {
  return (
    <div className={`relative w-16 h-32 ${className}`}>
      {[...Array(8)].map((_, i) => (
        <motion.div
          key={i}
          className="absolute left-1/2 -translate-x-1/2"
          style={{ top: `${i * 12.5}%` }}
          animate={{
            x: ['-8px', '8px', '-8px'],
          }}
          transition={{
            duration: 1.5,
            repeat: Infinity,
            delay: i * 0.1,
            ease: 'easeInOut',
          }}
        >
          <div className="flex items-center gap-1">
            <div
              className={`w-2 h-2 rounded-full ${
                i % 4 === 0 ? 'bg-dna-adenine' :
                i % 4 === 1 ? 'bg-dna-thymine' :
                i % 4 === 2 ? 'bg-dna-guanine' : 'bg-dna-cytosine'
              }`}
            />
            <div className="w-3 h-0.5 bg-muted-foreground/30" />
            <div
              className={`w-2 h-2 rounded-full ${
                i % 4 === 0 ? 'bg-dna-thymine' :
                i % 4 === 1 ? 'bg-dna-adenine' :
                i % 4 === 2 ? 'bg-dna-cytosine' : 'bg-dna-guanine'
              }`}
            />
          </div>
        </motion.div>
      ))}
    </div>
  );
}

// Sequence Colorizer Component
export function ColorizedSequence({
  sequence,
  highlightStart,
  highlightEnd,
  className = '',
}: {
  sequence: string;
  highlightStart?: number;
  highlightEnd?: number;
  className?: string;
}) {
  return (
    <span className={`font-mono tracking-wider ${className}`}>
      {sequence.split('').map((base, i) => {
        const isHighlighted = highlightStart !== undefined &&
          highlightEnd !== undefined &&
          i >= highlightStart &&
          i < highlightEnd;

        return (
          <span
            key={i}
            className={`
              ${base === 'A' ? 'text-dna-adenine' : ''}
              ${base === 'T' ? 'text-dna-thymine' : ''}
              ${base === 'G' ? 'text-dna-guanine' : ''}
              ${base === 'C' ? 'text-dna-cytosine' : ''}
              ${isHighlighted ? 'bg-primary/20 rounded' : ''}
            `}
          >
            {base}
          </span>
        );
      })}
    </span>
  );
}
