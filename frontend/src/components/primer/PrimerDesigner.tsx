import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Dna,
  FlaskConical,
  Zap,
  AlertTriangle,
  CheckCircle,
  Download,
  RefreshCw,
  Thermometer,
  Target
} from 'lucide-react';
import { useMutation } from '@tanstack/react-query';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
  Button,
  Tabs,
  TabsList,
  TabsTrigger,
  TabsContent,
  Badge,
  TooltipProvider,
} from '@/components/ui';
import { SequenceInput } from './SequenceInput';
import { ConstraintsPanel } from './ConstraintsPanel';
import { PrimerResults } from './PrimerResults';
import { designPrimers, downloadPdfReport } from '@/api/client';
import type {
  PrimerDesignRequest,
  PrimerDesignResponse,
  TaskType,
  DesignConstraints,
} from '@/types';
import { generateId } from '@/lib/utils';

const TASK_TYPES: { value: TaskType; label: string; description: string; icon: React.ReactNode }[] = [
  {
    value: 'qPCR',
    label: 'qPCR',
    description: 'Quantitative PCR (70-300bp)',
    icon: <Target className="w-4 h-4" />
  },
  {
    value: 'PCR',
    label: 'PCR',
    description: 'Standard PCR',
    icon: <FlaskConical className="w-4 h-4" />
  },
  {
    value: 'LAMP',
    label: 'LAMP',
    description: 'Isothermal amplification',
    icon: <Thermometer className="w-4 h-4" />
  },
  {
    value: 'NGS',
    label: 'NGS',
    description: 'Next-gen sequencing',
    icon: <Dna className="w-4 h-4" />
  },
];

const defaultConstraints: DesignConstraints = {
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

export function PrimerDesigner() {
  const [targetGene, setTargetGene] = useState('');
  const [sequence, setSequence] = useState('');
  const [taskType, setTaskType] = useState<TaskType>('qPCR');
  const [constraints, setConstraints] = useState<DesignConstraints>(defaultConstraints);
  const [checkSpecificity, setCheckSpecificity] = useState(true);
  const [checkVariants, setCheckVariants] = useState(true);
  const [genomeDatabase, setGenomeDatabase] = useState('hg38');
  const [result, setResult] = useState<PrimerDesignResponse | null>(null);
  const [activeTab, setActiveTab] = useState('design');

  const designMutation = useMutation({
    mutationFn: designPrimers,
    onSuccess: (data) => {
      setResult(data);
      setActiveTab('results');
    },
  });

  const handleSubmit = useCallback(async () => {
    const request: PrimerDesignRequest = {
      request_id: generateId(),
      target_gene: targetGene,
      sequence_template: sequence || undefined,
      task_type: taskType,
      constraints,
      num_primers_return: 5,
      check_specificity: checkSpecificity,
      genome_database: genomeDatabase,
      check_variants: checkVariants,
      variant_maf_threshold: 0.01,
      include_probe: taskType === 'qPCR',
    };

    designMutation.mutate(request);
  }, [targetGene, sequence, taskType, constraints, checkSpecificity, checkVariants, genomeDatabase, designMutation]);

  const handleDownloadReport = useCallback(async () => {
    if (result) {
      await downloadPdfReport(result);
    }
  }, [result]);

  const handleReset = useCallback(() => {
    setTargetGene('');
    setSequence('');
    setConstraints(defaultConstraints);
    setResult(null);
    setActiveTab('design');
  }, []);

  const isFormValid = targetGene.trim().length > 0 || sequence.trim().length > 0;

  return (
    <TooltipProvider>
      <div className="w-full max-w-7xl mx-auto">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <div className="flex items-center justify-between mb-6">
            <TabsList className="glass">
              <TabsTrigger value="design" className="gap-2">
                <FlaskConical className="w-4 h-4" />
                Design
              </TabsTrigger>
              <TabsTrigger value="results" className="gap-2" disabled={!result}>
                <CheckCircle className="w-4 h-4" />
                Results
                {result && (
                  <Badge variant="success" className="ml-1 text-[10px]">
                    {result.primer_pairs.length}
                  </Badge>
                )}
              </TabsTrigger>
            </TabsList>

            {result && (
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleReset}
                  className="gap-2"
                >
                  <RefreshCw className="w-4 h-4" />
                  New Design
                </Button>
                <Button
                  variant="glow"
                  size="sm"
                  onClick={handleDownloadReport}
                  className="gap-2"
                >
                  <Download className="w-4 h-4" />
                  Export PDF
                </Button>
              </div>
            )}
          </div>

          <TabsContent value="design" className="space-y-6">
            {/* Task Type Selection */}
            <Card variant="glass" className="overflow-hidden">
              <CardHeader className="pb-4">
                <CardTitle className="flex items-center gap-2 text-lg">
                  <Zap className="w-5 h-5 text-primary" />
                  Task Type
                </CardTitle>
                <CardDescription>
                  Select the primer design application
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {TASK_TYPES.map((type) => (
                    <motion.button
                      key={type.value}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={() => setTaskType(type.value)}
                      className={`
                        relative p-4 rounded-xl border-2 transition-all duration-200 text-left
                        ${taskType === type.value
                          ? 'border-primary bg-primary/10 shadow-lg shadow-primary/10'
                          : 'border-border hover:border-primary/50 bg-card/50'
                        }
                      `}
                    >
                      <div className="flex items-center gap-3 mb-2">
                        <div className={`
                          p-2 rounded-lg transition-colors
                          ${taskType === type.value ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'}
                        `}>
                          {type.icon}
                        </div>
                        <span className="font-semibold">{type.label}</span>
                      </div>
                      <p className="text-xs text-muted-foreground">{type.description}</p>
                      {taskType === type.value && (
                        <motion.div
                          layoutId="taskTypeIndicator"
                          className="absolute inset-0 rounded-xl border-2 border-primary"
                          initial={false}
                          transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
                        />
                      )}
                    </motion.button>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Main Design Form */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Sequence Input - Takes 2 columns */}
              <div className="lg:col-span-2">
                <SequenceInput
                  targetGene={targetGene}
                  sequence={sequence}
                  onTargetGeneChange={setTargetGene}
                  onSequenceChange={setSequence}
                  genomeDatabase={genomeDatabase}
                  onGenomeDatabaseChange={setGenomeDatabase}
                />
              </div>

              {/* Constraints Panel */}
              <div className="lg:col-span-1">
                <ConstraintsPanel
                  constraints={constraints}
                  onConstraintsChange={setConstraints}
                  checkSpecificity={checkSpecificity}
                  onCheckSpecificityChange={setCheckSpecificity}
                  checkVariants={checkVariants}
                  onCheckVariantsChange={setCheckVariants}
                  taskType={taskType}
                />
              </div>
            </div>

            {/* Submit Button */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="flex justify-center pt-4"
            >
              <Button
                size="xl"
                onClick={handleSubmit}
                disabled={!isFormValid || designMutation.isPending}
                loading={designMutation.isPending}
                className="min-w-[240px] gap-3"
              >
                {designMutation.isPending ? (
                  'Designing Primers...'
                ) : (
                  <>
                    <Dna className="w-5 h-5" />
                    Design Primers
                  </>
                )}
              </Button>
            </motion.div>

            {/* Error Display */}
            <AnimatePresence>
              {designMutation.isError && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="overflow-hidden"
                >
                  <Card className="border-destructive/50 bg-destructive/5">
                    <CardContent className="p-4 flex items-start gap-3">
                      <AlertTriangle className="w-5 h-5 text-destructive shrink-0 mt-0.5" />
                      <div>
                        <p className="font-medium text-destructive">Design Failed</p>
                        <p className="text-sm text-muted-foreground mt-1">
                          {designMutation.error?.message || 'An unexpected error occurred'}
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              )}
            </AnimatePresence>
          </TabsContent>

          <TabsContent value="results">
            <AnimatePresence mode="wait">
              {result && (
                <motion.div
                  key="results"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                >
                  <PrimerResults result={result} />
                </motion.div>
              )}
            </AnimatePresence>
          </TabsContent>
        </Tabs>
      </div>
    </TooltipProvider>
  );
}
