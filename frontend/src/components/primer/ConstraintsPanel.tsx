import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Settings,
  Thermometer,
  Percent,
  Ruler,
  Shield,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
  Slider,
  Switch,
  Input,
  Badge,
} from '@/components/ui';
import type { DesignConstraints, TaskType } from '@/types';

interface ConstraintsPanelProps {
  constraints: DesignConstraints;
  onConstraintsChange: (constraints: DesignConstraints) => void;
  checkSpecificity: boolean;
  onCheckSpecificityChange: (value: boolean) => void;
  checkVariants: boolean;
  onCheckVariantsChange: (value: boolean) => void;
  taskType: TaskType;
}

export function ConstraintsPanel({
  constraints,
  onConstraintsChange,
  checkSpecificity,
  onCheckSpecificityChange,
  checkVariants,
  onCheckVariantsChange,
  taskType,
}: ConstraintsPanelProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const updateConstraint = <K extends keyof DesignConstraints>(
    key: K,
    value: DesignConstraints[K]
  ) => {
    onConstraintsChange({ ...constraints, [key]: value });
  };

  return (
    <Card variant="glass" className="h-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-lg">
          <Settings className="w-5 h-5 text-primary" />
          Design Parameters
        </CardTitle>
        <CardDescription>
          Configure thermodynamic constraints
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Temperature Settings */}
        <div className="space-y-4">
          <div className="flex items-center gap-2 text-sm font-medium text-foreground/90">
            <Thermometer className="w-4 h-4 text-primary" />
            Melting Temperature (Tm)
          </div>

          <div className="space-y-4 pl-6">
            <Slider
              label="Tm Range"
              min={45}
              max={75}
              step={0.5}
              value={[constraints.tm_min]}
              onValueChange={([val]) => updateConstraint('tm_min', val)}
              formatValue={(v) => `${v}°C`}
            />
            <div className="flex items-center gap-4 text-xs text-muted-foreground">
              <span>Min: {constraints.tm_min}°C</span>
              <span>Max: {constraints.tm_max}°C</span>
            </div>
            <Slider
              min={45}
              max={75}
              step={0.5}
              value={[constraints.tm_max]}
              onValueChange={([val]) => updateConstraint('tm_max', val)}
              showValue={false}
            />
          </div>
        </div>

        {/* GC Content */}
        <div className="space-y-4">
          <div className="flex items-center gap-2 text-sm font-medium text-foreground/90">
            <Percent className="w-4 h-4 text-primary" />
            GC Content
          </div>

          <div className="space-y-4 pl-6">
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">
                {constraints.gc_min}% - {constraints.gc_max}%
              </span>
              <Badge
                variant={
                  constraints.gc_min >= 40 && constraints.gc_max <= 60
                    ? 'success'
                    : 'warning'
                }
                className="text-[10px]"
              >
                {constraints.gc_min >= 40 && constraints.gc_max <= 60 ? 'Optimal' : 'Non-optimal'}
              </Badge>
            </div>
            <div className="relative h-2 rounded-full bg-gradient-to-r from-yellow-500 via-green-500 to-yellow-500 overflow-hidden">
              <div
                className="absolute h-full bg-primary/80 rounded-full transition-all duration-200"
                style={{
                  left: `${constraints.gc_min}%`,
                  width: `${constraints.gc_max - constraints.gc_min}%`,
                }}
              />
            </div>
            <div className="flex gap-4">
              <div className="flex-1">
                <Input
                  type="number"
                  value={constraints.gc_min}
                  onChange={(e) => updateConstraint('gc_min', parseFloat(e.target.value))}
                  className="h-9 text-center"
                />
              </div>
              <div className="flex-1">
                <Input
                  type="number"
                  value={constraints.gc_max}
                  onChange={(e) => updateConstraint('gc_max', parseFloat(e.target.value))}
                  className="h-9 text-center"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Primer Length */}
        <div className="space-y-4">
          <div className="flex items-center gap-2 text-sm font-medium text-foreground/90">
            <Ruler className="w-4 h-4 text-primary" />
            Primer Length
          </div>

          <div className="space-y-2 pl-6">
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>{constraints.primer_length_min} - {constraints.primer_length_max} bp</span>
              <span>Optimal: {constraints.primer_length_optimal} bp</span>
            </div>
            <Slider
              min={15}
              max={35}
              step={1}
              value={[constraints.primer_length_optimal]}
              onValueChange={([val]) => updateConstraint('primer_length_optimal', val)}
              formatValue={(v) => `${v} bp`}
              showValue={false}
            />
          </div>
        </div>

        {/* Product Size */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm font-medium text-foreground/90">
              <Ruler className="w-4 h-4 text-primary rotate-90" />
              Product Size
            </div>
            <Badge variant="outline" className="text-[10px]">
              {taskType}
            </Badge>
          </div>

          <div className="space-y-2 pl-6">
            <div className="text-xs text-muted-foreground">
              {constraints.product_size_min} - {constraints.product_size_max} bp
            </div>
            <div className="flex gap-4">
              <div className="flex-1">
                <Input
                  type="number"
                  value={constraints.product_size_min}
                  onChange={(e) => updateConstraint('product_size_min', parseInt(e.target.value))}
                  className="h-9 text-center"
                />
              </div>
              <div className="flex-1">
                <Input
                  type="number"
                  value={constraints.product_size_max}
                  onChange={(e) => updateConstraint('product_size_max', parseInt(e.target.value))}
                  className="h-9 text-center"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Analysis Options */}
        <div className="space-y-4 pt-4 border-t border-border/50">
          <div className="flex items-center gap-2 text-sm font-medium text-foreground/90">
            <Shield className="w-4 h-4 text-primary" />
            Analysis Options
          </div>

          <div className="space-y-3 pl-6">
            <Switch
              label="Check Specificity"
              description="BLAST against reference genome"
              checked={checkSpecificity}
              onCheckedChange={onCheckSpecificityChange}
            />
            <Switch
              label="Check Variants"
              description="SNP/variant conflict detection"
              checked={checkVariants}
              onCheckedChange={onCheckVariantsChange}
            />
            <Switch
              label="GC Clamp"
              description="Require G/C at 3' end"
              checked={constraints.gc_clamp}
              onCheckedChange={(checked) => updateConstraint('gc_clamp', checked)}
            />
          </div>
        </div>

        {/* Advanced Settings Toggle */}
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center justify-between w-full py-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          <span>Advanced Settings</span>
          {showAdvanced ? (
            <ChevronUp className="w-4 h-4" />
          ) : (
            <ChevronDown className="w-4 h-4" />
          )}
        </button>

        {/* Advanced Settings */}
        <motion.div
          initial={false}
          animate={{
            height: showAdvanced ? 'auto' : 0,
            opacity: showAdvanced ? 1 : 0,
          }}
          className="overflow-hidden"
        >
          <div className="space-y-4 pt-2">
            <div className="grid grid-cols-2 gap-4">
              <Input
                label="Max Poly-X"
                type="number"
                value={constraints.max_poly_x}
                onChange={(e) => updateConstraint('max_poly_x', parseInt(e.target.value))}
                hint="Max consecutive repeats"
              />
              <Input
                label="Self Comp."
                type="number"
                value={constraints.max_self_complementarity}
                onChange={(e) => updateConstraint('max_self_complementarity', parseInt(e.target.value))}
                hint="Max self-complementarity"
              />
            </div>

            <div className="space-y-4">
              <p className="text-xs font-medium text-muted-foreground">Buffer Conditions</p>
              <div className="grid grid-cols-3 gap-3">
                <Input
                  label="Na+ (mM)"
                  type="number"
                  value={constraints.na_concentration}
                  onChange={(e) => updateConstraint('na_concentration', parseFloat(e.target.value))}
                  className="h-9 text-center text-xs"
                />
                <Input
                  label="Mg2+ (mM)"
                  type="number"
                  value={constraints.mg_concentration}
                  onChange={(e) => updateConstraint('mg_concentration', parseFloat(e.target.value))}
                  className="h-9 text-center text-xs"
                />
                <Input
                  label="dNTPs (mM)"
                  type="number"
                  value={constraints.dntp_concentration}
                  onChange={(e) => updateConstraint('dntp_concentration', parseFloat(e.target.value))}
                  className="h-9 text-center text-xs"
                />
              </div>
            </div>
          </div>
        </motion.div>
      </CardContent>
    </Card>
  );
}
