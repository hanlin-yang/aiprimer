import * as React from 'react';
import { cn } from '@/lib/utils';

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'default' | 'success' | 'warning' | 'error' | 'info' | 'outline';
}

function Badge({ className, variant = 'default', ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold transition-colors',
        variant === 'default' && 'bg-primary/10 text-primary',
        variant === 'success' && 'bg-green-500/10 text-green-400',
        variant === 'warning' && 'bg-yellow-500/10 text-yellow-400',
        variant === 'error' && 'bg-red-500/10 text-red-400',
        variant === 'info' && 'bg-blue-500/10 text-blue-400',
        variant === 'outline' && 'border border-border text-muted-foreground',
        className
      )}
      {...props}
    />
  );
}

export { Badge };
