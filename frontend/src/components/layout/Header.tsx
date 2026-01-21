import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Dna,
  Menu,
  X,
  Github,
  BookOpen,
  Settings,
  ChevronDown,
  ExternalLink
} from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { Button, Badge, Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from '@/components/ui';
import { getHealth } from '@/api/client';

export function Header() {
  const [isScrolled, setIsScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const { data: health, isLoading: healthLoading } = useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
    refetchInterval: 30000,
    retry: 1,
  });

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const isHealthy = health?.status === 'healthy';

  return (
    <TooltipProvider>
      <motion.header
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        className={`
          fixed top-0 left-0 right-0 z-50 transition-all duration-300
          ${isScrolled ? 'glass shadow-lg' : 'bg-transparent'}
        `}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <motion.div
              className="flex items-center gap-3"
              whileHover={{ scale: 1.02 }}
            >
              <div className="relative">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-accent flex items-center justify-center">
                  <Dna className="w-6 h-6 text-background" />
                </div>
                <motion.div
                  className="absolute inset-0 rounded-xl bg-gradient-to-br from-primary to-accent opacity-50 blur-lg"
                  animate={{
                    scale: [1, 1.2, 1],
                    opacity: [0.3, 0.5, 0.3],
                  }}
                  transition={{
                    duration: 3,
                    repeat: Infinity,
                    ease: 'easeInOut',
                  }}
                />
              </div>
              <div>
                <h1 className="text-xl font-bold tracking-tight">
                  <span className="gradient-text">Bio-AI</span>
                  <span className="text-foreground/80"> Primer</span>
                </h1>
                <p className="text-[10px] text-muted-foreground -mt-0.5">
                  Enterprise Primer Design
                </p>
              </div>
            </motion.div>

            {/* Desktop Navigation */}
            <nav className="hidden md:flex items-center gap-6">
              <NavLink href="#" active>Designer</NavLink>
              <NavLink href="#">Batch</NavLink>
              <NavLink href="#">History</NavLink>

              <DropdownMenu
                trigger={
                  <button className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors">
                    Resources
                    <ChevronDown className="w-4 h-4" />
                  </button>
                }
              >
                <DropdownItem icon={<BookOpen className="w-4 h-4" />} href="#">
                  Documentation
                </DropdownItem>
                <DropdownItem icon={<Github className="w-4 h-4" />} href="#">
                  GitHub
                </DropdownItem>
                <DropdownItem icon={<ExternalLink className="w-4 h-4" />} href="#">
                  API Reference
                </DropdownItem>
              </DropdownMenu>
            </nav>

            {/* Right Section */}
            <div className="flex items-center gap-4">
              {/* Health Status */}
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-muted/50">
                    <motion.div
                      className={`w-2 h-2 rounded-full ${
                        healthLoading ? 'bg-yellow-400' :
                        isHealthy ? 'bg-green-400' : 'bg-red-400'
                      }`}
                      animate={{
                        scale: [1, 1.2, 1],
                        opacity: [1, 0.7, 1],
                      }}
                      transition={{
                        duration: 2,
                        repeat: Infinity,
                      }}
                    />
                    <span className="text-xs text-muted-foreground">
                      {healthLoading ? 'Connecting...' : isHealthy ? 'Online' : 'Offline'}
                    </span>
                  </div>
                </TooltipTrigger>
                <TooltipContent>
                  <div className="space-y-1">
                    <p className="font-medium">Service Status</p>
                    {health?.services && Object.entries(health.services).map(([key, value]) => (
                      <div key={key} className="flex items-center justify-between gap-4 text-xs">
                        <span className="text-muted-foreground capitalize">{key}</span>
                        <Badge variant={value === 'available' ? 'success' : 'error'} className="text-[10px]">
                          {value}
                        </Badge>
                      </div>
                    ))}
                  </div>
                </TooltipContent>
              </Tooltip>

              {/* Settings Button */}
              <Button variant="ghost" size="icon" className="hidden sm:flex">
                <Settings className="w-5 h-5" />
              </Button>

              {/* Mobile Menu Toggle */}
              <Button
                variant="ghost"
                size="icon"
                className="md:hidden"
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              >
                {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </Button>
            </div>
          </div>
        </div>

        {/* Mobile Menu */}
        <AnimatePresence>
          {mobileMenuOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="md:hidden glass border-t border-border/50"
            >
              <nav className="flex flex-col p-4 space-y-2">
                <MobileNavLink href="#" active>Designer</MobileNavLink>
                <MobileNavLink href="#">Batch</MobileNavLink>
                <MobileNavLink href="#">History</MobileNavLink>
                <MobileNavLink href="#">Documentation</MobileNavLink>
                <MobileNavLink href="#">API Reference</MobileNavLink>
              </nav>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.header>
    </TooltipProvider>
  );
}

// Navigation Link Component
function NavLink({
  href,
  children,
  active = false,
}: {
  href: string;
  children: React.ReactNode;
  active?: boolean;
}) {
  return (
    <a
      href={href}
      className={`
        relative text-sm font-medium transition-colors
        ${active ? 'text-primary' : 'text-muted-foreground hover:text-foreground'}
      `}
    >
      {children}
      {active && (
        <motion.div
          layoutId="activeNav"
          className="absolute -bottom-1 left-0 right-0 h-0.5 bg-primary rounded-full"
        />
      )}
    </a>
  );
}

// Mobile Navigation Link
function MobileNavLink({
  href,
  children,
  active = false,
}: {
  href: string;
  children: React.ReactNode;
  active?: boolean;
}) {
  return (
    <a
      href={href}
      className={`
        px-4 py-2 rounded-lg text-sm font-medium transition-colors
        ${active
          ? 'bg-primary/10 text-primary'
          : 'text-muted-foreground hover:bg-muted hover:text-foreground'
        }
      `}
    >
      {children}
    </a>
  );
}

// Dropdown Menu Component
function DropdownMenu({
  trigger,
  children,
}: {
  trigger: React.ReactNode;
  children: React.ReactNode;
}) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div
      className="relative"
      onMouseEnter={() => setIsOpen(true)}
      onMouseLeave={() => setIsOpen(false)}
    >
      {trigger}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className="absolute top-full right-0 mt-2 w-48 glass rounded-lg shadow-xl border border-border/50 overflow-hidden"
          >
            <div className="py-1">
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// Dropdown Item Component
function DropdownItem({
  href,
  icon,
  children,
}: {
  href: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <a
      href={href}
      className="flex items-center gap-3 px-4 py-2 text-sm text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors"
    >
      {icon}
      {children}
    </a>
  );
}

// Footer Component
export function Footer() {
  return (
    <footer className="mt-auto py-8 border-t border-border/50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Dna className="w-5 h-5 text-primary" />
            <span className="text-sm text-muted-foreground">
              Bio-AI Primer Design v1.0.0
            </span>
          </div>

          <div className="flex items-center gap-6 text-sm text-muted-foreground">
            <a href="#" className="hover:text-foreground transition-colors">Documentation</a>
            <a href="#" className="hover:text-foreground transition-colors">API</a>
            <a href="#" className="hover:text-foreground transition-colors">Support</a>
          </div>

          <div className="text-xs text-muted-foreground">
            FDA 21 CFR Part 11 Compliant
          </div>
        </div>
      </div>
    </footer>
  );
}
