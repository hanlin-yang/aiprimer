import React from 'react';
import { motion } from 'framer-motion';
import { Header, Footer } from '@/components/layout';
import { PrimerDesigner } from '@/components/primer';
import { Dna, Sparkles, Zap, Shield, Activity } from 'lucide-react';

function App() {
  return (
    <div className="min-h-screen flex flex-col helix-pattern">
      <Header />

      {/* Hero Section */}
      <section className="pt-24 pb-12 px-4 sm:px-6">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-12"
          >
            {/* Floating DNA Icon */}
            <motion.div
              className="inline-block mb-6"
              animate={{
                y: [0, -10, 0],
                rotateZ: [0, 5, -5, 0],
              }}
              transition={{
                duration: 6,
                repeat: Infinity,
                ease: 'easeInOut',
              }}
            >
              <div className="relative">
                <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-primary via-primary/80 to-accent flex items-center justify-center shadow-2xl shadow-primary/30">
                  <Dna className="w-10 h-10 text-background" />
                </div>
                <motion.div
                  className="absolute -inset-4 rounded-3xl bg-gradient-to-br from-primary/20 to-accent/20 blur-2xl -z-10"
                  animate={{
                    scale: [1, 1.1, 1],
                    opacity: [0.5, 0.8, 0.5],
                  }}
                  transition={{
                    duration: 3,
                    repeat: Infinity,
                  }}
                />
              </div>
            </motion.div>

            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-4">
              <span className="gradient-text">AI-Powered</span>
              <br />
              <span className="text-foreground">Primer Design</span>
            </h1>

            <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto mb-8">
              Enterprise-grade PCR primer design with intelligent optimization,
              specificity verification, and variant analysis.
            </p>

            {/* Feature Pills */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="flex flex-wrap items-center justify-center gap-3"
            >
              <FeaturePill icon={<Sparkles className="w-4 h-4" />} text="AI Optimization" />
              <FeaturePill icon={<Zap className="w-4 h-4" />} text="Primer3 Engine" />
              <FeaturePill icon={<Shield className="w-4 h-4" />} text="BLAST Specificity" />
              <FeaturePill icon={<Activity className="w-4 h-4" />} text="Variant Detection" />
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Main Content */}
      <main className="flex-1 px-4 sm:px-6 pb-12">
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.6 }}
          className="max-w-7xl mx-auto"
        >
          <PrimerDesigner />
        </motion.div>
      </main>

      <Footer />

      {/* Background Effects */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden -z-10">
        {/* Gradient Orbs */}
        <motion.div
          className="absolute top-1/4 -left-32 w-96 h-96 bg-primary/10 rounded-full blur-3xl"
          animate={{
            x: [0, 50, 0],
            y: [0, 30, 0],
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
        <motion.div
          className="absolute bottom-1/4 -right-32 w-96 h-96 bg-accent/10 rounded-full blur-3xl"
          animate={{
            x: [0, -50, 0],
            y: [0, -30, 0],
          }}
          transition={{
            duration: 25,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />

        {/* Grid Pattern */}
        <div
          className="absolute inset-0 opacity-[0.015]"
          style={{
            backgroundImage: `
              linear-gradient(to right, currentColor 1px, transparent 1px),
              linear-gradient(to bottom, currentColor 1px, transparent 1px)
            `,
            backgroundSize: '60px 60px',
          }}
        />
      </div>
    </div>
  );
}

// Feature Pill Component
function FeaturePill({ icon, text }: { icon: React.ReactNode; text: string }) {
  return (
    <motion.div
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      className="flex items-center gap-2 px-4 py-2 rounded-full bg-muted/50 border border-border/50 text-sm"
    >
      <span className="text-primary">{icon}</span>
      <span className="text-muted-foreground">{text}</span>
    </motion.div>
  );
}

export default App;
