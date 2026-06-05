import { motion } from "framer-motion";
import { AlertTriangle, Smile, Meh, Activity } from "lucide-react";
import { cn } from "@/lib/utils";

export type StressLevel = "low" | "medium" | "high";

interface StressIndicatorProps {
  level: StressLevel;
  probability: number;
  className?: string;
}

const stressConfig = {
  low: {
    label: "Low Stress",
    icon: Smile,
    bgClass: "bg-stress-low/10",
    textClass: "text-stress-low",
    ringClass: "ring-stress-low",
    gradientClass: "gradient-success",
    description: "You're doing great! Keep up the positive habits.",
  },
  medium: {
    label: "Medium Stress",
    icon: Meh,
    bgClass: "bg-stress-medium/10",
    textClass: "text-stress-medium",
    ringClass: "ring-stress-medium",
    gradientClass: "gradient-warning",
    description: "Moderate stress detected. Consider taking a short break.",
  },
  high: {
    label: "High Stress",
    icon: AlertTriangle,
    bgClass: "bg-stress-high/10",
    textClass: "text-stress-high",
    ringClass: "ring-stress-high",
    gradientClass: "gradient-danger",
    description: "High stress levels detected. Please take care of yourself.",
  },
};

const StressIndicator = ({ level, probability, className }: StressIndicatorProps) => {
  const config = stressConfig[level];
  const Icon = config.icon;
  const percentage = Math.round(probability * 100);

  return (
    <motion.div
      initial={{ scale: 0.9, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className={cn("relative", className)}
    >
      <div className={cn(
        "relative flex flex-col items-center p-8 rounded-3xl",
        config.bgClass,
        "ring-2",
        config.ringClass
      )}>
        {/* Animated background glow */}
        <motion.div
          className={cn(
            "absolute inset-0 rounded-3xl opacity-20 blur-xl",
            config.gradientClass
          )}
          animate={{ scale: [1, 1.05, 1] }}
          transition={{ duration: 2, repeat: Infinity }}
        />

        {/* Icon */}
        <motion.div
          initial={{ rotate: -10, scale: 0 }}
          animate={{ rotate: 0, scale: 1 }}
          transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
          className={cn(
            "relative flex h-24 w-24 items-center justify-center rounded-full",
            config.gradientClass,
            "shadow-lg"
          )}
        >
          <Icon className="h-12 w-12 text-white" />
        </motion.div>

        {/* Level Label */}
        <motion.h3
          initial={{ y: 10, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.3 }}
          className={cn("mt-6 text-2xl font-bold", config.textClass)}
        >
          {config.label}
        </motion.h3>

        {/* Probability */}
        <motion.div
          initial={{ y: 10, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="mt-2 flex items-center gap-2"
        >
          <Activity className={cn("h-4 w-4", config.textClass)} />
          <span className="text-sm font-medium text-muted-foreground">
            Confidence: <span className={cn("font-bold", config.textClass)}>{percentage}%</span>
          </span>
        </motion.div>

        {/* Progress bar */}
        <motion.div
          initial={{ scaleX: 0 }}
          animate={{ scaleX: 1 }}
          transition={{ delay: 0.5, duration: 0.8 }}
          className="mt-4 h-2 w-full max-w-xs overflow-hidden rounded-full bg-muted"
          style={{ originX: 0 }}
        >
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${percentage}%` }}
            transition={{ delay: 0.6, duration: 0.8 }}
            className={cn("h-full rounded-full", config.gradientClass)}
          />
        </motion.div>

        {/* Description */}
        <motion.p
          initial={{ y: 10, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="mt-4 text-center text-sm text-muted-foreground"
        >
          {config.description}
        </motion.p>
      </div>
    </motion.div>
  );
};

export default StressIndicator;
