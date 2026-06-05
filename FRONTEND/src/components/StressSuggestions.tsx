import { motion } from "framer-motion";
import { 
  Heart, 
  Music, 
  Moon, 
  Waves, 
  Coffee, 
  TreePine,
  Sparkles,
  Trophy,
  Star,
  Zap
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { StressLevel } from "./StressIndicator";

interface StressSuggestionsProps {
  level: StressLevel;
}

const stressTips = {
  low: [
    { icon: Trophy, text: "Great job maintaining your calm! Keep up the good work." },
    { icon: Star, text: "Your stress management skills are excellent." },
    { icon: Zap, text: "You have high energy and focus. Channel it productively!" },
    { icon: Sparkles, text: "Continue your healthy habits for long-term wellbeing." },
  ],
  medium: [
    { icon: Coffee, text: "Take a 5-minute break and stretch your body." },
    { icon: Waves, text: "Practice deep breathing: 4 seconds in, 4 seconds out." },
    { icon: Music, text: "Listen to calming music to relax your mind." },
    { icon: TreePine, text: "Step outside for some fresh air if possible." },
  ],
  high: [
    { icon: Heart, text: "Stop and take 10 deep breaths right now." },
    { icon: Moon, text: "Consider taking a longer break or rest." },
    { icon: Waves, text: "Try the 4-7-8 breathing technique for immediate relief." },
    { icon: TreePine, text: "Go for a walk in nature to reset your mind." },
  ],
};

const motivationalMessages = {
  low: {
    title: "You're Doing Amazing! 🌟",
    message: "Your stress levels are well-managed. This is the result of your positive choices and healthy lifestyle. Keep nurturing your mental wellness!",
    gradient: "gradient-success",
  },
  medium: {
    title: "Time for Self-Care 💫",
    message: "You're experiencing some stress, which is completely normal. Taking small steps to manage it now will prevent it from building up. You've got this!",
    gradient: "gradient-warning",
  },
  high: {
    title: "Let's Take a Moment 🤗",
    message: "Your wellbeing matters most. High stress is a signal from your body asking for attention. Please prioritize rest and self-care right now.",
    gradient: "gradient-danger",
  },
};

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.2,
    },
  },
};

const item = {
  hidden: { x: -20, opacity: 0 },
  show: { x: 0, opacity: 1 },
};

const StressSuggestions = ({ level }: StressSuggestionsProps) => {
  const tips = stressTips[level];
  const motivation = motivationalMessages[level];

  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className="space-y-6"
    >
      {/* Motivational message */}
      <motion.div variants={item}>
        <Card className={cn(
          "overflow-hidden border-0 card-shadow-elevated",
          level === "low" && "bg-stress-low/5",
          level === "medium" && "bg-stress-medium/5",
          level === "high" && "bg-stress-high/5"
        )}>
          <div className={cn("h-1", motivation.gradient)} />
          <CardHeader className="pb-2">
            <CardTitle className="text-xl">{motivation.title}</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground">{motivation.message}</p>
          </CardContent>
        </Card>
      </motion.div>

      {/* Tips */}
      <motion.div variants={item}>
        <Card className="card-shadow">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-lg">
              <Sparkles className="h-5 w-5 text-primary" />
              {level === "low" ? "Keep It Up!" : "Suggestions for You"}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <motion.ul variants={container} className="space-y-3">
              {tips.map((tip, index) => (
                <motion.li
                  key={index}
                  variants={item}
                  className="flex items-start gap-3 rounded-lg p-3 transition-colors hover:bg-muted/50"
                >
                  <div className={cn(
                    "flex h-9 w-9 shrink-0 items-center justify-center rounded-full",
                    level === "low" && "bg-stress-low/10 text-stress-low",
                    level === "medium" && "bg-stress-medium/10 text-stress-medium",
                    level === "high" && "bg-stress-high/10 text-stress-high"
                  )}>
                    <tip.icon className="h-4 w-4" />
                  </div>
                  <span className="text-sm text-foreground">{tip.text}</span>
                </motion.li>
              ))}
            </motion.ul>
          </CardContent>
        </Card>
      </motion.div>
    </motion.div>
  );
};

export default StressSuggestions;
