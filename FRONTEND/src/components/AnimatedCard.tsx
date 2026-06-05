import { motion } from "framer-motion";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { LucideIcon } from "lucide-react";

interface AnimatedCardProps {
  icon: LucideIcon;
  label: string;
  value: string | number;
  trend?: "up" | "down" | "neutral";
  delay?: number;
  className?: string;
  iconClassName?: string;
}

const AnimatedCard = ({
  icon: Icon,
  label,
  value,
  trend,
  delay = 0,
  className,
  iconClassName,
}: AnimatedCardProps) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.4 }}
      whileHover={{ y: -4, transition: { duration: 0.2 } }}
    >
      <Card className={cn("card-shadow transition-shadow hover:card-shadow-elevated", className)}>
        <CardContent className="flex items-center gap-4 p-5">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: delay + 0.1, type: "spring", stiffness: 200 }}
            className={cn(
              "flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10",
              iconClassName
            )}
          >
            <Icon className="h-6 w-6 text-primary" />
          </motion.div>
          <div className="flex-1">
            <p className="text-sm font-medium text-muted-foreground">{label}</p>
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: delay + 0.2 }}
              className="text-2xl font-bold text-foreground"
            >
              {value}
            </motion.p>
          </div>
          {trend && (
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: delay + 0.3, type: "spring" }}
              className={cn(
                "rounded-full px-2 py-1 text-xs font-medium",
                trend === "up" && "bg-success/10 text-success",
                trend === "down" && "bg-destructive/10 text-destructive",
                trend === "neutral" && "bg-muted text-muted-foreground"
              )}
            >
              {trend === "up" ? "↑" : trend === "down" ? "↓" : "•"}
            </motion.div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
};

export default AnimatedCard;
