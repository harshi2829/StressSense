import { motion } from "framer-motion";
import { Radio, Wifi, WifiOff } from "lucide-react";
import { cn } from "@/lib/utils";

interface RealtimeStatusProps {
  isConnected: boolean;
  lastUpdate?: Date;
}

const RealtimeStatus = ({ isConnected, lastUpdate }: RealtimeStatusProps) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        "inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-medium",
        isConnected 
          ? "bg-success/10 text-success" 
          : "bg-muted text-muted-foreground"
      )}
    >
      {isConnected ? (
        <>
          <motion.div
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ duration: 1.5, repeat: Infinity }}
            className="relative"
          >
            <Radio className="h-3.5 w-3.5" />
            <motion.div
              animate={{ scale: [1, 2], opacity: [0.5, 0] }}
              transition={{ duration: 1.5, repeat: Infinity }}
              className="absolute inset-0 rounded-full bg-success"
            />
          </motion.div>
          <span>Live</span>
        </>
      ) : (
        <>
          <WifiOff className="h-3.5 w-3.5" />
          <span>Offline</span>
        </>
      )}
      {lastUpdate && (
        <span className="text-muted-foreground">
          · {new Date(lastUpdate).toLocaleTimeString()}
        </span>
      )}
    </motion.div>
  );
};

export default RealtimeStatus;
