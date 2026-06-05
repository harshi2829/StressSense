import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Activity, TrendingUp, Clock, FileText, Zap } from "lucide-react";
import Navbar from "@/components/Navbar";
import FileUpload from "@/components/FileUpload";
import StressIndicator, { StressLevel } from "@/components/StressIndicator";
import StressSuggestions from "@/components/StressSuggestions";
import RealtimeStatus from "@/components/RealtimeStatus";
import AnimatedCard from "@/components/AnimatedCard";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/hooks/useAuth";
import { supabase } from "@/integrations/supabase/client";

interface PredictionResult {
  label: string;
  probability: number;
  stressLevel: StressLevel;
}

interface PredictionStats {
  todayCount: number;
  avgConfidence: number;
  lastAnalysis: string;
}

const Dashboard = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [statusText, setStatusText] = useState<"idle" | "processing" | "done" | "error">("idle");
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [isRealtime, setIsRealtime] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | undefined>();
  const [stats, setStats] = useState<PredictionStats>({
    todayCount: 0,
    avgConfidence: 0,
    lastAnalysis: "Never",
  });
  const { toast } = useToast();
  const { user, isAdmin, signOut } = useAuth();

  // Subscribe to realtime updates
  useEffect(() => {
    if (!user) return;

    const channel = supabase
      .channel("predictions-realtime")
      .on(
        "postgres_changes",
        {
          event: "INSERT",
          schema: "public",
          table: "predictions",
          filter: `user_id=eq.${user.id}`,
        },
        () => {
          setLastUpdate(new Date());
          fetchStats();
          toast({
            title: "New Analysis Recorded",
            description: "Your prediction history has been updated.",
          });
        }
      )
      .subscribe((status) => {
        setIsRealtime(status === "SUBSCRIBED");
      });

    return () => {
      supabase.removeChannel(channel);
    };
  }, [user]);

  useEffect(() => {
    if (user) {
      fetchStats();
    }
  }, [user]);

  const fetchStats = async () => {
    if (!user) return;

    const today = new Date();
    today.setHours(0, 0, 0, 0);

    const { data: todayData } = await supabase
      .from("predictions")
      .select("*")
      .eq("user_id", user.id)
      .gte("created_at", today.toISOString());

    const { data: allData } = await supabase
      .from("predictions")
      .select("probability, created_at")
      .eq("user_id", user.id)
      .order("created_at", { ascending: false });

    if (allData && allData.length > 0) {
      const avgConf = allData.reduce((sum, p) => sum + Number(p.probability), 0) / allData.length;
      const lastDate = new Date(allData[0].created_at);
      const timeDiff = Date.now() - lastDate.getTime();
      const hours = Math.floor(timeDiff / (1000 * 60 * 60));

      setStats({
        todayCount: todayData?.length || 0,
        avgConfidence: Math.round(avgConf * 100),
        lastAnalysis: hours < 1 ? "Just now" : `${hours}h ago`,
      });
    }
  };

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setResult(null);
    setStatusText("idle");
  };

  const handleClearFile = () => {
    setSelectedFile(null);
    setResult(null);
    setStatusText("idle");
  };

  const getStressLevel = (probability: number): StressLevel => {
    if (probability < 0.4) return "low";
    if (probability < 0.7) return "medium";
    return "high";
  };

  const savePrediction = async (label: string, probability: number, stressLevel: string, fileName: string) => {
    if (!user) return;

    await supabase.from("predictions").insert({
      user_id: user.id,
      label,
      probability,
      stress_level: stressLevel,
      file_name: fileName,
    });

    fetchStats();
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setStatusText("processing");

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch("http://localhost:8000/predict-file", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Prediction failed");
      }

      const data = await response.json();
      const stressLevel = getStressLevel(data.probability);
      const predictionResult: PredictionResult = {
        label: data.label,
        probability: data.probability,
        stressLevel,
      };

      setResult(predictionResult);
      await savePrediction(data.label, data.probability, stressLevel, selectedFile.name);
      setStatusText("done");
      setStats((prev) => ({
  todayCount: prev.todayCount + 1,
  avgConfidence: Math.round(predictionResult.probability * 100),
  lastAnalysis: "Just now",
}));


      toast({
        title: "Analysis Complete",
        description: `Stress Level: ${stressLevel.toUpperCase()}`,
      });
    } catch (error) {
      // Mock prediction for demo
      const mockProbability = Math.random();
      const stressLevel = getStressLevel(mockProbability);
      const isStress = mockProbability > 0.4;

      const mockResult: PredictionResult = {
        label: isStress ? "Stress" : "No Stress (Baseline)",
        probability: mockProbability,
        stressLevel,
      };

      setResult(mockResult);
      await savePrediction(mockResult.label, mockResult.probability, stressLevel, selectedFile.name);
      setStatusText("done");
      setStats((prev) => ({
  todayCount: prev.todayCount + 1,
  avgConfidence: Math.round(mockResult.probability * 100),
  lastAnalysis: "Just now",
}));


      toast({
        title: "Analysis Complete (Demo)",
        description: `Using mock data. Connect your API at /predict-file`,
        variant: "default",
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.1 },
    },
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar isLoggedIn onLogout={signOut} isAdmin={isAdmin} />

      <main className="container mx-auto px-4 py-8">
        {/* Welcome section with realtime status */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between"
        >
          <div>
           <h1 className="text-3xl font-bold text-foreground sm:text-4xl">
  Welcome
</h1>

            <p className="mt-2 text-muted-foreground">
              Upload your physiological data to analyze stress levels in real-time
            </p>
          </div>
          <RealtimeStatus isConnected={true} lastUpdate={lastUpdate} />
        </motion.div>

        {/* Stats grid */}
        <motion.div
          variants={container}
          initial="hidden"
          animate="show"
          className="mb-8 grid gap-4 sm:grid-cols-3"
        >
          <AnimatedCard
            icon={FileText}
            label="Analyses Today"
            value={stats.todayCount}
            delay={0}
          />
          <AnimatedCard
            icon={TrendingUp}
            label="Avg. Confidence"
            value={stats.avgConfidence > 0 ? `${stats.avgConfidence}%` : "N/A"}
            delay={0.1}
          />
          <AnimatedCard
            icon={Clock}
            label="Last Analysis"
            value={stats.lastAnalysis}
            delay={0.2}
          />
        </motion.div>

        {/* Upload section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <Card className="mb-8 card-shadow-elevated overflow-hidden">
            <div className="h-1 gradient-primary" />
            <CardHeader>
              <div className="flex items-center gap-3">
                <motion.div
                  animate={{ rotate: isAnalyzing ? 360 : 0 }}
                  transition={{ duration: 2, repeat: isAnalyzing ? Infinity : 0, ease: "linear" }}
                  className="flex h-12 w-12 items-center justify-center rounded-xl gradient-primary"
                >
                  <Activity className="h-6 w-6 text-primary-foreground" />
                </motion.div>
                <div>
                  <CardTitle className="text-xl">Stress Analysis</CardTitle>
                  <CardDescription>
                    Upload a CSV file with ECG, Respiration, EMG, and physiological data
                  </CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <FileUpload
                onFileSelect={handleFileSelect}
                selectedFile={selectedFile}
                isLoading={isAnalyzing}
                onClear={handleClearFile}
              />

              {/* LOADING STATUS DISPLAY */}
              {statusText === "processing" && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground p-3 bg-muted/50 rounded-lg">
                  <div className="animate-spin h-4 w-4 rounded-full border-2 border-primary border-t-transparent" />
                  Processing your CSV data...
                </div>
              )}

              {statusText === "done" && result && (
                <div className="flex items-center gap-2 text-sm text-emerald-600 p-3 bg-emerald-50/50 rounded-lg border border-emerald-200">
                  <Zap className="h-4 w-4" />
                  Prediction Ready
                </div>
              )}

              {selectedFile && statusText === "idle" && !result && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  <Button
                    onClick={handleAnalyze}
                    size="lg"
                    className="w-full gap-2 gradient-primary text-primary-foreground sm:w-auto"
                  >
                    <Zap className="h-4 w-4" />
                    Analyze Data
                  </Button>
                </motion.div>
              )}
            </CardContent>
          </Card>
        </motion.div>

        {/* Results section */}
        {result && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
            className="space-y-6"
          >
            <h2 className="text-2xl font-bold text-foreground">Analysis Results</h2>

            <div className="grid gap-6 lg:grid-cols-2">
              <StressIndicator
                level={result.stressLevel}
                probability={result.probability}
              />
              <StressSuggestions level={result.stressLevel} />
            </div>

            <Button
              variant="outline"
              onClick={handleClearFile}
              className="mt-6"
            >
              New Analysis
            </Button>
          </motion.div>
        )}
      </main>
    </div>
  );
};

export default Dashboard;
