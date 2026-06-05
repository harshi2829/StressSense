import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Shield, Search, Download, Users, Activity, TrendingUp, BarChart3 } from "lucide-react";
import Navbar from "@/components/Navbar";
import AnimatedCard from "@/components/AnimatedCard";
import StressChart from "@/components/StressChart";
import UsersList from "@/components/UsersList";
import RealtimeStatus from "@/components/RealtimeStatus";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { useAuth } from "@/hooks/useAuth";
import { supabase } from "@/integrations/supabase/client";

interface Prediction {
  id: string;
  user_id: string;
  label: string;
  probability: number;
  stress_level: string | null;
  file_name: string | null;
  created_at: string;
  user_email?: string;
}

interface UserData {
  id: string;
  email: string;
  created_at: string;
  prediction_count: number;
  is_admin: boolean;
}

interface ChartData {
  time: string;
  stress: number;
  label: string;
}

const Admin = () => {
  const [searchQuery, setSearchQuery] = useState("");
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [users, setUsers] = useState<UserData[]>([]);
  const [chartData, setChartData] = useState<ChartData[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isRealtime, setIsRealtime] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | undefined>();
  const { isAdmin, signOut } = useAuth();

  // Subscribe to realtime updates
  useEffect(() => {
    const channel = supabase
      .channel("admin-predictions-realtime")
      .on(
        "postgres_changes",
        {
          event: "INSERT",
          schema: "public",
          table: "predictions",
        },
        () => {
          setLastUpdate(new Date());
          fetchPredictions();
        }
      )
      .subscribe((status) => {
        setIsRealtime(status === "SUBSCRIBED");
      });

    return () => {
      supabase.removeChannel(channel);
    };
  }, []);

  useEffect(() => {
    fetchPredictions();
    fetchUsers();
  }, []);

  const fetchUsers = async () => {
    const { data: profiles } = await supabase
      .from("profiles")
      .select("id, email, created_at");

    const { data: roles } = await supabase
      .from("user_roles")
      .select("user_id, role");

    const { data: predCounts } = await supabase
      .from("predictions")
      .select("user_id");

    if (profiles) {
      const userDataMap = new Map<string, UserData>();
      
      profiles.forEach((profile) => {
        const isAdmin = roles?.some(r => r.user_id === profile.id && r.role === "admin") || false;
        const predictionCount = predCounts?.filter(p => p.user_id === profile.id).length || 0;
        
        userDataMap.set(profile.id, {
          id: profile.id,
          email: profile.email || "Unknown",
          created_at: profile.created_at,
          prediction_count: predictionCount,
          is_admin: isAdmin,
        });
      });
      
      setUsers(Array.from(userDataMap.values()));
    }
  };

  const fetchPredictions = async () => {
    setIsLoading(true);

    const { data: predictionsData } = await supabase
      .from("predictions")
      .select("*")
      .order("created_at", { ascending: false })
      .limit(100);

    if (predictionsData) {
      const userIds = [...new Set(predictionsData.map((p) => p.user_id))];

      const { data: profiles } = await supabase
        .from("profiles")
        .select("id, email")
        .in("id", userIds);

      const profileMap = new Map(profiles?.map((p) => [p.id, p.email]) || []);

      const predictionsWithEmail = predictionsData.map((p) => ({
        ...p,
        user_email: profileMap.get(p.user_id) || "Unknown",
      }));

      setPredictions(predictionsWithEmail);

      // Generate chart data from last 7 days
      const last7Days = predictionsData
        .slice(0, 20)
        .reverse()
        .map((p, i) => ({
          time: new Date(p.created_at).toLocaleTimeString("en-US", {
            hour: "2-digit",
            minute: "2-digit",
          }),
          stress: Math.round(Number(p.probability) * 100),
          label: p.label,
        }));

      setChartData(last7Days);
    }

    setIsLoading(false);
  };

  const filteredPredictions = predictions.filter(
    (prediction) =>
      (prediction.user_email?.toLowerCase().includes(searchQuery.toLowerCase()) || false) ||
      prediction.label.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const stressCount = predictions.filter((p) => p.label === "Stress").length;
  const baselineCount = predictions.filter((p) => p.label !== "Stress").length;
  const highStressCount = predictions.filter((p) => p.stress_level === "high").length;

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString("en-US", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const getStressLevelBadge = (level: string | null) => {
    const config = {
      low: "bg-stress-low/20 text-stress-low hover:bg-stress-low/30",
      medium: "bg-stress-medium/20 text-stress-medium hover:bg-stress-medium/30",
      high: "bg-stress-high/20 text-stress-high hover:bg-stress-high/30",
    };
    return config[level as keyof typeof config] || "bg-muted text-muted-foreground";
  };

  const exportToCSV = () => {
    const headers = ["Timestamp", "User", "Label", "Stress Level", "Probability"];
    const rows = predictions.map((p) => [
      formatDate(p.created_at),
      p.user_email || "Unknown",
      p.label,
      p.stress_level || "N/A",
      `${(Number(p.probability) * 100).toFixed(1)}%`,
    ]);

    const csvContent = [headers, ...rows].map((row) => row.join(",")).join("\n");
    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "predictions.csv";
    a.click();
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar isLoggedIn onLogout={signOut} isAdmin={isAdmin} />

      <main className="container mx-auto px-4 py-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between"
        >
          <div className="flex items-center gap-4">
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ type: "spring", stiffness: 200 }}
              className="flex h-14 w-14 items-center justify-center rounded-2xl gradient-primary shadow-lg"
            >
              <Shield className="h-7 w-7 text-primary-foreground" />
            </motion.div>
            <div>
              <h1 className="text-3xl font-bold text-foreground sm:text-4xl">
                Admin Dashboard
              </h1>
              <p className="text-muted-foreground">
                Monitor stress analysis reports and user activity
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <RealtimeStatus isConnected={isRealtime} lastUpdate={lastUpdate} />
            <Button variant="outline" className="gap-2" onClick={exportToCSV}>
              <Download className="h-4 w-4" />
              Export
            </Button>
          </div>
        </motion.div>

        {/* Stats */}
        <div className="mb-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <AnimatedCard
            icon={Activity}
            label="Total Analyses"
            value={predictions.length}
            delay={0}
          />
          <AnimatedCard
            icon={Users}
            label="Total Users"
            value={users.length}
            delay={0.1}
          />
          <AnimatedCard
            icon={TrendingUp}
            label="Stress Detected"
            value={stressCount}
            delay={0.2}
            iconClassName="bg-stress-medium/10"
          />
          <AnimatedCard
            icon={BarChart3}
            label="High Stress Cases"
            value={highStressCount}
            delay={0.3}
            iconClassName="bg-stress-high/10"
          />
        </div>

        {/* Charts and Users Grid */}
        <div className="mb-8 grid gap-6 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <StressChart data={chartData} title="Recent Stress Trends" />
          </div>
          <UsersList users={users} isLoading={isLoading} />
        </div>

        {/* Predictions table */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <Card className="card-shadow-elevated overflow-hidden">
            <div className="h-1 gradient-primary" />
            <CardHeader>
              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <CardTitle className="text-xl">Prediction History</CardTitle>
                  <CardDescription>
                    All stress analysis results from users
                  </CardDescription>
                </div>
                <div className="relative w-full sm:w-72">
                  <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    placeholder="Search by user or label..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-10"
                  />
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="flex items-center justify-center py-12">
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                    className="h-8 w-8 rounded-full border-2 border-primary border-t-transparent"
                  />
                </div>
              ) : predictions.length === 0 ? (
                <div className="text-center py-12 text-muted-foreground">
                  No predictions yet. Users need to run analyses first.
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead className="w-[180px]">Timestamp</TableHead>
                        <TableHead>User</TableHead>
                        <TableHead>Label</TableHead>
                        <TableHead>Stress Level</TableHead>
                        <TableHead className="text-right">Confidence</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {filteredPredictions.map((prediction, index) => (
                        <motion.tr
                          key={prediction.id}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: index * 0.02 }}
                          className="border-b transition-colors hover:bg-muted/50"
                        >
                          <TableCell className="font-mono text-sm text-muted-foreground">
                            {formatDate(prediction.created_at)}
                          </TableCell>
                          <TableCell className="font-medium">
                            {prediction.user_email}
                          </TableCell>
                          <TableCell>
                            <Badge
                              variant="secondary"
                              className={
                                prediction.label === "Stress"
                                  ? "bg-stress-medium/20 text-stress-medium"
                                  : "bg-success/20 text-success"
                              }
                            >
                              {prediction.label}
                            </Badge>
                          </TableCell>
                          <TableCell>
                            <Badge
                              variant="secondary"
                              className={getStressLevelBadge(prediction.stress_level)}
                            >
                              {prediction.stress_level
                                ? prediction.stress_level.charAt(0).toUpperCase() +
                                  prediction.stress_level.slice(1)
                                : "N/A"}
                            </Badge>
                          </TableCell>
                          <TableCell className="text-right font-semibold">
                            {(Number(prediction.probability) * 100).toFixed(1)}%
                          </TableCell>
                        </motion.tr>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>
      </main>
    </div>
  );
};

export default Admin;