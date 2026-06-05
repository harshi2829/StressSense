import { AlertTriangle, Wind, Move, Coffee } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface StressResultCardProps {
  probability: number;
}

const StressResultCard = ({ probability }: StressResultCardProps) => {
  const suggestions = [
    {
      icon: Wind,
      title: "Deep Breathing",
      description: "Take 5 slow, deep breaths. Inhale for 4 seconds, hold for 4, exhale for 6.",
    },
    {
      icon: Move,
      title: "Quick Stretch",
      description: "Stand up and stretch your arms, neck, and shoulders for 2 minutes.",
    },
    {
      icon: Coffee,
      title: "Short Break",
      description: "Step away from your screen. Get some water or take a brief walk.",
    },
  ];

  return (
    <Card className="animate-slide-up border-warning/30 bg-gradient-to-br from-warning/5 to-warning/10 card-shadow">
      <CardHeader className="pb-4">
        <div className="flex items-center gap-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-warning/20">
            <AlertTriangle className="h-6 w-6 text-warning" />
          </div>
          <div>
            <CardTitle className="text-xl text-foreground">Stress Detected</CardTitle>
            <p className="text-sm text-muted-foreground">
              Confidence: {(probability * 100).toFixed(1)}%
            </p>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-muted-foreground">
          Your physiological signals indicate elevated stress levels. Here are some quick
          suggestions to help you relax:
        </p>

        <div className="grid gap-3 sm:grid-cols-3">
          {suggestions.map((suggestion, index) => (
            <div
              key={suggestion.title}
              className="rounded-lg border border-border bg-card p-4 transition-all hover:border-warning/50 hover:shadow-sm"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <suggestion.icon className="mb-2 h-5 w-5 text-warning" />
              <h4 className="font-medium text-foreground">{suggestion.title}</h4>
              <p className="mt-1 text-xs text-muted-foreground">{suggestion.description}</p>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

export default StressResultCard;
