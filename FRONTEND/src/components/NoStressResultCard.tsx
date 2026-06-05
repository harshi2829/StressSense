import { CheckCircle2, Sparkles } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface NoStressResultCardProps {
  probability: number;
}

const motivationalMessages = [
  "You're doing great! Keep up the positive momentum.",
  "Your body is in a calm, balanced state. Well done!",
  "Excellent! Your relaxation techniques are working.",
  "Your physiological markers show you're in a healthy baseline state.",
];

const NoStressResultCard = ({ probability }: NoStressResultCardProps) => {
  const message = motivationalMessages[Math.floor(Math.random() * motivationalMessages.length)];

  return (
    <Card className="animate-slide-up border-success/30 bg-gradient-to-br from-success/5 to-success/10 card-shadow">
      <CardHeader className="pb-4">
        <div className="flex items-center gap-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-success/20">
            <CheckCircle2 className="h-6 w-6 text-success" />
          </div>
          <div>
            <CardTitle className="text-xl text-foreground">No Stress (Baseline)</CardTitle>
            <p className="text-sm text-muted-foreground">
              Confidence: {(probability * 100).toFixed(1)}%
            </p>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-start gap-3 rounded-lg border border-success/20 bg-success/5 p-4">
          <Sparkles className="mt-0.5 h-5 w-5 flex-shrink-0 text-success" />
          <div>
            <p className="font-medium text-foreground">{message}</p>
            <p className="mt-1 text-sm text-muted-foreground">
              Continue maintaining your current routine and healthy habits.
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default NoStressResultCard;
