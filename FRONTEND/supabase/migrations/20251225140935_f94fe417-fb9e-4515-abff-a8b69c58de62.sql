-- Add stress_level column to predictions table
ALTER TABLE public.predictions 
ADD COLUMN stress_level text DEFAULT 'low' CHECK (stress_level IN ('low', 'medium', 'high'));

-- Enable realtime for predictions table
ALTER PUBLICATION supabase_realtime ADD TABLE public.predictions;