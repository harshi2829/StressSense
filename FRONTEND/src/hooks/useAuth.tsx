import { createContext, useContext, useState, ReactNode } from "react";
import { User } from "@supabase/supabase-js";

// Dummy user object (no backend)
const dummyUser: User = {
  id: "demo-user",
  aud: "authenticated",
  role: "authenticated",
  email: "demo@stresswatch.ai",
  phone: "",
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
  app_metadata: {},
  user_metadata: {},
};

interface AuthContextType {
  user: User | null;
  isAdmin: boolean;
  isLoading: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string) => Promise<void>;
  signOut: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<User | null>(null);

  return (
    <AuthContext.Provider
      value={{
        user,
        isAdmin: false,
        isLoading: false,

        // ✅ Dummy sign in
        signIn: async () => {
          setUser(dummyUser);
        },

        // ✅ Dummy sign up
        signUp: async () => {
          setUser(dummyUser);
        },

        // ✅ Dummy sign out
        signOut: () => {
          setUser(null);
        },
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
};
