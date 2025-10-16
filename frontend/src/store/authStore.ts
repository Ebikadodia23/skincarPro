import { create } from 'zustand';

interface AuthState {
  isAuthenticated: boolean;
  isLoading: boolean;
  user: any | null;
  initializeAuth: () => Promise<void>;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, username: string) => Promise<void>;
  logout: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  isAuthenticated: false,
  isLoading: true,
  user: null,

  initializeAuth: async () => {
    set({ isLoading: false });
  },

  login: async (email: string, _password: string) => {
    set({ isAuthenticated: true, user: { email } });
  },

  register: async (email: string, _password: string, username: string) => {
    set({ isAuthenticated: true, user: { email, username } });
  },

  logout: () => {
    set({ isAuthenticated: false, user: null });
  },
}));
