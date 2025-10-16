import { create } from 'zustand';

interface AnalysisState {
  results: any | null;
  clearResults: () => void;
  setResults: (results: any) => void;
}

export const useAnalysisStore = create<AnalysisState>((set) => ({
  results: null,
  clearResults: () => set({ results: null }),
  setResults: (results) => set({ results }),
}));
