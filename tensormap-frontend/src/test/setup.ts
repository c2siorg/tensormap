/**
 * Vitest test setup file.
 *
 * Configures the test environment with necessary polyfills and mocks
 * for React Testing Library and components that depend on browser APIs.
 */

import "@testing-library/jest-dom";

// Mock ResizeObserver (required for Recharts and other layout-dependent components)
global.ResizeObserver = class ResizeObserver {
  observe() {
    // Mock implementation
  }
  unobserve() {
    // Mock implementation
  }
  disconnect() {
    // Mock implementation
  }
};

// Mock window.matchMedia (required for responsive components)
Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {}, // Deprecated
    removeListener: () => {}, // Deprecated
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => true,
  }),
});

// Mock localStorage (required for Canvas component draft persistence in tests)
const createLocalStorageMock = () => {
  const store: Record<string, string> = {};
  return {
    getItem: vi.fn((key: string) => store[key] ?? null),
    setItem: vi.fn((key: string, value: string) => { store[key] = value; }),
    removeItem: vi.fn((key: string) => { delete store[key]; }),
    clear: vi.fn(() => { Object.keys(store).forEach(k => delete store[k]); }),
    _store: store, // For test debugging if needed
  };
};
const localStorageMock = createLocalStorageMock();
Object.defineProperty(window, "localStorage", {
  writable: true,
  value: localStorageMock,
});
