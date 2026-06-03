/**
 * Vitest test setup file.
 * 
 * Configures the test environment with necessary polyfills and mocks
 * for React Testing Library and components that depend on browser APIs.
 */

import '@testing-library/jest-dom';

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
Object.defineProperty(window, 'matchMedia', {
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
