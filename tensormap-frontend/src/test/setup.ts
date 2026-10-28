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

// Polyfill localStorage (in Node 22+, globalThis.localStorage is an experimental
// built-in that defaults to undefined, and happy-dom does not provide a default Storage).
class LocalStorageMock implements Storage {
  private store: Record<string, string> = {};

  get length(): number {
    return Object.keys(this.store).length;
  }

  clear(): void {
    this.store = {};
  }

  getItem(key: string): string | null {
    return this.store[key] ?? null;
  }

  key(index: number): string | null {
    return Object.keys(this.store)[index] ?? null;
  }

  removeItem(key: string): void {
    delete this.store[key];
  }

  setItem(key: string, value: string): void {
    this.store[key] = String(value);
  }
}

const storageInstance = new LocalStorageMock();

if (typeof window !== "undefined") {
  Object.defineProperty(window, "localStorage", {
    value: storageInstance,
    writable: true,
    configurable: true,
  });
}

Object.defineProperty(globalThis, "localStorage", {
  value: storageInstance,
  writable: true,
  configurable: true,
});
