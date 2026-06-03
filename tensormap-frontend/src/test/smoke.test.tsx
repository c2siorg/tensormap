import React from "react";
import { render } from "@testing-library/react";
import App from "../App";
import { RecoilRoot } from "recoil";
import { describe, it, expect } from "vitest";

describe("Frontend Smoke Test", () => {
  it("renders the App component without throwing", () => {
    // App already includes BrowserRouter internally, so we only wrap with RecoilRoot
    expect(() => {
      render(
        <RecoilRoot>
          <App />
        </RecoilRoot>,
      );
    }).not.toThrow();
  });
});
