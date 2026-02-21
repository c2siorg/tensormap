import tailwindcssAnimate from "tailwindcss-animate";

/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        "node-dense": { DEFAULT: "rgb(43, 161, 255)", header: "rgb(50, 113, 178)" },
        "node-input": { DEFAULT: "rgb(105, 172, 61)", header: "rgb(93, 149, 34)" },
        "node-flatten": { DEFAULT: "rgb(247, 173, 20)", header: "rgb(170, 121, 24)" },
        "node-conv": { DEFAULT: "rgb(255, 128, 43)", header: "rgb(255, 128, 43)" },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
    },
  },
  plugins: [tailwindcssAnimate],
};
