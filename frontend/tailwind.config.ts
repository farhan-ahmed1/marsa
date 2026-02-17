import type { Config } from "tailwindcss";

const config: Config = {
	darkMode: ["class"],
	content: [
		"./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
		"./src/components/**/*.{js,ts,jsx,tsx,mdx}",
		"./src/app/**/*.{js,ts,jsx,tsx,mdx}",
	],
	theme: {
		extend: {
			fontFamily: {
				mono: ["IBM Plex Mono", "JetBrains Mono", "Fira Code", "monospace"],
				sans: ["IBM Plex Sans", "-apple-system", "sans-serif"],
			},
			colors: {
				// Monochrome palette (matching prototype DS)
				terminal: {
					black: "#0C0C0C",
					surface: "#141414",
					surfaceHover: "#1A1A1A",
					border: "#282828",
					borderDotted: "#333333",
					white: "#E0E0E0",
					mid: "#999999",
					dim: "#5A5A5A",
					pure: "#FFFFFF",
				},
				// Semantic colors
				semantic: {
					pass: "#5AE05A",
					unknown: "#D4C55A",
					fail: "#D45A5A",
				},
				// Shadcn variables
				background: "hsl(var(--background))",
				foreground: "hsl(var(--foreground))",
				card: {
					DEFAULT: "hsl(var(--card))",
					foreground: "hsl(var(--card-foreground))",
				},
				popover: {
					DEFAULT: "hsl(var(--popover))",
					foreground: "hsl(var(--popover-foreground))",
				},
				primary: {
					DEFAULT: "hsl(var(--primary))",
					foreground: "hsl(var(--primary-foreground))",
				},
				secondary: {
					DEFAULT: "hsl(var(--secondary))",
					foreground: "hsl(var(--secondary-foreground))",
				},
				muted: {
					DEFAULT: "hsl(var(--muted))",
					foreground: "hsl(var(--muted-foreground))",
				},
				accent: {
					DEFAULT: "hsl(var(--accent))",
					foreground: "hsl(var(--accent-foreground))",
				},
				destructive: {
					DEFAULT: "hsl(var(--destructive))",
					foreground: "hsl(var(--destructive-foreground))",
				},
				border: "hsl(var(--border))",
				input: "hsl(var(--input))",
				ring: "hsl(var(--ring))",
			},
			borderRadius: {
				sm: "3px",
				md: "6px",
				lg: "8px",
			},
			animation: {
				"blink": "blink 1s step-end infinite",
				"fade-in": "fadeIn 0.3s ease-out",
				"slide-in": "slideIn 0.3s ease-out",
				"slide-left": "slideLeft 0.3s ease-out",
				"pulse-dot": "pulse 2s ease-in-out infinite",
				"spin-slow": "spin 4s linear infinite",
				"grow": "grow 0.7s ease-out",
			},
			keyframes: {
				blink: {
					"0%, 100%": { opacity: "1" },
					"50%": { opacity: "0" },
				},
				fadeIn: {
					"0%": { opacity: "0" },
					"100%": { opacity: "1" },
				},
				slideIn: {
					"0%": { opacity: "0", transform: "translateY(6px)" },
					"100%": { opacity: "1", transform: "translateY(0)" },
				},
				slideLeft: {
					"0%": { opacity: "0", transform: "translateX(-10px)" },
					"100%": { opacity: "1", transform: "translateX(0)" },
				},
				pulse: {
					"0%, 100%": { opacity: "0.4" },
					"50%": { opacity: "1" },
				},
				grow: {
					"0%": { width: "0%" },
					"100%": { width: "100%" },
				},
			},
		},
	},
	plugins: [require("tailwindcss-animate")],
};

export default config;

