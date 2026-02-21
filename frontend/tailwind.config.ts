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
				sans: ["Inter", "IBM Plex Sans", "-apple-system", "sans-serif"],
			},
			colors: {
				// Dark navy-tinted background palette — values come from CSS custom properties
				// so the light/dark toggle works by redefining the RGB channel vars.
				terminal: {
					black:        "rgb(var(--terminal-black) / <alpha-value>)",
					surface:      "rgb(var(--terminal-surface) / <alpha-value>)",
					surfaceHover: "rgb(var(--terminal-surface-hover) / <alpha-value>)",
					border:       "rgb(var(--terminal-border) / <alpha-value>)",
					borderDotted: "rgb(var(--terminal-border-dotted) / <alpha-value>)",
					white:        "rgb(var(--terminal-white) / <alpha-value>)",
					mid:          "rgb(var(--terminal-mid) / <alpha-value>)",
					dim:          "rgb(var(--terminal-dim) / <alpha-value>)",
					pure:         "rgb(var(--terminal-pure) / <alpha-value>)",
				},
				// Agent colors — each agent has a distinct hue
				agent: {
					planner: "#60A5FA",      // blue-400
					researcher: "#34D399",    // emerald-400
					fact_checker: "#FBBF24",  // amber-400
					synthesizer: "#A78BFA",   // violet-400
					system: "#94A3B8",        // slate-400
				},
				// Blue primary accent
				accent: {
					DEFAULT: "#3B82F6",
					hover: "#2563EB",
					subtle: "rgba(59,130,246,0.12)",
					subtleBorder: "rgba(59,130,246,0.3)",
					foreground: "hsl(var(--accent-foreground))",
				},
				// Semantic colors with proper vibrancy
				semantic: {
					pass: "#22C55E",
					passSubtle: "rgba(34,197,94,0.12)",
					passBorder: "rgba(34,197,94,0.3)",
					unknown: "#F59E0B",
					unknownSubtle: "rgba(245,158,11,0.12)",
					unknownBorder: "rgba(245,158,11,0.3)",
					fail: "#EF4444",
					failSubtle: "rgba(239,68,68,0.12)",
					failBorder: "rgba(239,68,68,0.3)",
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

				destructive: {
					DEFAULT: "hsl(var(--destructive))",
					foreground: "hsl(var(--destructive-foreground))",
				},
				border: "hsl(var(--border))",
				input: "hsl(var(--input))",
				ring: "hsl(var(--ring))",
			},
			borderRadius: {
				sm: "4px",
				md: "6px",
				lg: "10px",
				xl: "14px",
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

