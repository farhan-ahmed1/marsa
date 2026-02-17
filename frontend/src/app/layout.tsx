import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
	title: "MARSA",
	description: "Multi-Agent Research Assistant",
};

export default function RootLayout({
	children,
}: {
	children: React.ReactNode;
}) {
	return (
		<html lang="en" className="dark">
			<head>
				<link rel="preconnect" href="https://fonts.googleapis.com" />
				<link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
			</head>
			<body className="min-h-screen bg-terminal-black text-terminal-white font-mono">
				{children}
			</body>
		</html>
	);
}
