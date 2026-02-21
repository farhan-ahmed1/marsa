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
		<html lang="en" suppressHydrationWarning>
			<head>
				<link rel="preconnect" href="https://fonts.googleapis.com" />
				<link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
				{/* FOUC-prevention: apply saved theme before first paint */}
				<script
					dangerouslySetInnerHTML={{
						__html: `(function(){try{var t=localStorage.getItem('marsa-theme');if(t==='light'){document.documentElement.classList.add('light');}else{document.documentElement.classList.add('dark');}}catch(e){}})();`,
					}}
				/>
			</head>
			<body className="min-h-screen bg-terminal-black text-terminal-white font-mono">
				{children}
			</body>
		</html>
	);
}
