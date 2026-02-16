import "./globals.css";

export const metadata = {
	title: "MARSA",
	description: "Multi-Agent Research Assistant",
};

export default function RootLayout({
	children,
}: {
	children: React.ReactNode;
}) {
	return (
		<html lang="en">
			<body>{children}</body>
		</html>
	);
}
