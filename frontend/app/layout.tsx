import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Scientific Laboratory AR Assistant",
  description: "Browser-native AR guidance for real-time laboratory instrument identification."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
