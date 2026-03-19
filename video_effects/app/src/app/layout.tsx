import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Sidhant's Epic Video Effects Studio",
  description: "Video effects workflow preview",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-bg text-text min-h-screen antialiased font-mono">
        <header className="max-w-6xl mx-auto px-4 pt-6 pb-2 flex items-center gap-6">
          <span className="text-sm font-medium text-text-muted">Sidhant's Epic Video Effects Studio</span>
          <nav className="flex gap-4 text-sm">
            <Link href="/" className="text-text-dim hover:text-text transition-colors">VFX Studio</Link>
            <Link href="/templates" className="text-text-dim hover:text-text transition-colors">Gallery</Link>
          </nav>
        </header>
        <main className="max-w-6xl mx-auto px-4 py-8">{children}</main>
      </body>
    </html>
  );
}
