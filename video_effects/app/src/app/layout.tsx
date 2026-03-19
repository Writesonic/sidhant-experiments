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
      <body className="bg-neutral-950 text-neutral-100 min-h-screen antialiased">
        <header className="max-w-6xl mx-auto px-4 pt-6 pb-2 flex items-center gap-6">
          <span className="text-sm font-medium text-neutral-500">Sidhant's Epic Video Effects Studio</span>
          <nav className="flex gap-4 text-sm">
            <Link href="/" className="text-neutral-400 hover:text-white transition-colors">VFX Studio</Link>
            <Link href="/templates" className="text-neutral-400 hover:text-white transition-colors">Gallery</Link>
          </nav>
        </header>
        <main className="max-w-6xl mx-auto px-4 py-8">{children}</main>
      </body>
    </html>
  );
}
