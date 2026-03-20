import type { Metadata } from "next";
import "./globals.css";
import { NavLink } from "@/components/NavLink";

export const metadata: Metadata = {
  title: "VFX Studio",
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
        <header className="max-w-screen-2xl mx-auto px-6 sm:px-8 lg:px-12 pt-6 pb-4 flex items-center gap-6 border-b border-border">
          <span className="text-sm font-medium text-text-muted">VFX Studio</span>
          <nav className="flex gap-4 text-sm">
            <NavLink href="/">Studio</NavLink>
            <NavLink href="/templates">Gallery</NavLink>
          </nav>
        </header>
        <main className="max-w-screen-2xl mx-auto px-6 sm:px-8 lg:px-12 pt-6 pb-8">{children}</main>
      </body>
    </html>
  );
}
