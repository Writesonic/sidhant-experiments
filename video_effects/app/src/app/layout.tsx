import type { Metadata } from "next";
import "./globals.css";

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
      <body className="bg-neutral-950 text-neutral-100 min-h-screen">
        <main className="max-w-6xl mx-auto px-4 py-8">{children}</main>
      </body>
    </html>
  );
}
