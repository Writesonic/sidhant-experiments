"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export function NavLink({ href, children }: { href: string; children: React.ReactNode }) {
  const pathname = usePathname();
  const active = href === "/" ? pathname === "/" : pathname.startsWith(href);

  return (
    <Link
      href={href}
      className={`pb-1 transition-colors ${
        active
          ? "text-accent border-b border-accent"
          : "text-text-dim hover:text-text"
      }`}
    >
      {children}
    </Link>
  );
}
