import type { NextConfig } from "next";
import path from "path";

const remotionSrc = path.resolve(__dirname, "../remotion/src");

const nextConfig: NextConfig = {
  turbopack: {
    resolveAlias: {
      "@remotion-project": remotionSrc,
    },
  },
  webpack: (config) => {
    config.resolve.alias["@remotion-project"] = remotionSrc;
    // Deduplicate remotion — imported components from ../remotion/src must use
    // the same remotion instance as @remotion/player to share React context.
    config.resolve.alias["remotion"] = path.resolve(
      __dirname,
      "node_modules/remotion"
    );
    return config;
  },
  transpilePackages: ["remotion", "@remotion/player", "@remotion/google-fonts"],
};

export default nextConfig;
