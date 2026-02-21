import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Enable standalone output for Docker — copies only the minimal
  // set of files needed to run in production.
  output: "standalone",
};

export default nextConfig;
