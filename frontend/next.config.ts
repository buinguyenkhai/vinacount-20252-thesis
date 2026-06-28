import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async redirects() {
    return [
      {
        source: "/runs/:path*",
        destination: "/",
        permanent: false,
      },
    ];
  },
};

export default nextConfig;
