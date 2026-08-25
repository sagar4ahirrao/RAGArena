const isExport = process.env.NEXT_STATIC_EXPORT === "1";

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  ...(isExport
    ? { output: "export", images: { unoptimized: true }, trailingSlash: true }
    : {
        async rewrites() {
          return [
            { source: "/api/:path*", destination: "http://localhost:4000/api/:path*" },
          ];
        },
      }),
};

export default nextConfig;
