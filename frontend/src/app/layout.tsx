import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "./providers";
import { Header } from "./header";

export const metadata: Metadata = {
  title: "Vinacount",
  description:
    "Accounting irregularity risk-signal analysis for Vietnamese quarterly financial reports",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="vi" className="h-full antialiased">
      <body className="h-full flex flex-col">
        <Providers>
          <div className="h-0.5 bg-primary shrink-0" />
          <Header />
          <main className="flex-1 flex min-h-0">{children}</main>
        </Providers>
      </body>
    </html>
  );
}
