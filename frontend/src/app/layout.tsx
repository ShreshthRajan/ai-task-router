import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'AI Development Intelligence | Revolutionary Task Routing',
  description: 'The world\'s first truly intelligent task assignment system. Analyze GitHub repositories, predict task complexity, and optimize developer assignments with cutting-edge AI.',
  keywords: 'AI, task assignment, developer productivity, GitHub analysis, machine learning, software engineering',
  authors: [{ name: 'AI Task Router Team' }],
  viewport: 'width=device-width, initial-scale=1',
  robots: 'index, follow',
  openGraph: {
    title: 'AI Development Intelligence System',
    description: 'Revolutionary AI-powered task assignment and developer intelligence platform',
    type: 'website',
    locale: 'en_US',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'AI Development Intelligence System',
    description: 'Revolutionary AI-powered task assignment and developer intelligence platform',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="scroll-smooth">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body className={`${inter.className} antialiased`}>
        <div id="root">
          {children}
        </div>
      </body>
    </html>
  );
}