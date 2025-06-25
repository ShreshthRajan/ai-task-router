import './globals.css'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'AI Development Intelligence',
  description: 'Revolutionary AI-powered task assignment optimization system',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}