import "./globals.css";
export const metadata = {
    title: "IrisSecure | Biometric Intelligence",
    description: "Privacy-first iris recognition and verification dashboard.",
};
export default function RootLayout({ children }) {
    return <html lang="en"><body>{children}</body></html>;
}
