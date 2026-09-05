import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import "./index.css"
import App from "./App.tsx"
import { MotionConfig } from "framer-motion"
import { AppErrorBoundary } from "./components/AppErrorBoundary.tsx"

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <AppErrorBoundary>
      <MotionConfig reducedMotion="user" transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}>
        <App />
      </MotionConfig>
    </AppErrorBoundary>
  </StrictMode>,
)
