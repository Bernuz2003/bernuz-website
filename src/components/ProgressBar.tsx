import { motion, useScroll } from "framer-motion";

export default function ProgressBar() {
  const { scrollYProgress } = useScroll();

  return (
    <motion.div
      style={{
        scaleX: scrollYProgress,
        transformOrigin: "left",
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        height: "3px",
        background: "linear-gradient(90deg, var(--accent-cyan), var(--accent-purple))",
        zIndex: 2000,
      }}
    />
  );
}
