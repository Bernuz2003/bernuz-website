import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import PageTransition from './components/PageTransition.tsx';
import SiteNavbar from './components/SiteNavbar';
import Footer from './components/Footer';
import CommandPalette from './components/CommandPalette';
import HomePage from './pages/HomePage.tsx';
import WorkPage from './pages/WorkPage.tsx';
import WorkDetailPage from './pages/WorkDetailPage.tsx';
import ProgressBar from './components/ProgressBar.tsx';
import TerminalDrawer from './components/TerminalDrawer.tsx';
import MatrixRain from './components/MatrixRain.tsx';
import { useEffect } from 'react';

function AnimatedRoutes() {
  const location = useLocation();
  
  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/" element={<PageTransition><HomePage /></PageTransition>} />
        <Route path="/work" element={<PageTransition><WorkPage /></PageTransition>} />
        <Route path="/work/:slug" element={<PageTransition><WorkDetailPage /></PageTransition>} />
        <Route path="*" element={
          <PageTransition>
            <div className="container pt-5 text-center">
              <h2 className="text-info font-mono mb-4">404 - Not Found</h2>
              <p className="text-secondary">La pagina che cerchi non esiste.</p>
            </div>
          </PageTransition>
        } />
      </Routes>
    </AnimatePresence>
  );
}

export default function App() {
  useEffect(() => {
    let ticking = false;
    const handleMouseMove = (e: MouseEvent) => {
      if (!ticking) {
        window.requestAnimationFrame(() => {
          document.documentElement.style.setProperty('--mouse-x', `${e.clientX}px`);
          document.documentElement.style.setProperty('--mouse-y', `${e.clientY}px`);
          ticking = false;
        });
        ticking = true;
      }
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  return (
    <BrowserRouter>
      <ProgressBar />
      <MatrixRain />
      <div className="spotlight" />
      <CommandPalette />
      <TerminalDrawer />
      <div className="d-flex flex-column min-vh-100">
        <SiteNavbar />
        <div className="flex-grow-1 d-flex flex-column" style={{ paddingTop: '80px' }}>
          <AnimatedRoutes />
        </div>
        <Footer />
      </div>
    </BrowserRouter>
  );
}