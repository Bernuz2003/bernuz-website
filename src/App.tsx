import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import PageTransition from './components/PageTransition.tsx';
import SiteNavbar from './components/SiteNavbar';
import Footer from './components/Footer';
import CommandPalette from './components/CommandPalette';
import HomePage from './pages/HomePage.tsx';
import ProjectsPage from './pages/ProjectsPage.tsx';
import MLLabDetailPage from './pages/MLLabDetailPage.tsx';
import MLLabsPage from './pages/MLLabsPage.tsx'; 
import ProjectDetailPage from './pages/ProjectDetailPage.tsx';
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
        <Route path="/projects" element={<PageTransition><ProjectsPage /></PageTransition>} />
        <Route path="/projects/:slug" element={<PageTransition><ProjectDetailPage /></PageTransition>} />
        <Route path="/ml-labs" element={<PageTransition><MLLabsPage /></PageTransition>} /> 
        <Route path="/ml-labs/:slug" element={<PageTransition><MLLabDetailPage /></PageTransition>} />
      </Routes>
    </AnimatePresence>
  );
}

export default function App() {
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      document.documentElement.style.setProperty('--mouse-x', `${e.clientX}px`);
      document.documentElement.style.setProperty('--mouse-y', `${e.clientY}px`);
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