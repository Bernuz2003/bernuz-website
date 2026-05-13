import { BrowserRouter, Routes, Route } from 'react-router-dom';
import SiteNavbar from './components/SiteNavbar';
import Footer from './components/Footer';
import CommandPalette from './components/CommandPalette';
import HomePage from './pages/HomePage.tsx';
import ProjectsPage from './pages/ProjectsPage.tsx';
import MLLabDetailPage from './pages/MLLabDetailPage.tsx';
import MLLabsPage from './pages/MLLabsPage.tsx'; 
import ProjectDetailPage from './pages/ProjectDetailPage.tsx';

export default function App() {
  return (
    <BrowserRouter>
      <CommandPalette />
      <div className="d-flex flex-column min-vh-100">
        <SiteNavbar />
        <div className="flex-grow-1 d-flex flex-column" style={{ paddingTop: '80px' }}>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/projects" element={<ProjectsPage />} />
            <Route path="/projects/:slug" element={<ProjectDetailPage />} />
            <Route path="/ml-labs" element={<MLLabsPage />} /> 
            <Route path="/ml-labs/:slug" element={<MLLabDetailPage />} />
          </Routes>
        </div>
        <Footer />
      </div>
    </BrowserRouter>
  );
}