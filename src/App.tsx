import { BrowserRouter, Routes, Route } from 'react-router-dom';
import SiteNavbar from './components/SiteNavbar';
import Footer from './components/Footer';
import HomePage from './pages/HomePage.tsx';
import ProjectsPage from './pages/ProjectsPage.tsx';
import MLLabDetailPage from './pages/MLLabDetailPage.tsx';
import MLLabsPage from './pages/MLLabsPage.tsx'; 
import ProjectDetailPage from './pages/ProjectDetailPage.tsx';

export default function App() {
  return (
    <BrowserRouter>
      <SiteNavbar />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/projects" element={<ProjectsPage />} />
        <Route path="/projects/:slug" element={<ProjectDetailPage />} />
        <Route path="/ml-labs" element={<MLLabsPage />} /> 
        <Route path="/ml-labs/:slug" element={<MLLabDetailPage />} />
      </Routes>
      <Footer />
    </BrowserRouter>
  );
}