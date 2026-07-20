import { useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { Nav, Navbar } from 'react-bootstrap';

export default function SiteNavbar() {
  const [expanded, setExpanded] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const close = () => setExpanded(false);

  const handleScrollToSection = (sectionId: string) => {
    close();
    if (location.pathname === '/') {
      document.getElementById(sectionId)?.scrollIntoView({ behavior: 'smooth' });
    } else {
      navigate('/');
      setTimeout(() => {
        document.getElementById(sectionId)?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    }
  };

  return (
    <div className="fixed-top w-100 d-flex justify-content-center mt-3 mt-md-4 px-3" style={{ zIndex: 1030, pointerEvents: 'none' }}>
      <Navbar expanded={expanded} expand="lg" variant="dark" className="glass-panel py-2 px-4 shadow-lg w-100" style={{ maxWidth: '900px', pointerEvents: 'auto' }}>
        <Navbar.Brand as={Link} to="/" onClick={close} className="font-mono fw-bold">Bernuz<span className="text-info">.dev</span></Navbar.Brand>
        <Navbar.Toggle aria-controls="mainnav" onClick={() => setExpanded(prev => !prev)} />
        <Navbar.Collapse id="mainnav">
          <Nav className="ms-auto gap-lg-2">
            <Nav.Link as={Link} to="/work" onClick={close}>Work</Nav.Link>
            <Nav.Link onClick={() => handleScrollToSection('about')}>About</Nav.Link>
            <Nav.Link onClick={() => handleScrollToSection('contact')}>Contact</Nav.Link>
          </Nav>
          <div className="d-none d-lg-block ms-4 ps-4 border-start border-secondary">
            <span className="badge bg-secondary text-dark font-mono">⌘K</span>
          </div>
        </Navbar.Collapse>
      </Navbar>
    </div>
  );
}