import { useState } from 'react';
import { Link, NavLink, useNavigate, useLocation } from 'react-router-dom';
import { Container, Nav, Navbar } from 'react-bootstrap';

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
    <Navbar expanded={expanded} expand="lg" bg="dark" variant="dark" fixed="top" className="border-bottom border-secondary" style={{ backdropFilter: 'blur(6px)', background: 'rgba(17,21,27,0.90)' }}>
      <Container>
        <Navbar.Brand as={Link} to="/" onClick={close}>Bernuz<span className="text-info">.dev</span></Navbar.Brand>
        <Navbar.Toggle aria-controls="mainnav" onClick={() => setExpanded(prev => !prev)} />
        <Navbar.Collapse id="mainnav">
          <Nav className="ms-auto gap-lg-2">
            <Nav.Link onClick={() => handleScrollToSection('about')}>Chi sono</Nav.Link>
            <Nav.Link onClick={() => handleScrollToSection('ml')}>ML</Nav.Link>
            <Nav.Link onClick={() => handleScrollToSection('projects')}>Web Apps</Nav.Link>
            <Nav.Link onClick={() => handleScrollToSection('contact')}>Contatti</Nav.Link>
          </Nav>
        </Navbar.Collapse>
      </Container>
    </Navbar>
  );
}