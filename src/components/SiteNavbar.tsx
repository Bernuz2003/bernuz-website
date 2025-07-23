import { useState } from 'react';
import { Link, NavLink } from 'react-router-dom';
import { Container, Nav, Navbar } from 'react-bootstrap';

export default function SiteNavbar() {
  const [expanded, setExpanded] = useState(false);
  const close = () => setExpanded(false);
  return (
    <Navbar expanded={expanded} expand="lg" bg="dark" variant="dark" fixed="top" className="border-bottom border-secondary" style={{backdropFilter:'blur(6px)', background:'rgba(17,21,27,0.90)'}}>
      <Container>
        <Navbar.Brand as={Link} to="/" onClick={close}>Bernuz<span className="text-info">.dev</span></Navbar.Brand>
        <Navbar.Toggle aria-controls="mainnav" onClick={() => setExpanded(prev=>!prev)} />
        <Navbar.Collapse id="mainnav">
          <Nav className="ms-auto gap-lg-2">
            <Nav.Link as={NavLink} to="/" onClick={close} end>Home</Nav.Link>
            <Nav.Link as={NavLink} to="/about" onClick={close}>Chi sono</Nav.Link>
            <Nav.Link as={NavLink} to="/projects" onClick={close}>Progetti</Nav.Link>
            <Nav.Link as={NavLink} to="/contact" onClick={close}>Contatti</Nav.Link>
          </Nav>
        </Navbar.Collapse>
      </Container>
    </Navbar>
  );
}