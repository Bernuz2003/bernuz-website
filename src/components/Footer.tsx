import { Container } from 'react-bootstrap';

export default function Footer() {
  return (
    <footer className="border-top border-secondary mt-5 py-4 small glass-panel" style={{ borderRadius: 0, borderLeft: 'none', borderRight: 'none', borderBottom: 'none' }}>
      <Container className="d-flex flex-column flex-md-row justify-content-between align-items-center gap-3">
        <div className="font-mono text-secondary">© {new Date().getFullYear()} Emanuele Bernacchi</div>
        <div className="d-flex gap-3 flex-wrap align-items-center">
          <a href="https://github.com/Bernuz2003" target="_blank" rel="noopener noreferrer" className="text-secondary hover-white">GitHub</a>
          <a href="https://www.linkedin.com" target="_blank" rel="noopener noreferrer" className="text-secondary hover-white">LinkedIn</a>
          <a href="mailto:email@example.com" className="text-secondary hover-white">Email</a>
          <a href="/CV.pdf" target="_blank" rel="noopener noreferrer" className="text-secondary hover-white">CV</a>
        </div>
      </Container>
    </footer>
  );
}