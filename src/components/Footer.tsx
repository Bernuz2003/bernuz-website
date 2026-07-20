import { Container } from 'react-bootstrap';

export default function Footer() {
  return (
    <footer className="border-top border-secondary mt-5 py-4 small glass-panel" style={{ borderRadius: 0, borderLeft: 'none', borderRight: 'none', borderBottom: 'none' }}>
      <Container className="d-flex flex-column flex-md-row justify-content-between align-items-center gap-3">
        <div className="font-mono text-secondary">© {new Date().getFullYear()} Emanuele Bernucci</div>
        <div className="d-flex gap-3 flex-wrap align-items-center">
          <a href="https://github.com/Bernuz2003" target="_blank" rel="noopener noreferrer" className="text-secondary hover-white">GitHub</a>
          <a href="https://linkedin.com/in/emanuele-bernucci" target="_blank" rel="noopener noreferrer" className="text-secondary hover-white">LinkedIn</a>
          <a href="mailto:hello@bernuz.me" className="text-secondary hover-white">Email</a>
        </div>
      </Container>
    </footer>
  );
}