import { Container } from 'react-bootstrap';

export default function Footer() {
  return (
    <footer className="border-top border-secondary mt-5 py-4 small bg-dark" style={{background:'linear-gradient(180deg,#151a21,#0f1115)'}}>
      <Container className="d-flex flex-column flex-md-row justify-content-between gap-3">
        <div>© {new Date().getFullYear()} Emanuele Bernacchi</div>
        <div className="d-flex gap-3 flex-wrap">
          <a href="https://github.com/Bernuz2003" target="_blank" rel="noopener noreferrer">GitHub</a>
          <a href="https://www.linkedin.com" target="_blank" rel="noopener noreferrer">LinkedIn</a>
          <a href="mailto:email@example.com">Email</a>
          <a href="/CV.pdf" target="_blank" rel="noopener noreferrer">CV</a>
        </div>
      </Container>
    </footer>
  );
}