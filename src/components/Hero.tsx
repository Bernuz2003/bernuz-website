import { Container, Button } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import './../styles/Hero.css';

export default function Hero() {
  return (
    <div className="hero-gradient hero-section">
      <Container className="hero-container">
        <div className="col-lg-9 col-xl-8 hero-content">
          <h1 className="hero-title">
            Ciao, sono <span className="text-info">Emanuele</span>.<br />
            Esploro tecnologie, algoritmi e soluzioni innovative.
          </h1>
          <p className="hero-text">
            Studente di Ingegneria Informatica al Politecnico di Torino. Hub personale per progetti accademici, esperimenti di programmazione, visualizzazioni interattive e applicazioni web. Dal machine learning ai sorting visualizer, un mix di soluzioni pratiche e creatività tecnica.
          </p>
          <div className="hero-buttons">
            <Button 
              as={Link} 
              to="/projects" 
              variant="info" 
              className="hero-btn-primary"
            >
              Esplora Progetti
            </Button>
            <Button 
              href="/CV.pdf" 
              variant="outline-info" 
              className="hero-btn-secondary"
            >
              Scarica CV
            </Button>
          </div>
        </div>
      </Container>
    </div>
  );
}