import { Container, Button } from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';
import './../styles/Hero.css';

export default function Hero() {
  const navigate = useNavigate();

  return (
    <div className="hero-gradient hero-section">
      <Container className="hero-container">
        <div className="col-lg-9 col-xl-8 hero-content">
          <h1 className="hero-title">
            Ciao, sono <span className="text-info">Emanuele</span>.<br />
            Esploro tecnologie, algoritmi e soluzioni innovative.
          </h1>
          <p className="hero-text">
            Hub personale per progetti accademici, esperimenti di programmazione, visualizzazioni interattive e applicazioni web. Dal machine learning ai sorting visualizer, un mix di soluzioni pratiche e creatività tecnica.
          </p>
          <div className="hero-buttons">
            <Button 
              onClick={() => navigate('/ml-labs')}
              variant="info" 
              className="hero-btn-primary"
            >
              ML Labs
            </Button>
            <Button 
              onClick={() => navigate('/projects')}
              variant="outline-info" 
              className="hero-btn-secondary"
            >
              Web Apps
            </Button>
          </div>
        </div>
      </Container>
    </div>
  );
}