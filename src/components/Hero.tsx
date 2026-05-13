import { Container, Button } from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';
import { useState, useEffect } from 'react';
import './../styles/Hero.css';

export default function Hero() {
  const navigate = useNavigate();
  const [crtMode, setCrtMode] = useState(false);

  useEffect(() => {
    const syncState = () => setCrtMode(document.body.classList.contains('crt-mode'));
    window.addEventListener('crtToggled', syncState);
    syncState(); // initial sync

    return () => window.removeEventListener('crtToggled', syncState);
  }, []);

  const handleCrtToggle = () => {
    document.body.classList.toggle('crt-mode');
    window.dispatchEvent(new Event('crtToggled'));
  };

  return (
    <div className="hero-gradient hero-section">
      <Container className="hero-container">
        <div className="col-lg-10 col-xl-9 mx-auto">
          <div className="hero-content glass-panel p-4 p-md-5">
            <div className="terminal-header mb-3">
              <span className="terminal-dot close"></span>
              <span className="terminal-dot minimize"></span>
              <span className="terminal-dot maximize"></span>
              <span className="terminal-title font-mono text-secondary">~/portfolio/welcome.sh</span>
            </div>
            <h1 className="hero-title font-mono">
              <span className="text-secondary">$</span> echo <span className="text-gradient">"Ciao, sono Emanuele"</span>
              <span className="cursor-blink">_</span>
            </h1>
            <h2 className="h4 mt-3 text-secondary fw-normal">
              Esploro tecnologie, algoritmi e soluzioni innovative.
            </h2>
            <p className="hero-text mt-4">
              Hub personale per progetti accademici, esperimenti di programmazione, visualizzazioni interattive e applicazioni web. Dal machine learning ai sorting visualizer, un mix di soluzioni pratiche e creatività tecnica.
            </p>
            <div className="hero-buttons mt-5">
              <Button 
                onClick={() => navigate('/ml-labs')}
                variant="info" 
                className="btn-glow-primary"
              >
                ML Labs
              </Button>
              <Button 
                onClick={() => navigate('/projects')}
                variant="outline-info" 
                className="btn-glow-secondary"
              >
                Web Apps
              </Button>
            </div>
            
            <div className="mt-5 pt-3 font-mono" style={{ fontSize: '0.85rem', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
              <span className="text-secondary">$</span> ./apply_filter.sh --retro-crt={' '}
              <span 
                style={{ cursor: 'pointer' }} 
                className={crtMode ? 'text-success' : 'text-danger'}
                onClick={handleCrtToggle}
                title="Attiva/Disattiva effetto CRT vintage"
              >
                {crtMode ? 'ON' : 'OFF'}
              </span>
            </div>
          </div>
        </div>
      </Container>
    </div>
  );
}