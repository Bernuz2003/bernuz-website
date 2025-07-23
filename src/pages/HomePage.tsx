import { FaCode, FaDesktop, FaNetworkWired, FaDatabase, FaGlobe, FaMicrochip, FaRobot, FaProjectDiagram, FaShieldAlt, FaLanguage, FaTools, FaCogs, FaServer, FaCloud } from 'react-icons/fa';
import '../styles/HomePage.css';
import Hero from '../components/Hero';
import Section from '../components/Section';
import ProjectsGrid from '../components/ProjectsGrid';
import { Link } from 'react-router-dom';
import MLGrid from '../components/MLGrid';

export default function HomePage() {
  const competences = [
    { icon: <FaCode />, title: 'Programmazione', detail: 'C, C++, Java, Python, Rust' },
    { icon: <FaDesktop />, title: 'Sistemi Operativi', detail: 'OS161, Linux, Windows' },
    { icon: <FaNetworkWired />, title: 'Reti di Calcolatori', detail: 'Protocolli, configurazione, sicurezza' },
    { icon: <FaDatabase />, title: 'Basi di Dati', detail: 'SQL, PostgreSQL, MySQL' },
    { icon: <FaGlobe />, title: 'Sviluppo Web', detail: 'HTML, CSS, JavaScript, Bootstrap' },
    { icon: <FaMicrochip />, title: 'Microcontrollori', detail: 'Arduino, Raspberry Pi, LandTiger' },
    { icon: <FaRobot />, title: 'AI & Machine Learning', detail: 'Supervised ML, deep learning, big data' },
    { icon: <FaProjectDiagram />, title: 'Algoritmi & Strutture Dati', detail: 'Ordinamento, ricerca, heap, grafi' },
    { icon: <FaTools />, title: 'DevOps & CI/CD', detail: 'Docker, GitHub Actions, pipeline automation' },
    { icon: <FaCogs />, title: 'Software Engineering', detail: 'Design Patterns, metodologie Agile' },
    { icon: <FaServer />, title: 'Kernel Internals', detail: 'Scheduling, memory management su OS/161' },
    { icon: <FaCloud />, title: 'Cloud Basics', detail: 'AWS EC2/S3, GCP' },
    { icon: <FaShieldAlt />, title: 'Cybersecurity', detail: 'Principi di sicurezza, mitigazioni' },
    { icon: <FaLanguage />, title: 'Lingue', detail: 'Italiano (madrelingua), Inglese (C1)' }
  ];

  return (
    <>
      <Hero />
      <Section id="about" title="Chi sono" lead="">
        <p className="text-secondary">Studente di Ingegneria Informatica al Politecnico di Torino con una formazione completa che spazia dai sistemi a basso livello all'intelligenza artificiale, mantenendo sempre vivo l'interesse per l'innovazione tecnologica.</p>
        <p className="text-secondary">Questo sito raccoglie progetti e sperimentazioni nate durante il mio percorso accademico e personale. Il corso di Ingegneria Informatica mi ha permesso di esplorare l'informatica a 360°: dalla programmazione di sistema e reti alla sicurezza informatica, dai database ai microcontrollori, dall'analisi degli algoritmi al machine learning.</p>
        <p className="text-secondary">Questa formazione trasversale mi ha dato una visione d'insieme che applico nei miei progetti, spaziando tra diverse tecnologie e domini. Attualmente sto concentrando maggiori energie sul Machine Learning e AI, ma rimango sempre curioso verso ogni aspetto dell'informatica moderna.</p>

        {/* Competenze grid */}
        <h3 className="h6 text-uppercase text-info mt-5 mb-3">Competenze principali</h3>
        <div className="competences-grid">
          {competences.map((c, idx) => (
            <div key={idx} className="col">
              <div className="competence-card">
                <div className="competence-icon">{c.icon}</div>
                <div>
                  <strong>{c.title}:</strong> {c.detail}
                </div>
              </div>
            </div>
          ))}
        </div>
      </Section>

      <Section id="ml" title="Machine Learning Labs" lead="Approfondimenti tecnici su temi come dimensionality reduction, density estimation e altro."
        titleAction={
          <Link to="/ml-labs" className="btn btn-outline-info btn-sm">
            Vedi tutti &rarr;
          </Link>
        }
      >
        <MLGrid limit={3} />
      </Section>

      <Section id="projects" title="Progetti in evidenza" lead="Una selezione di progetti che rappresentano diversi aspetti della mia formazione, dal web development agli algoritmi, tutti con codice disponibile su GitHub."
        titleAction={
          <Link to="/projects" className="btn btn-outline-info btn-sm">
            Vedi tutti &rarr;
          </Link>
        }
      >
        <ProjectsGrid limit={3} />
      </Section>
    </>
  );
}
