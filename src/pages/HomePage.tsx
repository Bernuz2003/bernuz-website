import React, { useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import emailjs from '@emailjs/browser';
import { FaCode, FaDesktop, FaNetworkWired, FaDatabase, FaGlobe, FaMicrochip, FaRobot, FaProjectDiagram, FaShieldAlt, FaLanguage, FaTools, FaCogs, FaServer, FaCloud } from 'react-icons/fa';
import '../styles/HomePage.css';
import Hero from '../components/Hero';
import Section from '../components/Section';
import ProjectsGrid from '../components/ProjectsGrid';
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

  const form = useRef<HTMLFormElement>(null);
  const [isSending, setIsSending] = useState(false);
  const [isSent, setIsSent] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendEmail = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!form.current) return;

    setIsSending(true);
    setError(null);
    setIsSent(false);

    // !!! SOSTITUISCI CON I TUOI VALORI DA EMAILJS !!!
    const serviceID = 'service_sjd3ku4';
    const templateID = 'template_1akquw8';
    const publicKey = '1_0E712SRg_gpF4w-';

    emailjs.sendForm(serviceID, templateID, form.current, publicKey)
      .then(() => {
        setIsSent(true);
        form.current?.reset();
      }, (error) => {
        console.error('FAILED...', error);
        setError('Impossibile inviare il messaggio. Riprova più tardi.');
      })
      .finally(() => {
        setIsSending(false);
      });
  };

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

      <Section id="ml" title="Machine Learning Labs" lead=""
        titleAction={
          <Link to="/ml-labs" className="btn btn-outline-info btn-sm">
            Vedi tutti &rarr;
          </Link>
        }
      >
        <p className="text-secondary">La sezione Machine Learning Labs presenta un percorso didattico completo attraverso 8 laboratori pratici che coprono l'intero spettro del machine learning moderno: dai fondamenti statistici e classificatori gaussiani (Lab 1-3), ai modelli discriminativi avanzati come regressione logistica e SVM (Lab 4-6), fino ai sofisticati Gaussian Mixture Models (Lab 7) e alle tecniche di calibrazione e fusione per l'ottimizzazione finale (Lab 8). Ogni laboratorio combina teoria rigorosa, implementazione pratica e analisi sperimentale approfondita su dataset reali, offrendo una formazione completa dal machine learning teorico alle applicazioni industriali moderne.</p>
        <p className="text-secondary">
          <strong>Codice Sorgente Labs:</strong>{" "}
          <a
            href="https://github.com/Bernuz2003/MachineLearning-Labs"
            target="_blank"
            rel="noopener noreferrer"
            className="text-info text-decoration-none"
          >GitHub</a>
        </p>
        <MLGrid limit={3} />
      </Section>

      <Section id="projects" title="Web Applications" lead=""
        titleAction={
          <Link to="/projects" className="btn btn-outline-info btn-sm">
            Vedi tutti &rarr;
          </Link>
        }
      >
        <p className="text-secondary">Esplora una collezione di applicazioni web interattive progettate per rendere tangibili concetti algoritmici complessi. Dai visualizzatori di algoritmi di ordinamento che mostrano in tempo reale il comportamento di QuickSort e MergeSort, ai simulatori di pathfinding che illustrano A* e Dijkstra in azione, fino a semplici giochi educativi che combinano divertimento e apprendimento informatico.</p>

        <ProjectsGrid limit={3} />
      </Section>

      <Section id="contact" title="Contattami" lead="">
        <p className="text-secondary">Hai un progetto in mente o vuoi semplicemente salutarmi? Scrivi qui sotto.</p>
        <div className="contact-form-container">
          <form ref={form} onSubmit={sendEmail} className="contact-form">
            <div className="form-group">
              <label htmlFor="user_name">Nome</label>
              <input type="text" name="user_name" id="user_name" className="form-control" required />
            </div>
            <div className="form-group">
              <label htmlFor="user_email">Email</label>
              <input type="email" name="user_email" id="user_email" className="form-control" required />
            </div>
            <div className="form-group">
              <label htmlFor="message">Messaggio</label>
              <textarea name="message" id="message" rows={5} className="form-control" required></textarea>
            </div>
            <button type="submit" className="btn btn-info text-dark fw-semibold" disabled={isSending}>
              {isSending ? 'Invio in corso...' : 'Invia Messaggio'}
            </button>
          </form>
          {isSent && <p className="text-success mt-3">Messaggio inviato con successo!</p>}
          {error && <p className="text-danger mt-3">{error}</p>}
        </div>
      </Section>
    </>
  );
}
