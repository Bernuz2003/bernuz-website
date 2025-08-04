import React, { useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { Accordion } from 'react-bootstrap';
import emailjs from '@emailjs/browser';
import '../styles/HomePage.css';
import Hero from '../components/Hero';
import Section from '../components/Section';
import ProjectsGrid from '../components/ProjectsGrid';
import MLGrid from '../components/MLGrid';
import CompetencesGrid from '../components/CompetencesGrid';
import { competences } from '../data/competences';

export default function HomePage() {

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
        <p className="text-secondary">Sono uno studente di Ingegneria Informatica al Politecnico di Torino, affascinato dalla capacità del software di modellare e risolvere problemi complessi. La mia curiosità mi spinge a esplorare l'informatica in tutte le sue forme, dai meccanismi interni di un sistema operativo fino alle architetture di intelligenza artificiale.</p>
        <p className="text-secondary">Questo sito è il mio taccuino digitale: una raccolta di progetti nati per dare una forma concreta a concetti teorici. Qui troverai esperimenti che spaziano dalla visualizzazione di algoritmi alla classificazione biometrica, riflettendo un percorso che unisce la programmazione di sistema, le reti e l'analisi dei dati.</p>
        <p className="text-secondary">Ogni progetto è un'opportunità per imparare e applicare una visione d'insieme. Sebbene il mio interesse attuale sia rivolto principalmente al Machine Learning, credo che le soluzioni più innovative nascano dall'intersezione di discipline diverse. Spero che questa raccolta possa essere uno spunto di ispirazione.</p>

        {/* Competenze grid */}
        <h3 className="h6 text-uppercase text-info mt-5 mb-3">Competenze principali</h3>
        <div className="d-md-none">
          <Accordion>
            <Accordion.Item eventKey="0">
              <Accordion.Header>Mostra competenze</Accordion.Header>
              <Accordion.Body>
                <CompetencesGrid competences={competences} />
              </Accordion.Body>
            </Accordion.Item>
          </Accordion>
        </div>
        <div className="d-none d-md-block">
          <CompetencesGrid competences={competences} />
        </div>
      </Section>

      <Section
        id="ml"
        title="Machine Learning Labs"
        lead=""
        className="home-carousel-section" // <-- Aggiungi questa classe
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

      <Section
        id="projects"
        title="Web Applications"
        lead=""
        className="home-carousel-section" // <-- Aggiungi questa classe
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
