import React, { useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import emailjs from '@emailjs/browser';
import Hero from '../components/Hero';
import Section from '../components/Section';
import TypographicList from '../components/TypographicList';
import type { TypographicItem } from '../components/TypographicList';
import ScrollReveal from '../components/ScrollReveal';
import { projects } from '../data/projects';
import { mlLabs } from '../data/mlLabs';
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

    emailjs.sendForm('service_sjd3ku4', 'template_1akquw8', form.current, '1_0E712SRg_gpF4w-')
      .then(() => {
        setIsSent(true);
        form.current?.reset();
      }, (error) => {
        console.error('FAILED...', error);
        setError('Impossibile inviare il messaggio.');
      })
      .finally(() => setIsSending(false));
  };

  const projectItems: TypographicItem[] = projects.map(p => ({
    title: p.title,
    slug: p.slug,
    description: p.description,
    techStack: p.techStack,
    link: `/projects/${p.slug}`
  }));

  const mlItems: TypographicItem[] = mlLabs.map(l => ({
    title: l.title,
    slug: l.slug,
    description: l.teaser || l.description,
    techStack: l.tags,
    link: `/ml-labs/${l.slug}`
  }));

  return (
    <>
      <Hero />
      
      <Section id="about" title="Chi sono" lead="">
        <ScrollReveal>
          <div className="mx-auto" style={{ maxWidth: '800px', fontSize: '1.1rem', lineHeight: '1.8' }}>
            <p className="text-secondary">Sono uno studente di Ingegneria Informatica al Politecnico di Torino, affascinato dalla capacità del software di modellare e risolvere problemi complessi.</p>
            <p className="text-secondary mb-5">Questo spazio è il mio <strong>workspace digitale</strong>: esplora i laboratori di Machine Learning per vedere i miei lavori di classificazione, o le Web Apps per algoritmi interattivi. La combinazione di solide basi teoriche e interfacce moderne è il cuore di ogni mio progetto.</p>
          </div>
        </ScrollReveal>
        
        <ScrollReveal delay={0.2}>
          <div className="mx-auto" style={{ maxWidth: '800px' }}>
            <h3 className="h6 text-uppercase text-info mt-5 mb-4 font-mono">Tech Stack</h3>
            <div className="d-flex flex-wrap gap-2">
              {competences.map((comp, idx) => (
                <span key={idx} className="badge bg-transparent border border-secondary text-light font-mono px-3 py-2">
                  {comp.title}
                </span>
              ))}
            </div>
          </div>
        </ScrollReveal>
      </Section>

      <Section 
        id="selected-works" 
        title="Selected Works" 
        lead=""
        titleAction={
          <div className="d-flex gap-3">
            <Link to="/ml-labs" className="btn btn-outline-secondary btn-sm font-mono">ML Labs &rarr;</Link>
            <Link to="/projects" className="btn btn-outline-info btn-sm font-mono">Web Apps &rarr;</Link>
          </div>
        }
      >
        <div className="mt-4">
          <h3 className="h5 text-secondary mb-3 font-mono">/ Machine Learning</h3>
          <TypographicList items={mlItems} limit={2} />
          
          <h3 className="h5 text-secondary mt-5 mb-3 font-mono">/ Web Applications</h3>
          <TypographicList items={projectItems} limit={2} />
        </div>
      </Section>

      <Section id="contact" title="Contattami" lead="">
        <div className="mx-auto" style={{ maxWidth: '600px' }}>
          <p className="text-secondary mb-4 text-center">Hai un progetto in mente o vuoi semplicemente salutarmi? Usa il form qui sotto o usa lo script <code className="text-info">contact.sh</code>.</p>
          
          <div className="glass-panel p-4 p-md-5">
            <form ref={form} onSubmit={sendEmail}>
              <div className="mb-4">
                <label className="form-label text-secondary font-mono small">NOME</label>
                <input type="text" name="user_name" className="form-control bg-transparent text-light border-secondary" required />
              </div>
              <div className="mb-4">
                <label className="form-label text-secondary font-mono small">EMAIL</label>
                <input type="email" name="user_email" className="form-control bg-transparent text-light border-secondary" required />
              </div>
              <div className="mb-4">
                <label className="form-label text-secondary font-mono small">MESSAGGIO</label>
                <textarea name="message" rows={4} className="form-control bg-transparent text-light border-secondary" required></textarea>
              </div>
              <button type="submit" className="btn btn-info w-100 fw-bold" disabled={isSending}>
                {isSending ? 'Invio in corso...' : 'Invia Messaggio'}
              </button>
            </form>
            {isSent && <p className="text-success mt-3 text-center font-mono">&gt; [OK] Messaggio inviato con successo.</p>}
            {error && <p className="text-danger mt-3 text-center font-mono">&gt; [ERROR] {error}</p>}
          </div>
        </div>
      </Section>
    </>
  );
}
