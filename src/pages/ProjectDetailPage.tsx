import { useParams, Link, useNavigate } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { getProjectBySlug } from '../data/projects';
import Section from '../components/Section';
import { Button, Spinner } from 'react-bootstrap';
import Markdown from '../components/Markdown'; // Importa il componente Markdown

export default function ProjectDetailPage() {
  const { slug } = useParams();
  const navigate = useNavigate();
  const project = getProjectBySlug(slug || '');
  const [md, setMd] = useState<string | null>(null);

  // carica README se presente
  useEffect(() => {
    if (project?.readmePath) {
      fetch(project.readmePath)
        .then(res => res.text())
        .then(text => setMd(text))
        .catch(() => setMd('*README non disponibile*'));
    }
  }, [project]);

  if (!project)
    return (
      <Section title="Progetto non trovato">
        <p className="text-secondary">
          Il progetto richiesto non esiste.{' '}
          <Link to="/projects" className="text-info">Torna all'elenco</Link>.
        </p>
      </Section>
    );

  return (
    <Section title={project.title} lead="" className='page-header-section'>
      {/* tag tech */}
      <div className="mb-4 d-flex gap-2 flex-wrap">
        {project.tech.map(t => (
          <span key={t} className="badge-tech badge rounded-pill px-3 py-2">
            {t}
          </span>
        ))}
      </div>

      {/* --- README RENDERED --- */}
      {project.readmePath && (
        <div className="mt-5">
          {!md ? (
            <Spinner animation="border" variant="info" />
          ) : (
            <Markdown source={md} />
          )}
        </div>
      )}

      {/* bottoni utility */}
      <div className="d-flex gap-3 flex-wrap mt-4">
        <Button
          href={project.demoUrl}
          target="_blank"
          variant="info"
          className="text-dark fw-semibold"
        >
          Demo
        </Button>
        <Button href={project.repoUrl} target="_blank" variant="outline-info">
          Codice
        </Button>
        <Button onClick={() => navigate('/projects')} variant="outline-light">
          ← Indietro
        </Button>
      </div>
    </Section>
  );
}
