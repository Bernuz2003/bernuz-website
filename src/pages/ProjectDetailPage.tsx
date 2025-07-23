import { useParams, Link } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { getProjectBySlug } from '../data/projects';
import Section from '../components/Section';
import { Button, Spinner } from 'react-bootstrap';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export default function ProjectDetailPage() {
  const { slug } = useParams();
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
    <Section title={project.title} lead={project.description}>
      {/* tag tech */}
      <div className="mb-4 d-flex gap-2 flex-wrap">
        {project.tech.map(t => (
          <span key={t} className="badge-tech badge rounded-pill px-3 py-2">
            {t}
          </span>
        ))}
      </div>

      {/* immagine hero */}
      {project.image && (
        <div className="ratio ratio-16x9 mb-4 bg-secondary">
          <img
            src={project.image}
            alt={project.title}
            style={{ objectFit: 'cover' }}
          />
        </div>
      )}

      {/* introduzione breve */}
      {project.longDescription && (
        <p className="text-secondary" style={{ maxWidth: '70ch' }}>
          {project.longDescription}
        </p>
      )}

      {/* --- README RENDERED --- */}
      {project.readmePath && (
        <div className="mt-5">
          {!md ? (
            <Spinner animation="border" variant="info" />
          ) : (
            <article className="markdown-body bg-dark border border-secondary rounded-2 p-4">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{md}</ReactMarkdown>
            </article>
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
        <Button as={Link} to="/projects" variant="outline-light">
          ← Torna ai progetti
        </Button>
      </div>
    </Section>
  );
}
