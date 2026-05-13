import { useParams, Link, useNavigate } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { getProjectBySlug, projects } from '../data/projects';
import { Container, Button, Spinner } from 'react-bootstrap';
import Markdown from '../components/Markdown';
import TypographicList from '../components/TypographicList';
import type { TypographicItem } from '../components/TypographicList';

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

  const projectItems: TypographicItem[] = projects.map(p => ({
    title: p.title,
    slug: p.slug,
    link: `/projects/${p.slug}`
  }));

  return (
    <Container className="pt-4 pb-5">
      <div className="row">
        <div className="col-lg-4 pe-lg-4 d-none d-lg-block" style={{ borderRight: '1px solid rgba(255,255,255,0.05)' }}>
          <h4 className="h6 text-secondary text-uppercase mb-4 font-mono">Index</h4>
          <TypographicList items={projectItems} />
        </div>
        
        <div className="col-lg-8 ps-lg-5 mt-4 mt-lg-0">
          <Button onClick={() => navigate('/projects')} variant="link" className="text-secondary p-0 mb-4 font-mono d-lg-none text-decoration-none">
            ← Torna all'elenco
          </Button>
          
          <h2 className="text-info mb-4">{project.title}</h2>
          
          <div className="mb-4 d-flex gap-2 flex-wrap">
            {project.tech.map(t => (
              <span key={t} className="badge bg-transparent border border-secondary text-light font-mono px-3 py-2">
                {t}
              </span>
            ))}
          </div>

          <div className="d-flex gap-3 flex-wrap mt-4 mb-5 pb-4 border-bottom border-secondary">
            <Button href={project.demoUrl} target="_blank" variant="info" className="text-dark fw-bold font-mono">
              Live Demo
            </Button>
            <Button href={project.repoUrl} target="_blank" variant="outline-light" className="font-mono">
              Sorgente
            </Button>
          </div>

          {project.readmePath && (
            <div className="markdown-body">
              {!md ? (
                <Spinner animation="border" variant="info" />
              ) : (
                <Markdown source={md} />
              )}
            </div>
          )}
        </div>
      </div>
    </Container>
  );
}
