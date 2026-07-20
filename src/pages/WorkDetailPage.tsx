import { useParams, Link, useNavigate } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { getWorkItemBySlug, workItems } from '../data/workItems';
import { Container, Button, Spinner, Badge } from 'react-bootstrap';
import Markdown from '../components/Markdown';
import TypographicList from '../components/TypographicList';
import type { TypographicItem } from '../components/TypographicList';
import Section from '../components/Section';

export default function WorkDetailPage() {
  const { slug } = useParams();
  const navigate = useNavigate();
  const item = getWorkItemBySlug(slug || '');
  const [md, setMd] = useState<string | null>(null);

  useEffect(() => {
    if (item?.readmePath) {
      fetch(item.readmePath)
        .then(res => res.text())
        .then(text => setMd(text))
        .catch(() => setMd('*Contenuto non disponibile*'));
    }
  }, [item]);

  if (!item)
    return (
      <Section title="Progetto non trovato">
        <p className="text-secondary">
          Il lavoro richiesto non esiste.{' '}
          <Link to="/work" className="text-info">Torna al portfolio</Link>.
        </p>
      </Section>
    );

  const relatedItems = workItems
    .filter(w => w.domain === item.domain)
    .map(p => ({
      title: p.title,
      slug: p.slug,
      link: `/work/${p.slug}`
    } as TypographicItem));

  return (
    <Container className="pt-4 pb-5">
      <div className="row">
        <div className="col-lg-4 pe-lg-4 d-none d-lg-block" style={{ borderRight: '1px solid rgba(255,255,255,0.05)' }}>
          <h4 className="h6 text-secondary text-uppercase mb-4 font-mono">Related in {item.domain}</h4>
          <TypographicList items={relatedItems} />
        </div>
        
        <div className="col-lg-8 ps-lg-5 mt-4 mt-lg-0">
          <Button onClick={() => navigate('/work')} variant="link" className="text-secondary p-0 mb-4 font-mono d-lg-none text-decoration-none">
            ← Torna al portfolio
          </Button>
          
          <div className="mb-3 d-flex flex-wrap gap-2">
            <Badge bg="transparent" className="border border-info text-info font-mono">{item.domain}</Badge>
            <Badge bg="transparent" className="border border-secondary text-secondary font-mono">{item.type}</Badge>
            {item.featured && <Badge bg="info" className="text-dark font-mono">Featured Case Study</Badge>}
          </div>

          <h2 className="text-light mb-2">{item.title}</h2>
          <p className="text-secondary lead mb-4">{item.longDescription || item.description}</p>
          
          <div className="mb-4 d-flex gap-2 flex-wrap">
            {item.tech.map(t => (
              <span key={t} className="badge bg-transparent border border-secondary text-light font-mono px-3 py-2">
                {t}
              </span>
            ))}
          </div>

          <div className="d-flex gap-3 flex-wrap mt-4 mb-5 pb-4 border-bottom border-secondary">
            {item.demoUrl && (
              <Button href={item.demoUrl} target="_blank" variant="info" className="text-dark fw-bold font-mono">
                Live Demo
              </Button>
            )}
            {item.repoUrl && (
              <Button href={item.repoUrl} target="_blank" variant="outline-light" className="font-mono">
                Codice Sorgente
              </Button>
            )}
          </div>

          {item.readmePath && (
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
