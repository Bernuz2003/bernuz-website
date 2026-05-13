import { useParams, Link, useNavigate } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { mlLabs } from '../data/mlLabs';
import Section from '../components/Section';
import { Button, Spinner } from 'react-bootstrap';
import Markdown from '../components/Markdown';
import TypographicList from '../components/TypographicList';
import type { TypographicItem } from '../components/TypographicList';

export default function MLLabDetailPage() {
  const { slug } = useParams();
  const navigate = useNavigate();
  const lab = mlLabs.find(l => l.slug === (slug || ''));
  const [md, setMd] = useState<string | null>(null);

  useEffect(() => {
    if (lab?.mdPath) {
      fetch(lab.mdPath)
        .then(r => r.text())
        .then(setMd)
        .catch(() => setMd('*File non disponibile*'));
    }
  }, [lab]);

  if (!lab) {
    return (
      <Section title="Argomento non trovato">
        <p className="text-secondary">
          Non trovo questo laboratorio ML.{' '}
          <Link to="/" className="text-info">
            Torna alla home
          </Link>
          .
        </p>
      </Section>
    );
  }

  const mlItems: TypographicItem[] = mlLabs.map(l => ({
    title: l.title,
    slug: l.slug,
    link: `/ml-labs/${l.slug}`
  }));

  return (
    <Section title="Machine Learning Labs" lead="Esplora i laboratori pratici completi sui classificatori statistici e discriminativi." className='page-header-section'>
      <div className="row mt-5">
        <div className="col-lg-4 pe-lg-4 d-none d-lg-block" style={{ borderRight: '1px solid rgba(255,255,255,0.05)' }}>
          <h4 className="h6 text-secondary text-uppercase mb-4 font-mono">Index</h4>
          <TypographicList items={mlItems} />
        </div>
        
        <div className="col-lg-8 ps-lg-5 mt-4 mt-lg-0">
          <Button onClick={() => navigate('/ml-labs')} variant="link" className="text-secondary p-0 mb-4 font-mono d-lg-none text-decoration-none">
            ← Torna all'elenco
          </Button>

          <h2 className="text-info mb-4">{lab.title}</h2>
          
          <div className="mb-4 d-flex gap-2 flex-wrap">
            {lab.tags.map(t => (
              <span key={t} className="badge bg-transparent border border-secondary text-light font-mono px-3 py-2">
                {t}
              </span>
            ))}
          </div>

          <div className="d-flex gap-3 flex-wrap mt-4 mb-5 pb-4 border-bottom border-secondary">
            <Button href={lab.repoUrl} target="_blank" variant="outline-light" className="font-mono">
              Codice Sorgente
            </Button>
            <Button href="/mlLabs_data/Appunti_ML.pdf" download="Appunti_ML.pdf" variant="outline-info" className="font-mono">
              Appunti Teorici
            </Button>
          </div>

          {lab.mdPath && (
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
    </Section>
  );
}
