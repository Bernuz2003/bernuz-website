import { useParams, Link, useNavigate } from 'react-router-dom';
import { useEffect, useState } from 'react';
import Section from '../components/Section';
import { mlLabs } from '../data/mlLabs';
import { Spinner, Button } from 'react-bootstrap';
import Markdown from '../components/Markdown';

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

  return (
    <Section title={lab.title} lead="" className='page-header-section'>

      {/* tag tech */}
      <div className="mb-4 d-flex gap-2 flex-wrap">
        {lab.tags.map(t => (
          <span key={t} className="badge-tech badge rounded-pill px-3 py-2">
            {t}
          </span>
        ))}
      </div>

      {!md ? (
        <Spinner animation="border" variant="info" />
      ) : (
        <Markdown source={md} />
      )}

      <div className="d-flex gap-3 flex-wrap mt-4">
        <Button onClick={() => navigate('/ml-labs')} variant="outline-light">
          ← Indietro
        </Button>
      </div>
    </Section>
  );
}
