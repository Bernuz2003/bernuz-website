import { useParams, Link } from 'react-router-dom';
import { useEffect, useState } from 'react';
import Section from '../components/Section';
import { mlLabs } from '../data/mlLabs';
import { Spinner, Button } from 'react-bootstrap';
import Markdown from '../components/Markdown';

export default function MLLabDetailPage() {
  const { slug } = useParams();
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
    <Section title={lab.title} lead={lab.teaser}>
      {lab.image && (
        <div className="ratio ratio-16x9 mb-4 bg-secondary">
          <img src={lab.image} alt={lab.title} style={{ objectFit: 'cover' }} />
        </div>
      )}

      {!md ? (
        <Spinner animation="border" variant="info" />
      ) : (
        <Markdown source={md} />
      )}

      <div className="d-flex gap-3 flex-wrap mt-4">
        <Button as={Link} to="/" variant="outline-light">
          ← Torna alla home
        </Button>
      </div>
    </Section>
  );
}
