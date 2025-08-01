import { Row, Col, Card, Button } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import { mlLabs } from '../data/mlLabs.ts';
import '../styles/MLGrid.css';

interface MLGridProps {
  limit?: number;
}

export default function MLGrid({ limit }: MLGridProps) {
  const labsToShow = limit ? mlLabs.slice(0, limit) : mlLabs;

  return (
    <Row className="g-4">
      {labsToShow.map(lab => (
        <Col key={lab.slug} xs={12} md={6} lg={4}>
          <Card className="ml-lab-card">
            {lab.image && (
              <div className="ml-lab-image-container">
                <img src={lab.image} alt={lab.title} className="ml-lab-image" />
              </div>
            )}
            <Card.Body className="ml-lab-card-body">
              <Card.Title className="ml-lab-card-title">{lab.title}</Card.Title>
              <Card.Text as="p" className="ml-lab-card-teaser">{lab.teaser}</Card.Text>
              <div className="ml-lab-tags-container">
                {lab.tags.map(t => (
                  <span key={t} className="ml-lab-tag-badge">
                    {t}
                  </span>
                ))}
              </div>
              <div className="ml-lab-button-container">
                <Button
                  as={Link}
                  to={`/ml-labs/${lab.slug}`}
                  size="sm"
                  variant="info"
                  className="text-dark fw-semibold"
                >
                  Apri
                </Button>
                {lab.repoUrl && (
                  <Button
                    href={lab.repoUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    size="sm"
                    variant="outline-info"
                  >
                    Codice
                  </Button>
                )}
              </div>
            </Card.Body>
          </Card>
        </Col>
      ))}
    </Row>
  );
}
