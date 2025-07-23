import { Card, Button } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import type { Project } from '../data/projects';
import './../styles/Projects.css';

export default function ProjectCard({ project }: { project: Project }) {
  return (
    <Card className="project-card">
      {project.image && (
        <div className="project-image-container">
          <img src={project.image} alt={project.title} className="project-image" />
        </div>
      )}
      <Card.Body className="project-card-body">
        <Card.Title className="project-card-title">{project.title}</Card.Title>
        <Card.Text className="project-card-description">{project.description}</Card.Text>
        <div className="project-tech-container">
          {project.tech.map(t => (
            <span key={t} className="project-tech-badge">{t}</span>
          ))}
        </div>
        <div className="project-buttons">
          <Button 
            href={project.demoUrl} 
            target="_blank" 
            rel="noopener noreferrer" 
            size="sm" 
            variant="info" 
            className="project-btn-demo"
          >
            Demo
          </Button>
          <Button 
            href={project.repoUrl} 
            target="_blank" 
            rel="noopener noreferrer" 
            size="sm" 
            variant="outline-info"
          >
            Codice
          </Button>
          <Link to={`/projects/${project.slug}`} className="btn btn-outline-light btn-sm">
            Dettagli
          </Link>
        </div>
      </Card.Body>
    </Card>
  );
}
