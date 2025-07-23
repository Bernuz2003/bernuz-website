import { projects } from '../data/projects';
import ProjectCard from './ProjectCard';
import './../styles/Projects.css';

interface ProjectsGridProps {
  limit?: number;
}

export default function ProjectsGrid({ limit }: ProjectsGridProps) {
  const projectsToShow = limit ? projects.slice(0, limit) : projects;

  return (
    <div className="projects-grid">
      {projectsToShow.map(p => (
        <div key={p.slug} className="projects-grid-item">
          <ProjectCard project={p} />
        </div>
      ))}
    </div>
  );
}