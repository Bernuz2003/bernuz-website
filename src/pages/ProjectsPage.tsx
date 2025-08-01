import Section from '../components/Section';
import ProjectsGrid from '../components/ProjectsGrid';

export default function ProjectsPage() {
  return (
    <Section 
      title="Web Applications" 
      lead=""
      className="page-header-section"
    >
      <ProjectsGrid />
    </Section>
  );
}