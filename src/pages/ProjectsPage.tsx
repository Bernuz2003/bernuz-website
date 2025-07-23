import Section from '../components/Section';
import ProjectsGrid from '../components/ProjectsGrid';

export default function ProjectsPage() {
  return (
    <Section 
      title="Tutti i Progetti" 
      lead="Una collezione completa di progetti accademici, esperimenti di programmazione e applicazioni web sviluppate durante il mio percorso di studi al Politecnico di Torino."
    >
      <ProjectsGrid />
    </Section>
  );
}