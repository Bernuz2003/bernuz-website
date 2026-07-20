import Section from '../components/Section';
import TypographicList from '../components/TypographicList';
import type { TypographicItem } from '../components/TypographicList';
import { projects } from '../data/projects';

export default function ProjectsPage() {
  const projectItems: TypographicItem[] = projects.map(p => ({
    title: p.title,
    slug: p.slug,
    description: p.description,
    techStack: p.tech,
    link: `/projects/${p.slug}`
  }));

  return (
    <Section 
      title="Web Applications" 
      lead="Applicazioni interattive e visualizzatori algoritmici."
      className="page-header-section"
    >
      <div className="row mt-5">
        <div className="col-lg-5 pe-lg-4" style={{ borderRight: '1px solid rgba(255,255,255,0.05)' }}>
          <TypographicList items={projectItems} />
        </div>
        <div className="col-lg-7 d-none d-lg-flex align-items-center justify-content-center">
          <p className="text-secondary font-mono opacity-50">&lt; Seleziona un progetto dalla lista</p>
        </div>
      </div>
    </Section>
  );
}