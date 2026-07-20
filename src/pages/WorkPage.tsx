import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Section from '../components/Section';
import TypographicList from '../components/TypographicList';
import type { TypographicItem } from '../components/TypographicList';
import { workItems, getFeaturedWorkItems } from '../data/workItems';
import type { WorkDomain } from '../data/workItems';
import ScrollReveal from '../components/ScrollReveal';
import { Badge, Button } from 'react-bootstrap';

const DOMAINS: ("All" | WorkDomain)[] = [
  "All",
  "Data Engineering",
  "Machine Learning",
  "Deep Learning",
  "Operating Systems",
  "Web Applications",
  "Algorithms",
  "Research"
];

export default function WorkPage() {
  const [activeFilter, setActiveFilter] = useState<"All" | WorkDomain>("All");
  const navigate = useNavigate();

  const featured = getFeaturedWorkItems();
  
  const filteredItems = workItems.filter(item => {
    if (activeFilter === "All") return true;
    return item.domain === activeFilter;
  });

  const mapToTypographic = (items: typeof workItems): TypographicItem[] => 
    items.map(p => ({
      title: p.title,
      slug: p.slug,
      description: p.description,
      techStack: p.tech,
      link: `/work/${p.slug}`
    }));

  return (
    <div className="page-header-section">
      <Section title="Featured Case Studies" lead="I progetti più maturi e complessi.">
        <div className="row g-4 mt-2">
          {featured.map((item, idx) => (
            <div className="col-12" key={item.slug}>
              <ScrollReveal delay={idx * 0.1}>
                <div 
                  className="glass-panel p-4 p-lg-5 cursor-pointer card-glow d-flex flex-column flex-lg-row gap-4 align-items-lg-center"
                  onClick={() => navigate(`/work/${item.slug}`)}
                >
                  <div className="flex-grow-1">
                    <div className="d-flex flex-wrap gap-2 mb-3">
                      <Badge bg="transparent" className="border border-info text-info font-mono">{item.domain}</Badge>
                      <Badge bg="transparent" className="border border-secondary text-secondary font-mono">{item.type}</Badge>
                    </div>
                    <h3 className="text-light mb-2">{item.title}</h3>
                    <p className="text-secondary mb-4">{item.longDescription || item.description}</p>
                    <div className="d-flex flex-wrap gap-2 mb-4 mb-lg-0">
                      {item.tech.map(t => (
                        <span key={t} className="badge bg-transparent border border-secondary text-light font-mono px-2 py-1">
                          {t}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="d-flex flex-column gap-2 flex-shrink-0" style={{ minWidth: '150px' }}>
                     <Button variant="outline-info" className="w-100 font-mono">Case Study &rarr;</Button>
                  </div>
                </div>
              </ScrollReveal>
            </div>
          ))}
        </div>
      </Section>

      <Section title="All Work" lead="Esplora il resto del portfolio navigando per dominio." id="all-work">
        <div className="mb-5 d-flex flex-wrap gap-2">
          {DOMAINS.map(domain => {
            const isActive = activeFilter === domain;
            return (
              <button
                key={domain}
                onClick={() => setActiveFilter(domain)}
                className={`btn btn-sm font-mono ${isActive ? 'btn-info text-dark fw-bold' : 'btn-outline-secondary text-light'}`}
                style={{ borderRadius: '20px' }}
              >
                {domain}
              </button>
            )
          })}
        </div>

        <div className="row mt-4">
          <div className="col-lg-5 pe-lg-4" style={{ borderRight: '1px solid rgba(255,255,255,0.05)' }}>
            <TypographicList items={mapToTypographic(filteredItems)} />
            {filteredItems.length === 0 && (
              <p className="text-secondary mt-4 font-mono">Nessun progetto trovato per questo dominio.</p>
            )}
          </div>
          <div className="col-lg-7 d-none d-lg-flex align-items-center justify-content-center">
            <p className="text-secondary font-mono opacity-50">&lt; Seleziona un progetto dalla lista</p>
          </div>
        </div>
      </Section>
    </div>
  );
}
