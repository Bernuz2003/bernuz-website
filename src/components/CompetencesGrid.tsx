import type { Competence } from '../data/competences';

interface Props {
  competences: Competence[];
}

export default function CompetencesGrid({ competences }: Props) {
  return (
    <div className="competences-grid">
      {competences.map((comp, idx) => {
        const Icon = comp.icon;
        return (
          <div key={idx} className="competence-card glass-panel">
            <div className="competence-icon">
              <Icon />
            </div>
            <div>
              <span className="font-mono text-gradient fw-bold d-block">{comp.title}</span>
              <span className="text-secondary small">{comp.detail}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}