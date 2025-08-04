import type { Competence } from '../data/competences';

interface Props {
  competences: Competence[];
}

export default function CompetencesGrid({ competences }: Props) {
  return (
    <div className="competences-grid">
      {competences.map((c, idx) => {
        const Icon = c.icon;
        return (
          <div key={idx} className="col">
            <div className="competence-card">
              <div className="competence-icon">
                <Icon aria-hidden="true" />
              </div>
              <div>
                <strong>{c.title}:</strong> {c.detail}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
