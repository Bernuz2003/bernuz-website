import Section from '../components/Section';
import SkillBadge from '../components/SkillBadge';

const skills = ['TypeScript','React','Next.js','Vite','Tailwind','Bootstrap','Node.js','Express','Git','Docker','SQL Basics','Machine Learning Basics'];
const interests = ['Visualizzazione algoritmi','Performance','UI minimale','Tooling','Automazione'];

export default function AboutPage(){
  return (
    <Section title="Chi sono" lead="Approfondimento sul mio percorso, filosofia di sviluppo e obiettivi.">
      <div className="row g-4">
        <div className="col-lg-7">
          <p className="text-secondary">Sono uno studente di Ingegneria Informatica al Politecnico di Torino. Uso il codice come mezzo per esplorare concetti teorici traducendoli in interfacce interattive. Prediligo stack leggeri (Vite, React) e un approccio incrementale: feature piccole, refactoring frequenti, attenzione alla DX.</p>
          <p className="text-secondary">Obiettivi a breve termine: consolidare pattern full‑stack moderni, migliorare nelle metriche di performance e contribuire a progetti open‑source. A lungo termine vorrei lavorare su prodotti developer‑centric.</p>
        </div>
        <div className="col-lg-5">
          <h3 className="h6 text-info text-uppercase">Competenze</h3>
          <div className="mb-3">{skills.map(s=> <SkillBadge key={s} label={s} />)}</div>
          <h3 className="h6 text-info text-uppercase mt-3">Interessi</h3>
          <div>{interests.map(i=> <SkillBadge key={i} label={i} />)}</div>
        </div>
      </div>
    </Section>
  );
}