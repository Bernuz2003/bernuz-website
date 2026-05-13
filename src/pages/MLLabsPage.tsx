import Section from '../components/Section';
import TypographicList from '../components/TypographicList';
import type { TypographicItem } from '../components/TypographicList';
import { mlLabs } from '../data/mlLabs';

export default function MLLabsPage() {
  const mlItems: TypographicItem[] = mlLabs.map(l => ({
    title: l.title,
    slug: l.slug,
    description: l.teaser,
    techStack: l.tags,
    link: `/ml-labs/${l.slug}`
  }));

  return (
    <Section
      title="Machine Learning Labs"
      lead="Esplora i laboratori pratici completi sui classificatori statistici e discriminativi."
      className="page-header-section"
    >
      <div className="row mt-5">
        <div className="col-lg-5 pe-lg-4" style={{ borderRight: '1px solid rgba(255,255,255,0.05)' }}>
          <TypographicList items={mlItems} />
        </div>
        <div className="col-lg-7 d-none d-lg-flex flex-column px-lg-5 mt-2">
          <p className="text-secondary font-mono opacity-50 mb-4">&lt; Seleziona un lab dalla lista</p>
          <div className="p-4 rounded border border-secondary" style={{ backgroundColor: 'rgba(255,255,255,0.02)' }}>
            <h5 className="text-info font-mono mb-3">Informazioni Dataset</h5>
            <p className="text-secondary small text-start">Il dataset utilizzato rappresenta un problema di autenticazione biometrica per distinguere impronte digitali genuine (classe 1) da quelle falsificate (classe 0). È composto da 6000 campioni con 6 feature numeriche.</p>
            <a href="/mlLabs_data/trainData.csv" download="trainData.csv" className="btn btn-outline-secondary btn-sm font-mono mt-2">
              Download trainData.csv
            </a>
          </div>
        </div>
      </div>
    </Section>
  );
}