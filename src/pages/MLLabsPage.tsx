import Section from '../components/Section';
import MLGrid from '../components/MLGrid';

export default function MLLabsPage() {
  return (
    <Section 
      title="Tutti i Laboratori ML" 
      lead="Una collezione di analisi approfondite su concetti chiave del Machine Learning, dall'estrazione di feature alla valutazione e fusione di modelli complessi."
    >
      <MLGrid />
    </Section>
  );
}