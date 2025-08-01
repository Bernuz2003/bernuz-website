import Section from '../components/Section';
import MLGrid from '../components/MLGrid';

export default function MLLabsPage() {
  return (
    <Section
      title="Machine Learning Labs"
      lead=""
      className="page-header-section"
    >
      <p className="text-secondary">Il dataset utilizzato in tutti i laboratori rappresenta un problema di autenticazione biometrica per distinguere impronte digitali genuine (classe 1) da quelle falsificate (classe 0), una sfida critica in applicazioni di sicurezza digitale. È composto da 6000 campioni con 6 feature numeriche estratte automaticamente dalle immagini attraverso algoritmi di image processing, che catturano proprietà geometriche e topologiche delle creste papillari come curvature, densità e punti di biforcazione.</p>
      <p className="text-secondary">
        <strong>Download Dataset:</strong>{" "}
        <a
          href="/mlLabs_data/trainData.csv"
          download="trainData.csv"
          className="text-info text-decoration-none"
        >trainData.csv</a>
      </p>

      <MLGrid />
    </Section>
  );
}