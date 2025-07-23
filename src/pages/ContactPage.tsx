import Section from '../components/Section';

export default function ContactPage(){
  return (
    <Section title="Contatti" lead="Per collaborazioni, domande o opportunità.">
      <ul className="list-unstyled text-secondary small" style={{maxWidth:'55ch'}}>
        <li><strong>Email:</strong> <a href="mailto:email@example.com" className="text-info">email@example.com</a></li>
        <li><strong>GitHub:</strong> <a href="https://github.com/Bernuz2003" className="text-info" target="_blank" rel="noopener noreferrer">github.com/Bernuz2003</a></li>
        <li><strong>LinkedIn:</strong> <a href="https://www.linkedin.com" className="text-info" target="_blank" rel="noopener noreferrer">LinkedIn</a></li>
      </ul>
      <p className="text-secondary small">Sostituire i placeholder con i riferimenti reali; aggiungere un form (EmailJS / Formspree) se necessario.</p>
    </Section>
  );
}
