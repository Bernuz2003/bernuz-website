import type { IconType } from 'react-icons';
import { FaCode, FaDesktop, FaNetworkWired, FaDatabase, FaGlobe, FaMicrochip, FaRobot, FaProjectDiagram, FaShieldAlt, FaLanguage, FaTools, FaCogs, FaServer, FaCloud } from 'react-icons/fa';

export interface Competence {
  icon: IconType;
  title: string;
  detail: string;
}

export const competences: Competence[] = [
  { icon: FaCode, title: 'Programmazione', detail: 'C, C++, Java, Python, Rust' },
  { icon: FaDesktop, title: 'Sistemi Operativi', detail: 'OS161, Linux, Windows' },
  { icon: FaNetworkWired, title: 'Reti di Calcolatori', detail: 'Protocolli, configurazione, sicurezza' },
  { icon: FaDatabase, title: 'Basi di Dati', detail: 'SQL, PostgreSQL, MySQL' },
  { icon: FaGlobe, title: 'Sviluppo Web', detail: 'HTML, CSS, JavaScript, Bootstrap' },
  { icon: FaMicrochip, title: 'Microcontrollori', detail: 'Arduino, Raspberry Pi, LandTiger' },
  { icon: FaRobot, title: 'AI & Machine Learning', detail: 'Supervised ML, deep learning, big data' },
  { icon: FaProjectDiagram, title: 'Algoritmi & Strutture Dati', detail: 'Ordinamento, ricerca, heap, grafi' },
  { icon: FaTools, title: 'DevOps & CI/CD', detail: 'Docker, GitHub Actions, pipeline automation' },
  { icon: FaCogs, title: 'Software Engineering', detail: 'Design Patterns, metodologie Agile' },
  { icon: FaServer, title: 'Kernel Internals', detail: 'Scheduling, memory management su OS/161' },
  { icon: FaCloud, title: 'Cloud Basics', detail: 'AWS EC2/S3, GCP' },
  { icon: FaShieldAlt, title: 'Cybersecurity', detail: 'Principi di sicurezza, mitigazioni' },
  { icon: FaLanguage, title: 'Lingue', detail: 'Italiano (madrelingua), Inglese (C1)' },
];
