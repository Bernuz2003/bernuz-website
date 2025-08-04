import type { IconType } from 'react-icons';
import {
  FaGlobe,
  FaMicrochip,
  FaRobot,
  FaProjectDiagram,
  FaTools,
  FaCogs,
  FaServer,
  FaCloud,
  FaShieldAlt,
  FaLanguage,
} from 'react-icons/fa';

export interface Competence {
  icon: IconType;
  title: string;
  detail: string;
}

export const competences: Competence[] = [
  { icon: FaGlobe, title: 'Sviluppo Web', detail: 'HTML, CSS, JavaScript, Bootstrap' },
  { icon: FaMicrochip, title: 'Microcontrollori', detail: 'Arduino, Raspberry Pi, LandTiger' },
  { icon: FaRobot, title: 'AI & Machine Learning', detail: 'Supervised ML, deep learning, big data' },
  { icon: FaProjectDiagram, title: 'Algoritmi & Strutture Dati', detail: 'Ordinamento, ricerca, heap, grafi' },
  { icon: FaTools, title: 'DevOps & CI/CD', detail: 'Docker, GitHub Actions, pipeline automation' },
  { icon: FaCogs, title: 'Software Engineering', detail: 'Design Patterns, metodologie Agile' },
  { icon: FaServer, title: 'Kernel Internals', detail: 'Scheduling, memory management su OS/161' },
  { icon: FaCloud, title: 'Cloud Basics', detail: 'AWS EC2/S3, GCP' },
  { icon: FaShieldAlt, title: 'Cybersecurity', detail: 'Principi di sicurezza, mitigazioni' },
  { icon: FaLanguage, title: 'Lingue', detail: 'Italiano, Inglese (C1)' },
];