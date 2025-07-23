export interface Project {
  title: string;
  slug: string;
  description: string;
  tech: string[];
  demoUrl: string;
  repoUrl: string;
  image?: string;
  longDescription?: string;
  readmePath?: string;
}

export const projects: Project[] = [

  {
    title: 'Pathfinding Visualizer',
    slug: 'pathfinding-visualizer',
    description: 'Visualizzatore interattivo dei principali algoritmi di pathfinding su griglia.',
    tech: ['React', 'Vite', 'TypeScript', 'CSS Modules'],
    demoUrl: 'https://pathfining-visualizer.vercel.app',
    repoUrl: 'https://github.com/Bernuz2003/pathfinding-visualizer',
    image: '/proj_screens/pathfinding-visualizer.png',
    longDescription: 'Strumento educativo per esplorare il funzionamento degli algoritmi di ricerca come A*, Dijkstra, BFS, DFS e Greedy. Permette di definire punto di partenza, arrivo e ostacoli, visualizzando il processo passo-passo su una griglia dinamica e responsive.',
    readmePath: '/projects_md/pathfinding-visualizer.md'
  },
  {
    title: 'Sortify – Sorting Visualizer',
    slug: 'sortify',
    description: 'Visualizzatore interattivo di algoritmi (bubble, quick, merge ecc.) con animazioni controllabili.',
    tech: ['React', 'Vite', 'TypeScript', 'CSS'],
    demoUrl: 'https://sortify-sorting-algorithm-visualize.vercel.app',
    repoUrl: 'https://github.com/Bernuz2003/Sortify-Sorting-Algorithm-Visualizer',
    image: '/proj_screens/sortify.png',
    longDescription: 'Permette di confrontare algoritmi di ordinamento modificando dimensione array, velocità, distribuzione dei valori e osservando passo per passo.',
    readmePath: '/projects_md/sortify.md'
  },
  {
    title: 'Gioco della Sfortuna',
    slug: 'gioco-della-sfortuna',
    description: 'Mini‑gioco ironico basato su probabilità e scelte sfavorevoli.',
    tech: ['HTML', 'CSS', 'JavaScript'],
    demoUrl: 'https://gioco-della-sfortuna.vercel.app',
    repoUrl: 'https://github.com/Bernuz2003/Gioco_della_Sfortuna',
    image: '/proj_screens/gioco-sfortuna.png',
    longDescription:
      'Single‑page web‑game che mette alla prova la tua “fortuna”: posiziona carte sfortunate in ordine corretto prima che scada il tempo. Supporta modalità demo anonima e partite complete per utenti loggati.',
    readmePath: '/projects_md/gioco-della-sfortuna.md'
  },
  {
    title: 'Il Pistrellista',
    slug: 'il-pistrellista',
    description: 'Applicazione interattiva per layout e pattern di piastrellatura configurabili.',
    tech: ['React', 'Vite', 'TypeScript', 'Tailwind'],
    demoUrl: 'https://il-piastrellista.vercel.app',
    repoUrl: 'https://github.com/Bernuz2003/Il-Piastrellista',
    image: '/proj_screens/il-piastrellista.png',
    longDescription: 'Strumento sperimentale per generare, modificare e visualizzare pattern di piastrellatura con parametri regolabili. Focus su UI rapida e render efficiente.',
    readmePath: '/projects_md/il-piastrellista.md'
  }
];

export function getProjectBySlug(slug: string): Project | undefined {
  return projects.find(p => p.slug === slug);
}