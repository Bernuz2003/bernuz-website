export type WorkDomain =
  | "Data Engineering"
  | "Machine Learning"
  | "Deep Learning"
  | "Operating Systems"
  | "Web Applications"
  | "Algorithms"
  | "Research";

export type WorkType =
  | "Case Study"
  | "Academic Project"
  | "Interactive Demo"
  | "Research Experiment"
  | "System Prototype"
  | "Web App";

export type WorkStatus = "completed" | "in-progress" | "archived";

export interface WorkItem {
  title: string;
  slug: string;
  domain: WorkDomain;
  type: WorkType;
  year: string;
  status: WorkStatus;
  featured: boolean;
  description: string;
  longDescription?: string;
  tech: string[];
  image?: string;
  repoUrl?: string;
  demoUrl?: string;
  readmePath?: string;
}

export const workItems: WorkItem[] = [
  // Featured project from user prompt
  {
    title: "Urban Environmental Monitoring",
    slug: "urban-environmental-monitoring",
    domain: "Data Engineering",
    type: "Case Study",
    year: "2024",
    status: "completed",
    featured: true,
    description: "Polyglot data architecture for city-scale environmental monitoring.",
    longDescription: "Polyglot architecture for real-time and historical urban environmental monitoring, combining time-series telemetry, spatial context, alerting and Grafana dashboards.",
    tech: ["InfluxDB", "PostgreSQL/PostGIS", "Kafka", "Telegraf", "Grafana", "Docker"],
    readmePath: 'projects_md/urban-environmental-monitoring.md'
  },

  // From projects.ts
  {
    title: 'Pathfinding Visualizer',
    slug: 'pathfinding-visualizer',
    domain: 'Algorithms',
    type: 'Interactive Demo',
    year: '2023',
    status: 'completed',
    featured: false,
    description: 'Visualizzatore interattivo dei principali algoritmi di pathfinding su griglia.',
    tech: ['React', 'Vite', 'TypeScript', 'CSS Modules'],
    demoUrl: 'https://pathfining-visualizer.vercel.app',
    repoUrl: 'https://github.com/Bernuz2003/Pathfining-Visualizer.git',
    image: '/proj_screens/Pathfinding_visualizer.png',
    longDescription: 'Strumento educativo per esplorare il funzionamento degli algoritmi di ricerca come A*, Dijkstra, BFS, DFS e Greedy. Permette di definire punto di partenza, arrivo e ostacoli, visualizzando il processo passo-passo su una griglia dinamica e responsive.',
    readmePath: '/projects_md/pathfinding-visualizer.md'
  },
  {
    title: 'Sortify – Sorting Visualizer',
    slug: 'sortify',
    domain: 'Algorithms',
    type: 'Interactive Demo',
    year: '2023',
    status: 'completed',
    featured: false,
    description: 'Visualizzatore interattivo di algoritmi (bubble, quick, merge ecc.) con animazioni controllabili.',
    tech: ['React', 'Vite', 'TypeScript', 'CSS'],
    demoUrl: 'https://sortify-sorting-algorithm-visualize.vercel.app',
    repoUrl: 'https://github.com/Bernuz2003/Sortify-Sorting-Algorithm-Visualizer',
    image: '/proj_screens/Sortify.png',
    longDescription: 'Permette di confrontare algoritmi di ordinamento modificando dimensione array, velocità, distribuzione dei valori e osservando passo per passo.',
    readmePath: '/projects_md/sortify.md'
  },
  {
    title: 'Gioco della Sfortuna',
    slug: 'gioco-della-sfortuna',
    domain: 'Web Applications',
    type: 'Web App',
    year: '2023',
    status: 'completed',
    featured: false,
    description: 'Mini‑gioco ironico basato su probabilità e scelte sfavorevoli.',
    tech: ['HTML', 'CSS', 'JavaScript'],
    demoUrl: 'https://gioco-della-sfortuna.vercel.app',
    repoUrl: 'https://github.com/Bernuz2003/Gioco_della_Sfortuna',
    image: '/proj_screens/Gioco_della_sfortuna.png',
    longDescription: 'Single‑page web‑game che mette alla prova la tua “fortuna”: posiziona carte sfortunate in ordine corretto prima che scada il tempo. Supporta modalità demo anonima e partite complete per utenti loggati.',
    readmePath: '/projects_md/gioco-della-sfortuna.md'
  },
  {
    title: 'Il Piastrellista',
    slug: 'il-piastrellista',
    domain: 'Web Applications',
    type: 'Web App',
    year: '2023',
    status: 'completed',
    featured: false,
    description: 'Applicazione interattiva per layout e pattern di piastrellatura configurabili.',
    tech: ['React', 'Vite', 'TypeScript', 'Tailwind'],
    demoUrl: 'https://il-piastrellista.vercel.app',
    repoUrl: 'https://github.com/Bernuz2003/Il-Piastrellista',
    image: '/proj_screens/Il_piastrellista.png',
    longDescription: 'Strumento sperimentale per generare, modificare e visualizzare pattern di piastrellatura con parametri regolabili. Focus su UI rapida e render efficiente.',
    readmePath: '/projects_md/il-piastrellista.md'
  },

  // From mlLabs.ts
  {
    title: 'Dimensionality Reduction',
    slug: 'dimensionality-reduction',
    domain: 'Machine Learning',
    type: 'Academic Project',
    year: '2024',
    status: 'completed',
    featured: false,
    description: "Confronto tra PCA e LDA che dimostra come ridurre da 6 a 4 dimensioni mantenendo il 90.8% di accuratezza nella classificazione biometrica.",
    image: '/mlLabs_logos/PCA_LDA_logo.png',
    tech: ['PCA', 'LDA', 'Feature Extraction', 'SVD'],
    readmePath: '/mlLabs_md/01_PCA_LDA.md',
    repoUrl: 'https://github.com/Bernuz2003/MachineLearning-Labs/tree/main/01_Dimensionality%20Reduction'
  },
  {
    title: 'Univariate Density Estimation',
    slug: 'univariate-density-estimation',
    domain: 'Machine Learning',
    type: 'Academic Project',
    year: '2024',
    status: 'completed',
    featured: false,
    description: "Modellazione Gaussiana feature-by-feature tramite MLE: scopri come media e varianza rivelano i pattern discriminanti nascosti nei dati biometrici",
    image: '/mlLabs_logos/density_esitmation.png',
    tech: ['MLE', 'Gaussian Distribution', 'Density Estimation', 'Univariate'],
    readmePath: '/mlLabs_md/02_Density_estimation.md',
    repoUrl: 'https://github.com/Bernuz2003/MachineLearning-Labs/tree/main/02_Gaussian%20Density%20Estimation'
  },
  {
    title: 'Multivariate Gaussian Classifiers',
    slug: 'multivariate-gaussian-classifiers',
    domain: 'Machine Learning',
    type: 'Academic Project',
    year: '2024',
    status: 'completed',
    featured: false,
    description: "Classificatori generativi basati su Bayes: scopri come la scelta della matrice di covarianza (Full, Naive, o Tied) sveli la vera struttura dei dati e determini il modello vincente.",
    image: '/mlLabs_logos/MVG_logo.png',
    tech: ['MVG', 'Naive Bayes', 'Tied Covariance', 'Generative Models'],
    readmePath: '/mlLabs_md/03_MVG.md',
    repoUrl: 'https://github.com/Bernuz2003/MachineLearning-Labs/tree/main/03_Generative%20Gaussian%20Model'
  },
  {
    title: 'Bayes Decision & Risk Analysis',
    slug: 'bayes-decision-risk-analysis',
    domain: 'Machine Learning',
    type: 'Academic Project',
    year: '2024',
    status: 'completed',
    featured: false,
    description: 'Applicazione della Teoria delle Decisioni di Bayes per valutare i classificatori in base al rischio (DCF), dimostrando come la scelta del modello ottimale dipenda dal contesto applicativo.',
    image: '/mlLabs_logos/Bayes_Decision_logo.png',
    tech: ['Bayes Risk', 'DCF', 'minDCF', 'Model Evaluation'],
    readmePath: '/mlLabs_md/04_BDM.md',
    repoUrl: 'https://github.com/Bernuz2003/MachineLearning-Labs/tree/main/04_Bayes%20Decision%20Model'
  },
  {
    title: 'Linear & Quadratic Logistic Regression',
    slug: 'logistic-regression',
    domain: 'Machine Learning',
    type: 'Academic Project',
    year: '2024',
    status: 'completed',
    featured: false,
    description: 'Come l\'espansione quadratica delle feature nella Regolazione Logistica svela relazioni non lineari, migliorando drasticamente le performance di classificazione.',
    image: '/mlLabs_logos/LR_logo.png',
    tech: ['Logistic Regression', 'Regularization', 'Quadratic Features', 'Discriminative Models'],
    readmePath: '/mlLabs_md/05_LR.md',
    repoUrl: 'https://github.com/Bernuz2003/MachineLearning-Labs/tree/main/05_Logistic%20Regression'
  },
  {
    title: 'Support Vector Machines',
    slug: 'support-vector-machines',
    domain: 'Machine Learning',
    type: 'Academic Project',
    year: '2024',
    status: 'completed',
    featured: false,
    description: 'Dalla ricerca del margine massimo lineare alla potenza dei kernel (Polinomiale, RBF), raggiungendo performance SOTA a scapito della calibrazione dei punteggi.',
    image: '/mlLabs_logos/SVM_logo.png',
    tech: ['SVM', 'Kernel Trick', 'RBF Kernel', 'Max-Margin'],
    readmePath: '/mlLabs_md/06_SVM.md',
    repoUrl: 'https://github.com/Bernuz2003/MachineLearning-Labs/tree/main/06_Support%20Vector%20Machine'
  },
  {
    title: 'Gaussian Mixture Models',
    slug: 'gaussian-mixture-models',
    domain: 'Machine Learning',
    type: 'Academic Project',
    year: '2024',
    status: 'completed',
    featured: false,
    description: 'Superare i limiti della singola Gaussiana con i GMM per modellare distribuzioni complesse, scoprendo come un modello a covarianza diagonale possa diventare il migliore.',
    image: '/mlLabs_logos/GMM_logo.png',
    tech: ['GMM', 'EM Algorithm', 'Model Complexity', 'Density Estimation'],
    readmePath: '/mlLabs_md/07_GMM.md',
    repoUrl: 'https://github.com/Bernuz2003/MachineLearning-Labs/tree/main/07_Gaussian%20Mixture%20Models'
  },
  {
    title: 'Model Calibration & Fusion',
    slug: 'calibration-fusion',
    domain: 'Machine Learning',
    type: 'Academic Project',
    year: '2024',
    status: 'completed',
    featured: false,
    description: 'L\'atto finale dell\'ottimizzazione: la calibrazione corregge i punteggi dei modelli e la fusione di sistemi eterogenei crea un classificatore finale più robusto di ogni singolo componente.',
    image: '/mlLabs_logos/Model_Calibration_logo.png',
    tech: ['Calibration', 'Fusion', 'Ensemble', 'Model Stacking'],
    readmePath: '/mlLabs_md/08_CF.md',
    repoUrl: 'https://github.com/Bernuz2003/MachineLearning-Labs/tree/main/08_Calibration%20Fusion'
  }
];

export function getWorkItemBySlug(slug: string): WorkItem | undefined {
  return workItems.find(w => w.slug === slug);
}

export function getFeaturedWorkItems(): WorkItem[] {
  return workItems.filter(w => w.featured);
}
