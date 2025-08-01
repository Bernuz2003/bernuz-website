export interface MLLab {
  title: string;
  slug: string;
  teaser: string;
  image?: string;
  tags: string[];
  mdPath: string;
  repoUrl?: string;
}

export const mlLabs: MLLab[] = [
  {
    title: 'Dimensionality Reduction',
    slug: 'dimensionality-reduction',
    teaser: "Confronto tra PCA e LDA che dimostra come ridurre da 6 a 4 dimensioni mantenendo il 90.8% di accuratezza nella classificazione biometrica.",
    image: '/mlLabs_logos/PCA_LDA_logo.png',
    tags: ['PCA', 'LDA', 'Feature Extraction', 'SVD'],
    mdPath: '/mlLabs_md/01_PCA_LDA.md',
    repoUrl: 'https://github.com/Bernuz2003/MachineLearning-Labs/tree/main/01_Dimensionality%20Reduction'

  },
  {
    title: 'Univariate Density Estimation',
    slug: 'univariate-density-estimation',
    teaser: "Modellazione Gaussiana feature-by-feature tramite MLE: scopri come media e varianza rivelano i pattern discriminanti nascosti nei dati biometrici",
    image: '/mlLabs_logos/density_esitmation.png',
    tags: ['MLE', 'Gaussian Distribution', 'Density Estimation', 'Univariate'],
    mdPath: '/mlLabs_md/02_Density_estimation.md',
    repoUrl: 'https://github.com/Bernuz2003/MachineLearning-Labs/tree/main/02_Gaussian%20Density%20Estimation'

  },
  {
    title: 'Multivariate Gaussian Classifiers',
    slug: 'multivariate-gaussian-classifiers',
    teaser: "Classificatori generativi basati su Bayes: scopri come la scelta della matrice di covarianza (Full, Naive, o Tied) sveli la vera struttura dei dati e determini il modello vincente.",
    image: '/mlLabs_logos/MVG_logo.png',
    tags: ['MVG', 'Naive Bayes', 'Tied Covariance', 'Generative Models'],
    mdPath: '/mlLabs_md/03_MVG.md',
    repoUrl: 'https://github.com/Bernuz2003/MachineLearning-Labs/tree/main/03_Generative%20Gaussian%20Model'

  },
  {
    title: 'Bayes Decision & Risk Analysis',
    slug: 'bayes-decision-risk-analysis',
    teaser: 'Applicazione della Teoria delle Decisioni di Bayes per valutare i classificatori in base al rischio (DCF), dimostrando come la scelta del modello ottimale dipenda dal contesto applicativo.',
    image: '/mlLabs_logos/Bayes_Decision_logo.png',
    tags: ['Bayes Risk', 'DCF', 'minDCF', 'Model Evaluation'],
    mdPath: '/mlLabs_md/04_BDM.md',
    repoUrl: 'https://github.com/Bernuz2003/MachineLearning-Labs/tree/main/04_Bayes%20Decision%20Model'

  },
  {
    title: 'Linear & Quadratic Logistic Regression',
    slug: 'logistic-regression',
    teaser: 'Come l\'espansione quadratica delle feature nella Regolazione Logistica svela relazioni non lineari, migliorando drasticamente le performance di classificazione.',
    image: '/mlLabs_logos/LR_logo.png',
    tags: ['Logistic Regression', 'Regularization', 'Quadratic Features', 'Discriminative Models'],
    mdPath: '/mlLabs_md/05_LR.md',
    repoUrl: 'https://github.com/Bernuz2003/MachineLearning-Labs/tree/main/05_Logistic%20Regression'

  },
  {
    title: 'Support Vector Machines',
    slug: 'support-vector-machines',
    teaser: 'Dalla ricerca del margine massimo lineare alla potenza dei kernel (Polinomiale, RBF), raggiungendo performance SOTA a scapito della calibrazione dei punteggi.',
    image: '/mlLabs_logos/SVM_logo.png',
    tags: ['SVM', 'Kernel Trick', 'RBF Kernel', 'Max-Margin'],
    mdPath: '/mlLabs_md/06_SVM.md',
    repoUrl: 'https://github.com/Bernuz2003/MachineLearning-Labs/tree/main/06_Support%20Vector%20Machine'

  },
  {
    title: 'Gaussian Mixture Models',
    slug: 'gaussian-mixture-models',
    teaser: 'Superare i limiti della singola Gaussiana con i GMM per modellare distribuzioni complesse, scoprendo come un modello a covarianza diagonale possa diventare il migliore.',
    image: '/mlLabs_logos/GMM_logo.png',
    tags: ['GMM', 'EM Algorithm', 'Model Complexity', 'Density Estimation'],
    mdPath: '/mlLabs_md/07_GMM.md',
    repoUrl: 'https://github.com/Bernuz2003/MachineLearning-Labs/tree/main/07_Gaussian%20Mixture%20Models'

  },
  {
    title: 'Model Calibration & Fusion',
    slug: 'calibration-fusion',
    teaser: 'L\'atto finale dell\'ottimizzazione: la calibrazione corregge i punteggi dei modelli e la fusione di sistemi eterogenei crea un classificatore finale più robusto di ogni singolo componente.',
    image: '/mlLabs_logos/Model_Calibration_logo.png',
    tags: ['Calibration', 'Fusion', 'Ensemble', 'Model Stacking'],
    mdPath: '/mlLabs_md/08_CF.md',
    repoUrl: 'https://github.com/Bernuz2003/MachineLearning-Labs/tree/main/08_Calibration%20Fusion'

  }
];

