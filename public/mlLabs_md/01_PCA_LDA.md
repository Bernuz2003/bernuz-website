### Analisi della Riduzione Dimensionale: PCA vs. LDA

L'obiettivo di questo studio è analizzare un dataset di accessi biometrici per valutare la separabilità tra le classi "Genuine" (classe 1) e "Fake" (classe 0). Vengono confrontate due tecniche di riduzione dimensionale, Principal Component Analysis (PCA) e Linear Discriminant Analysis (LDA), sia come strumenti di analisi esplorativa sia come pre-processing per un classificatore.

#### Dataset e Preprocessing

Il dataset è composto da **6000 campioni** con **6 feature** ciascuno, equamente bilanciato tra le due classi (2990 campioni classe 0, 3010 campioni classe 1). Per la valutazione delle performance, il dataset viene suddiviso con proporzione 2/3 per il training (4000 campioni) e 1/3 per la validation (2000 campioni).

```python
def split_dataset_train_eval(data, labels, seed=0):
    """Divide il dataset in training e validation set (2/3 - 1/3)."""
    train_fraction = int(data.shape[1] * 2/3)
    np.random.seed(seed)
    indices = np.random.permutation(data.shape[1])
    train_indices = indices[:train_fraction]
    eval_indices = indices[train_fraction:]

    data_train = data[:, train_indices]
    labels_train = labels[train_indices]
    data_eval = data[:, eval_indices]
    labels_eval = labels[eval_indices]
    
    return (data_train, labels_train), (data_eval, labels_eval)
```

#### Analisi Esplorativa con Principal Component Analysis (PCA)

La PCA è una tecnica non supervisionata che trasforma i dati in un nuovo sistema di coordinate (le Componenti Principali) per massimizzare la varianza. L'obiettivo è proiettare i dati in uno spazio a dimensionalità inferiore preservando la maggior quantità di informazione possibile.

**Fondamenti Matematici della PCA**

L'obiettivo della PCA può essere formulato come la ricerca di una proiezione che minimizza l'errore quadratico medio di ricostruzione. Dato un set di dati $x_n$, se $\tilde{x}_n$ è la sua versione ricostruita dopo la proiezione in uno spazio a dimensionalità inferiore, la PCA cerca di minimizzare la funzione di costo:

$$ J = \frac{1}{N} \sum_{n=1}^{N} \|x_n - \tilde{x}_n\|^2 $$

Per un dataset centrato con matrice di covarianza $\mathbf{C} = \frac{1}{N}\mathbf{X}\mathbf{X}^T$, dove $\mathbf{X}$ è la matrice dei dati centrati, la soluzione ottimale è data dalla decomposizione agli autovalori:

$$ \mathbf{C} = \mathbf{U}\mathbf{\Lambda}\mathbf{U}^T $$

dove $\mathbf{U}$ contiene gli autovettori (componenti principali) e $\mathbf{\Lambda}$ è una matrice diagonale con gli autovalori ordinati in modo decrescente.

```python
def compute_mean_and_covariance(data):
    """Calcola media e matrice di covarianza."""
    mean_vec = as_column(data.mean(axis=1))
    centered = data - mean_vec
    cov_matrix = centered @ centered.T / data.shape[1]
    return mean_vec, cov_matrix

def compute_PCA(data, m):
    """Calcola la matrice di proiezione PCA con *m* componenti principali."""
    _, cov_matrix = compute_mean_and_covariance(data)
    U, s, _ = np.linalg.svd(cov_matrix)
    
    # Calcola varianza spiegata
    explained_variance_ratio = s / np.sum(s) * 100
    print(f"PCA {m}D: varianza spiegata = {np.sum(explained_variance_ratio[:m]):.1f}%")
    
    return U[:, :m]
```

**Analisi della Varianza Spiegata**

La distribuzione della varianza tra le componenti principali rivela informazioni cruciali sulla struttura del dataset:

- **PC1**: Spiega il 23.8% della varianza totale
- **PC1-PC2**: Insieme spiegano il 41.1% della varianza
- **PC1-PC3**: Insieme spiegano il 57.9% della varianza
- **PC1-PC4**: Insieme spiegano il 74.6% della varianza
- **PC1-PC5**: Insieme spiegano il 90.6% della varianza

Questa distribuzione suggerisce che le prime due componenti catturano meno della metà della varianza totale, indicando che l'informazione discriminante potrebbe essere distribuita su multiple dimensioni.

![](/mlLabs_screens/01_DR/PCA_Full_histograms-1.png)

L'analisi degli istogrammi delle componenti principali mostra come le distribuzioni delle due classi si proiettano su ciascuna delle nuove feature. Si osserva che le prime due componenti, PC1 e PC2, mostrano una certa separazione tra le medie delle classi "Genuine" e "Fake". Le componenti successive (da PC3 a PC6) presentano invece una sovrapposizione molto maggiore, confermando che l'informazione discriminante è concentrata principalmente nelle prime dimensioni.

![](/mlLabs_screens/01_DR/PCA_Full_scatter_matrix-1.png)

La scatter matrix conferma questa osservazione, mostrando che la combinazione di PC1 e PC2 offre la separazione visiva più netta tra i punti delle due classi.

#### Analisi Discriminante con Linear Discriminant Analysis (LDA)

A differenza della PCA, l'LDA è una tecnica supervisionata che proietta i dati in uno spazio a dimensionalità inferiore con l'obiettivo esplicito di massimizzare la separabilità tra le classi.

**Fondamenti Matematici della LDA**

L'LDA persegue il suo obiettivo massimizzando il rapporto tra la varianza *tra le classi* e la varianza *all'interno delle classi*. Questo si formalizza attraverso due matricri di scatter:

1. **Matrice di scatter between-class** ($\mathbf{S}_b$):
$$ \mathbf{S}_b = \sum_{i=1}^{C} N_i (\boldsymbol{\mu}_i - \boldsymbol{\mu})(\boldsymbol{\mu}_i - \boldsymbol{\mu})^T $$

2. **Matrice di scatter within-class** ($\mathbf{S}_w$):
$$ \mathbf{S}_w = \sum_{i=1}^{C} \sum_{\mathbf{x} \in C_i} (\mathbf{x} - \boldsymbol{\mu}_i)(\mathbf{x} - \boldsymbol{\mu}_i)^T $$

dove $\boldsymbol{\mu}_i$ è la media della classe $i$, $\boldsymbol{\mu}$ è la media globale, $N_i$ è il numero di campioni nella classe $i$.

L'LDA trova la direzione di proiezione $\mathbf{w}$ che massimizza il criterio di Fisher:

$$ J(\mathbf{w}) = \frac{\mathbf{w}^T \mathbf{S}_b \mathbf{w}}{\mathbf{w}^T \mathbf{S}_w \mathbf{w}} $$

La soluzione $\mathbf{w}$ è l'autovettore associato al massimo autovalore del problema agli autovalori generalizzato:

$$ \mathbf{S}_b \mathbf{w} = \lambda \mathbf{S}_w \mathbf{w} $$

```python
def compute_scatter_matrices(data, labels):
    """Calcola le matrici di scatter between-class (Sb) e within-class (Sw)."""
    Sb, Sw = 0, 0
    mu_global = as_column(data.mean(axis=1))
    
    for c in [0, 1]:
        class_data = data[:, labels == c]
        mu_class = as_column(class_data.mean(axis=1))
        
        # Between-class scatter
        diff = mu_class - mu_global
        Sb += class_data.shape[1] * (diff @ diff.T)
        
        # Within-class scatter
        Sw += (class_data - mu_class) @ (class_data - mu_class).T
    
    return Sb / data.shape[1], Sw / data.shape[1]

def compute_LDA(data, labels):
    """Calcola la direzione discriminante LDA per classificazione binaria."""
    Sb, Sw = compute_scatter_matrices(data, labels)
    eigenvalues, eigenvectors = scipy.linalg.eigh(Sb, Sw)
    
    # Prende l'autovettore con autovalore massimo
    w = eigenvectors[:, -1:]
    
    print(f"LDA: autovalore massimo = {eigenvalues[-1]:.3f}")
    
    return w
```

**Interpretazione dell'Autovalore Massimo**

L'autovalore massimo ottenuto è **1.683** sull'intero dataset e **1.636** sul training set. Questo valore rappresenta il rapporto tra la varianza between-class e within-class lungo la direzione discriminante ottimale. Un valore maggiore di 1 indica che la varianza tra le classi è superiore alla varianza all'interno delle classi, suggerendo una buona capacità discriminante.

![](/mlLabs_screens/01_DR/LDA_Full_histogram-1.png)

L'istogramma della proiezione LDA mostra chiaramente due distribuzioni gaussiane separate con una zona di sovrapposizione limitata, confermando l'efficacia della tecnica nel separare le due classi.

### Valutazione delle Performance di Classificazione

**Classificatore LDA con Ottimizzazione della Soglia**

Il classificatore LDA utilizza una soglia per separare le due classi nella proiezione 1D. La soglia di default è calcolata come:

$$ \text{threshold} = \frac{1}{2}(\mu_0 + \mu_1) $$

dove $\mu_0$ e $\mu_1$ sono le medie delle proiezioni delle due classi.

```python
def lda_classify(data_train, labels_train, data_eval, threshold=None):
    """Classifica usando LDA con soglia specificata o ottimale."""
    # Calcola LDA
    w = compute_LDA(data_train, labels_train)
    
    # Proiezione
    proj_train = project_LDA(data_train, w)
    proj_eval = project_LDA(data_eval, w)
    
    # Orienta genuine (classe 1) a destra
    mean_0 = proj_train[0, labels_train == 0].mean()
    mean_1 = proj_train[0, labels_train == 1].mean()
    
    if mean_1 < mean_0:
        w = -w
        proj_train = project_LDA(data_train, w)
        proj_eval = project_LDA(data_eval, w)
        mean_0 = proj_train[0, labels_train == 0].mean()
        mean_1 = proj_train[0, labels_train == 1].mean()
    
    # Soglia di decisione
    if threshold is None:
        threshold = 0.5 * (mean_0 + mean_1)
    
    # Classificazione
    predictions = (proj_eval.ravel() >= threshold).astype(np.int32)
    
    return predictions, proj_eval, threshold
```

**Risultati della Classificazione LDA**

- **LDA Baseline**: Accuratezza del **90.7%** (errore 9.3%)
- **Soglia di default**: -0.0185
- **Soglia ottimale**: -0.1412 (ottenuta mediante ricerca esaustiva)
- **Miglioramento con soglia ottimale**: 0.0% (nessun miglioramento significativo)

La mancanza di miglioramento con l'ottimizzazione della soglia suggerisce che la soglia di default basata sulle medie delle classi è già ottimale o molto vicina all'ottimo.

**Pipeline PCA + LDA: Analisi Dettagliata**

La combinazione di PCA come preprocessing seguito da LDA per la classificazione mostra risultati interessanti:

| Dimensioni PCA | Varianza Spiegata | Autovalore LDA | Errore | Accuratezza |
|----------------|-------------------|----------------|---------|-------------|
| 5D | 90.6% | 1.636 | 9.3% | **90.7%** |
| 4D | 74.6% | 1.635 | 9.2% | **90.8%** |
| 3D | 57.9% | 1.634 | 9.2% | **90.8%** |
| 2D | 41.1% | 1.627 | 9.2% | **90.8%** |
| 1D | 23.8% | 1.623 | 9.3% | **90.7%** |

**Interpretazione dei Risultati**

1. **Plateau di Performance**: Le configurazioni con 2, 3 e 4 componenti PCA raggiungono tutte l'accuratezza massima del 90.8%, suggerendo che l'informazione discriminante è contenuta principalmente nelle prime 2-4 componenti principali.

2. **Effetto di Denoising**: Il leggero miglioramento (0.1%) rispetto alla baseline LDA indica che la PCA agisce come un filtro di denoising, rimuovendo componenti meno informative che potrebbero introdurre rumore nella classificazione.

3. **Degradazione con 1D**: La riduzione a una sola componente principale causa una perdita di performance, indicando che almeno due dimensioni sono necessarie per mantenere l'informazione discriminante.

4. **Stabilità degli Autovalori**: Gli autovalori LDA rimangono stabili (1.636 → 1.623) anche con riduzione dimensionale drastica, confermando che la struttura discriminante è preservata.

**Analisi della Complessità Computazionale**

La riduzione da 6 a 2 dimensioni comporta:
- **Riduzione dello spazio feature**: 67% in meno di parametri
- **Velocità di classificazione**: Miglioramento teorico proporzionale alla riduzione dimensionale
- **Robustezza**: Minore rischio di overfitting con meno parametri

### Conclusioni e Considerazioni Teoriche

L'analisi dimostra che entrambe le tecniche sono efficaci, con risultati che forniscono insights importanti sulla natura del dataset:

1. **Complementarità delle Tecniche**: La PCA identifica le direzioni di massima varianza, mentre l'LDA trova la direzione di massima separabilità. Il fatto che la combinazione migliori leggermente le performance suggerisce che esiste rumore nelle componenti meno significative.

2. **Struttura Intrinseca del Dataset**: La distribuzione della varianza spiegata (solo 41.1% nelle prime due componenti) indica che il dataset ha una struttura complessa, ma l'informazione discriminante è comunque concentrata in poche dimensioni.

3. **Efficienza del Modello**: Ottenere performance equivalenti o superiori con solo 2-4 feature anziché 6 costituisce un vantaggio significativo per applicazioni real-time e riduce il rischio di overfitting.

4. **Robustezza della Classificazione**: La stabilità delle performance attraverso diverse riduzioni dimensionali (2D-4D) suggerisce che il modello è robusto e non dipende criticamente da feature specifiche.

La **configurazione ottimale** identificata è **PCA(4D) + LDA** con un'accuratezza del 90.8%, che rappresenta il miglior compromesso tra performance, efficienza computazionale e robustezza del modello.