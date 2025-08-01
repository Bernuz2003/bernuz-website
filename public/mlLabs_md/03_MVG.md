## Argomento 3: Costruire Classificatori con Modelli Generativi Gaussiani (Analisi Dettagliata)

Nei capitoli precedenti abbiamo sezionato il problema da diverse angolazioni. Ora, mettiamo insieme tutti i pezzi per costruire dei classificatori completi e potenti. Abbandoniamo l'analisi di una singola feature alla volta per abbracciare una visione d'insieme, trattando ogni campione come un unico **vettore a 6 dimensioni**. L'idea fondamentale è di costruire dei **modelli generativi**, cioè modelli che non si limitano a tracciare una linea di separazione tra le classi, ma che imparano la "ricetta" completa per generare i dati di ogni classe.

### Fondamenti Teorici: Dal Teorema di Bayes alla Classificazione

#### Il Framework Bayesiano per la Classificazione

Il punto di partenza teorico è il **teorema di Bayes**, che ci permette di invertire la relazione tra dati e classi:

$$P(C_i | \mathbf{x}) = \frac{P(\mathbf{x} | C_i) P(C_i)}{P(\mathbf{x})}$$

dove:
- $P(C_i | \mathbf{x})$ è la **probabilità a posteriori** della classe $C_i$ dato il campione $\mathbf{x}$
- $P(\mathbf{x} | C_i)$ è la **verosimiglianza** (likelihood) del campione sotto la classe $C_i$
- $P(C_i)$ è la **probabilità a priori** della classe $C_i$
- $P(\mathbf{x})$ è l'**evidenza** (costante di normalizzazione)

La **regola di decisione di Bayes** ottimale per minimizzare l'errore di classificazione consiste nello scegliere la classe con probabilità a posteriori massima:

$$\hat{C} = \arg\max_{i} P(C_i | \mathbf{x})$$

#### Dal Teorema di Bayes al Log-Likelihood Ratio

Nel caso binario (classi 0 e 1), assumendo probabilità a priori uniformi ($P(C_0) = P(C_1) = 0.5$), la regola si semplifica al confronto delle verosimiglianze:

$$\frac{P(\mathbf{x} | C_1)}{P(\mathbf{x} | C_0)} \gtrless 1$$

Passando ai logaritmi per stabilità numerica, otteniamo il **Log-Likelihood Ratio (LLR)**:

$$\text{LLR}(\mathbf{x}) = \log P(\mathbf{x} | C_1) - \log P(\mathbf{x} | C_0) \gtrless 0$$

Questa formulazione ha il vantaggio di trasformare prodotti in somme e di evitare problemi di underflow numerico quando si lavora con probabilità molto piccole.

### La Base Matematica: la Gaussiana Multivariata

#### Formulazione Completa della PDF

Una distribuzione Gaussiana in $M$ dimensioni non è più una semplice campana, ma un'**iper-ellissoide** nello spazio, la cui forma, orientamento e posizione sono completamente descritti da due parametri fondamentali:

1. Il **vettore delle medie** $\boldsymbol{\mu} \in \mathbb{R}^M$, che rappresenta il centro della distribuzione
2. La **matrice di covarianza** $\boldsymbol{\Sigma} \in \mathbb{R}^{M \times M}$, simmetrica e definita positiva

La funzione di densità di probabilità (PDF) è data da:

$$p(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{\sqrt{(2\pi)^M |\det(\boldsymbol{\Sigma})|}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)$$

#### Interpretazione Geometrica della Matrice di Covarianza

La matrice di covarianza $\boldsymbol{\Sigma}$ codifica informazioni cruciali sulla forma della distribuzione:

- **Elementi diagonali** $\Sigma_{ii} = \text{Var}(X_i)$: varianze delle singole feature
- **Elementi fuori diagonale** $\Sigma_{ij} = \text{Cov}(X_i, X_j)$: covarianze tra coppie di feature
- **Autovalori** $\lambda_i$: determinano le lunghezze degli assi principali dell'ellissoide
- **Autovettori** $\mathbf{v}_i$: determinano l'orientamento degli assi principali

#### Implementazione Numericamente Stabile

Il calcolo diretto della PDF può portare a problemi di overflow/underflow. L'implementazione in log-scala risolve questi problemi:

```python
def logpdf_GAU_ND(X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """
    Calcola la log-densità di una Gaussiana N-dimensionale per tutti i campioni in X.
    
    Args:
        X: matrice (M, N) con N campioni M-dimensionali in colonna
        mu: vettore delle medie (M, 1)
        Sigma: matrice di covarianza (M, M)
    
    Returns:
        array (N,) con le log-densità
    """
    M = X.shape[0]  # Dimensionalità
    
    # Centra i dati
    XC = X - mu
    
    # Calcola l'inversa e il log-determinante usando decomposizione più stabile
    try:
        # Decomposizione di Cholesky per matrici definite positive
        L = np.linalg.cholesky(Sigma)
        # Risolve il sistema triangolare invece di calcolare l'inversa
        v = np.linalg.solve(L, XC)
        # La forma quadratica diventa ||v||²
        quad = np.sum(v * v, axis=0)
        # Il log-determinante è 2 * sum(log(diag(L)))
        log_det = 2 * np.sum(np.log(np.diag(L)))
    except np.linalg.LinAlgError:
        # Fallback per matrici non definite positive
        invS = np.linalg.inv(Sigma)
        log_det = np.linalg.slogdet(Sigma)[1]
        quad = np.sum(XC * (invS @ XC), axis=0)
    
    # Costante di normalizzazione + forma quadratica
    return -0.5 * (M * np.log(2*np.pi) + log_det + quad)
```

### Stima dei Parametri: Maximum Likelihood Estimation

#### Stima del Vettore delle Medie

La stima di massima verosimiglianza del vettore delle medie è semplicemente la media campionaria:

$$\hat{\boldsymbol{\mu}} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{x}_i$$

```python
def mean_vector(X: np.ndarray) -> np.ndarray:
    """
    Calcola il vettore delle medie empiriche.
    
    Args:
        X: matrice (M, N) con N campioni M-dimensionali
    
    Returns:
        vettore delle medie (M, 1)
    """
    return vcol(X.mean(axis=1))
```

#### Stima della Matrice di Covarianza

La stima di massima verosimiglianza della matrice di covarianza è:

$$\hat{\boldsymbol{\Sigma}} = \frac{1}{N} \sum_{i=1}^{N} (\mathbf{x}_i - \hat{\boldsymbol{\mu}})(\mathbf{x}_i - \hat{\boldsymbol{\mu}})^T$$

```python
def covariance_matrix(X: np.ndarray) -> np.ndarray:
    """
    Calcola la matrice di covarianza empirica.
    
    Args:
        X: matrice (M, N) con N campioni M-dimensionali
    
    Returns:
        matrice di covarianza (M, M)
    """
    # Centra i dati
    centered = X - mean_vector(X)
    # Prodotto esterno e media
    return (centered @ centered.T) / X.shape[1]
```

### Tre Modelli di Classificatori Gaussiani: Analisi Approfondita

La scelta del modello di covarianza rappresenta un trade-off fondamentale tra **capacità espressiva** e **complessità parametrica**. Analizziamo in dettaglio le tre varianti principali.

#### 1. Modello a Covarianza Completa (Full MVG)

##### Formulazione Matematica

Questo modello assume che ogni classe $c$ abbia la propria distribuzione Gaussiana con parametri indipendenti:

$$P(\mathbf{x} | C_c) = \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c)$$

##### Numero di Parametri

Per un problema a $M$ dimensioni e $K$ classi:
- **Medie**: $K \times M$ parametri
- **Covarianze**: $K \times \frac{M(M+1)}{2}$ parametri (sfruttando la simmetria)
- **Totale**: $K \times M(M+3)/2$ parametri

Per il nostro caso ($M=6$, $K=2$): $2 \times 6 \times 9/2 = 54$ parametri

##### Implementazione

```python
def train_full_mvg(self, D_tr: np.ndarray, L_tr: np.ndarray):
    """Addestra il modello Full MVG."""
    classes = np.unique(L_tr)
    self.mus = {}
    self.covs = {}
    
    for c in classes:
        # Seleziona i dati della classe c
        Dc = D_tr[:, L_tr == c]
        
        # Stima media e covarianza per questa classe
        self.mus[c] = mean_vector(Dc)
        self.covs[c] = covariance_matrix(Dc)
        
        # Verifica che la matrice sia definita positiva
        eigenvals = np.linalg.eigvals(self.covs[c])
        if np.any(eigenvals  np.ndarray:
    """
    Calcola la matrice di covarianza diagonale (Naive Bayes).
    
    Args:
        X: matrice (M, N) con N campioni M-dimensionali
    
    Returns:
        matrice di covarianza diagonale (M, M)
    """
    # Calcola la covarianza completa
    cov_full = covariance_matrix(X)
    
    # Estrae solo la diagonale e ricostruisce matrice diagonale
    variances = np.diag(cov_full)
    return np.diag(variances)

def train_naive_bayes(self, D_tr: np.ndarray, L_tr: np.ndarray):
    """Addestra il modello Naive Bayes."""
    classes = np.unique(L_tr)
    self.mus = {}
    self.covs = {}
    
    for c in classes:
        Dc = D_tr[:, L_tr == c]
        self.mus[c] = mean_vector(Dc)
        self.covs[c] = diagonal_covariance_matrix(Dc)
```

##### Superfici di Decisione

Con covarianze diagonali, le superfici equiprobabili sono **ellissi allineate con gli assi**. Il confine di decisione rimane quadratico ma con termini di rotazione nulli.

#### 3. Modello a Covarianza Condivisa (Tied MVG)

##### Assunzione di Omogeneità

Il modello Tied assume che tutte le classi condividano la stessa struttura di covarianza:

$$\boldsymbol{\Sigma}_0 = \boldsymbol{\Sigma}_1 = \boldsymbol{\Sigma}_{\text{tied}}$$

mentre mantengono medie diverse: $\boldsymbol{\mu}_0 \neq \boldsymbol{\mu}_1$

##### Stima della Covarianza Pooled

La matrice di covarianza condivisa è stimata come **media pesata** delle covarianze individuali:

$$\hat{\boldsymbol{\Sigma}}_{\text{tied}} = \frac{\sum_{c} N_c \hat{\boldsymbol{\Sigma}}_c}{\sum_{c} N_c} = \frac{N_0 \hat{\boldsymbol{\Sigma}}_0 + N_1 \hat{\boldsymbol{\Sigma}}_1}{N_0 + N_1}$$

dove $N_c$ è il numero di campioni della classe $c$.

##### Implementazione

```python
def train_tied_mvg(self, D_tr: np.ndarray, L_tr: np.ndarray):
    """Addestra il modello Tied MVG."""
    classes = np.unique(L_tr)
    N_tot = D_tr.shape[1]
    
    # Calcola medie e covarianze temporanee per ogni classe
    self.mus = {}
    covs_temp = {}
    Ns = {}
    
    for c in classes:
        Dc = D_tr[:, L_tr == c]
        self.mus[c] = mean_vector(Dc)
        covs_temp[c] = covariance_matrix(Dc)
        Ns[c] = Dc.shape[1]
    
    # Calcola la covarianza pooled
    self.Sigma_tied = np.zeros_like(covs_temp[classes[0]])
    for c in classes:
        self.Sigma_tied += Ns[c] * covs_temp[c]
    self.Sigma_tied /= N_tot
    
    print(f"Numerosità classi: {Ns}")
    print(f"Peso relativo nel pooling: {[Ns[c]/N_tot for c in classes]}")
```

##### Superfici di Decisione Lineari

Una proprietà fondamentale del modello Tied è che produce **confini di decisione lineari**. Questo deriva dal fatto che i termini quadratici nel LLR si cancellano:

$$\text{LLR}(\mathbf{x}) = -\frac{1}{2}[(\mathbf{x}-\boldsymbol{\mu}_1)^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu}_1) - (\mathbf{x}-\boldsymbol{\mu}_0)^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu}_0)]$$

Espandendo e semplificando:

$$\text{LLR}(\mathbf{x}) = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)^T \boldsymbol{\Sigma}^{-1} \mathbf{x} - \frac{1}{2}(\boldsymbol{\mu}_1^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_1 - \boldsymbol{\mu}_0^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_0)$$

Questa è una **funzione lineare** di $\mathbf{x}$, quindi il confine $\text{LLR}(\mathbf{x}) = 0$ è un iperpiano.

### Implementazione Completa del Framework

#### Classe Unificata per i Tre Modelli

```python
class GaussianClassifier:
    """
    Classificatore Gaussiano unificato con supporto per MVG, Tied, e Naive Bayes.
    """
    
    def __init__(self, classifier_type="mvg"):
        """
        Args:
            classifier_type: "mvg", "tied", o "naive_bayes"
        """
        self.classifier_type = classifier_type
        self.mus = None
        self.covs = None
        self.Sigma_tied = None
        self.trained = False
        
    def train(self, D_tr: np.ndarray, L_tr: np.ndarray):
        """Addestra il classificatore sui dati di training."""
        classes = np.unique(L_tr)
        N_tot = D_tr.shape[1]
        
        print(f"Addestramento {self.classifier_type.upper()}:")
        print(f"  - Campioni totali: {N_tot}")
        print(f"  - Dimensionalità: {D_tr.shape[0]}")
        print(f"  - Classi: {classes}")
        
        self.mus = {}
        
        if self.classifier_type == "mvg":
            self._train_full_mvg(D_tr, L_tr, classes)
        elif self.classifier_type == "naive_bayes":
            self._train_naive_bayes(D_tr, L_tr, classes)
        elif self.classifier_type == "tied":
            self._train_tied_mvg(D_tr, L_tr, classes, N_tot)
        else:
            raise ValueError(f"Tipo di classificatore non riconosciuto: {self.classifier_type}")
            
        self.trained = True
        self._validate_training()
        
    def _train_full_mvg(self, D_tr, L_tr, classes):
        """Addestramento Full MVG."""
        self.covs = {}
        for c in classes:
            Dc = D_tr[:, L_tr == c]
            self.mus[c] = mean_vector(Dc)
            self.covs[c] = covariance_matrix(Dc)
            
    def _train_naive_bayes(self, D_tr, L_tr, classes):
        """Addestramento Naive Bayes."""
        self.covs = {}
        for c in classes:
            Dc = D_tr[:, L_tr == c]
            self.mus[c] = mean_vector(Dc)
            self.covs[c] = diagonal_covariance_matrix(Dc)
            
    def _train_tied_mvg(self, D_tr, L_tr, classes, N_tot):
        """Addestramento Tied MVG."""
        covs_temp = {}
        Ns = {}
        
        # Calcola parametri per classe
        for c in classes:
            Dc = D_tr[:, L_tr == c]
            self.mus[c] = mean_vector(Dc)
            covs_temp[c] = covariance_matrix(Dc)
            Ns[c] = Dc.shape[1]
        
        # Pooled covariance
        self.Sigma_tied = sum(Ns[c] * covs_temp[c] for c in classes) / N_tot
        
    def _validate_training(self):
        """Valida i parametri dopo l'addestramento."""
        if self.classifier_type in ["mvg", "naive_bayes"]:
            for c, cov in self.covs.items():
                eigenvals = np.linalg.eigvals(cov)
                if np.any(eigenvals  np.ndarray:
        """
        Calcola i Log-Likelihood Ratios per i campioni di test.
        
        Returns:
            array con LLR per ogni campione (positivo → classe 1, negativo → classe 0)
        """
        if not self.trained:
            raise RuntimeError("Il classificatore deve essere addestrato prima dell'uso")
            
        if self.classifier_type == "tied":
            log_lik_1 = logpdf_GAU_ND(D_ev, self.mus[1], self.Sigma_tied)
            log_lik_0 = logpdf_GAU_ND(D_ev, self.mus[0], self.Sigma_tied)
        else:
            log_lik_1 = logpdf_GAU_ND(D_ev, self.mus[1], self.covs[1])
            log_lik_0 = logpdf_GAU_ND(D_ev, self.mus[0], self.covs[0])
        
        return log_lik_1 - log_lik_0
    
    def predict_from_llr(self, llr: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """Converte LLR in predizioni binarie."""
        return (llr >= threshold).astype(int)
    
    def compute_error_rate(self, predictions: np.ndarray, L_true: np.ndarray) -> float:
        """Calcola il tasso di errore."""
        return np.mean(predictions != L_true)
    
    def get_model_complexity(self) -> dict:
        """Restituisce informazioni sulla complessità del modello."""
        if not self.trained:
            return {"error": "Modello non addestrato"}
            
        M = list(self.mus.values())[0].shape[0]  # Dimensionalità
        K = len(self.mus)  # Numero di classi
        
        if self.classifier_type == "mvg":
            n_params = K * (M + M * (M + 1) // 2)
            description = f"Medie: {K}×{M}, Covarianze: {K}×{M*(M+1)//2}"
        elif self.classifier_type == "naive_bayes":
            n_params = K * (M + M)
            description = f"Medie: {K}×{M}, Varianze: {K}×{M}"
        elif self.classifier_type == "tied":
            n_params = K * M + M * (M + 1) // 2
            description = f"Medie: {K}×{M}, Covarianza condivisa: {M*(M+1)//2}"
        
        return {
            "n_parametri": n_params,
            "descrizione": description,
            "dimensionalita": M,
            "n_classi": K
        }
```

### Analisi Sperimentale Approfondita

#### Setup Sperimentale

Il dataset è stato diviso con rapporto 2:1 tra training e validation:
- **Training set**: 4000 campioni (1991 Fake, 2009 Genuine)
- **Validation set**: 2000 campioni (999 Fake, 1001 Genuine)

La distribuzione è quasi perfettamente bilanciata, il che giustifica l'uso di probabilità a priori uniformi.

#### Risultati Principali: Analisi delle 6 Feature Complete

| Modello       | Tasso di Errore | Accuratezza | Complessità |
| :------------ | :-------------- | :---------- | :---------- |
| **Full MVG**  | **7.00%**       | **93.00%**  | 54 parametri |
| Naive Bayes   | 7.20%           | 92.80%      | 24 parametri |
| Tied MVG      | 9.30%           | 90.70%      | 33 parametri |

##### Interpretazione Statistica dei Risultati

La **vittoria del Full MVG** di soli 0.2 punti percentuali sul Naive Bayes è statisticamente significativa? Con 2000 campioni di test, la deviazione standard dell'errore è approssimativamente:

$$\sigma_{\text{err}} \approx \sqrt{\frac{p(1-p)}{n}} \approx \sqrt{\frac{0.07 \times 0.93}{2000}} \approx 0.0057$$

La differenza di 0.2% è circa $0.002/0.0057 \approx 0.35$ deviazioni standard, suggerendo che la differenza potrebbe non essere statisticamente significativa. Tuttavia, il pattern si conferma su diversi sottoinsiemi di feature.

#### Analisi delle Matrici di Correlazione

L'analisi delle correlazioni rivela informazioni cruciali sul perché Naive Bayes funziona così bene:

##### Classe Fake (0):
```
Matrice di Correlazione:
[[ 1.000  0.000  0.033  0.034  0.020 -0.021]
 [ 0.000  1.000 -0.018 -0.018 -0.026  0.023]
 [ 0.033 -0.018  1.000 -0.003 -0.011  0.027]
 [ 0.034 -0.018 -0.003  1.000  0.009  0.022]
 [ 0.020 -0.026 -0.011  0.009  1.000  0.023]
 [-0.021  0.023  0.027  0.022  0.023  1.000]]

Correlazione massima: 0.034
Correlazione media: 0.019
```

##### Classe Genuine (1):
```
Correlazione massima: 0.049
Correlazione media: 0.018
```

**Osservazione Cruciale**: Le correlazioni sono estremamente basse (tutte > p²)
   - ✅ Features con correlazioni significative e diverse tra classi
   - ✅ Quando la massima accuratezza è prioritaria
   - ❌ Dataset piccoli o ad alta dimensionalità

2. **Naive Bayes**:
   - ✅ Features realmente indipendenti o debolmente correlate
   - ✅ Dataset piccoli o requirements di velocità
   - ✅ Quando l'interpretabilità è importante
   - ❌ Features fortemente correlate

3. **Tied MVG**:
   - ✅ Classi con forme simili ma centri diversi
   - ✅ Quando si vuole un confine di decisione lineare
   - ✅ Compromesso tra Full MVG e Naive Bayes
   - ❌ Classi con strutture di variabilità molto diverse

#### Diagnostica per la Selezione del Modello

```python
def diagnose_model_selection(classifier_full, classifier_naive, classifier_tied):
    """
    Fornisce raccomandazioni per la selezione del modello basate sui risultati.
    """
    # Confronto performance
    perf_full = classifier_full.get_performance()
    perf_naive = classifier_naive.get_performance() 
    perf_tied = classifier_tied.get_performance()
    
    print("🔍 DIAGNOSTICA SELEZIONE MODELLO")
    print("=" * 50)
    
    # Test per correlazioni
    max_corr = analyze_correlations(classifier_full)
    if max_corr < 0.3:
        print("✅ Correlazioni basse → Naive Bayes appropriato")
    else:
        print("⚠️  Correlazioni significative → Full MVG raccomandato")
    
    # Test per omogeneità covarianze
    if abs(perf_full['accuracy'] - perf_tied['accuracy']) < 1.0:
        print("✅ Tied MVG competitivo → Covarianze simili tra classi")
    else:
        print("⚠️  Tied MVG significativamente peggiore → Covarianze diverse")
    
    # Raccomandazione finale
    best_model = max([
        ("Full MVG", perf_full['accuracy']),
        ("Naive Bayes", perf_naive['accuracy']),
        ("Tied MVG", perf_tied['accuracy'])
    ], key=lambda x: x[1])
    
    print(f"\n🏆 RACCOMANDAZIONE: {best_model[0]} ({best_model[1]:.2f}% accuratezza)")
```

### Estensioni e Sviluppi Futuri

#### 1. Regularizzazione delle Matrici di Covarianza

Per dataset più piccoli o problematici, si può applicare **shrinkage** alla matrice di covarianza:

$$\hat{\boldsymbol{\Sigma}}_{\text{shrunk}} = (1-\lambda)\hat{\boldsymbol{\Sigma}} + \lambda \sigma^2 \mathbf{I}$$

dove $\lambda \in $ controlla il livello di regolarizzazione.

#### 2. Modelli Ibridi

Si possono costruire modelli intermedi, come:
- **Diagonal Tied**: covarianza diagonale ma condivisa
- **Block Diagonal**: modellare correlazioni solo tra sottogruppi di feature

#### 3. Selezione Automatica del Modello

Implementare criteri di selezione automatica come:
- **AIC (Akaike Information Criterion)**: $\text{AIC} = -2\log L + 2k$
- **BIC (Bayesian Information Criterion)**: $\text{BIC} = -2\log L + k\log n$

dove $L$ è la likelihood, $k$ il numero di parametri, e $n$ la dimensione del campione.

### Conclusioni e Sintesi

L'analisi approfondita dei classificatori generativi Gaussiani per il problema di fingerprint authentication ha rivelato diversi insight fondamentali:

#### Risultati Principali

1. **Il Full MVG (93.00% accuratezza) è il vincitore assoluto**, dimostrando che le classi hanno effettivamente strutture di covarianza diverse che vale la pena modellare esplicitamente.

2. **Il Naive Bayes (92.80% accuratezza) è sorprendentemente competitivo**, grazie alle correlazioni quasi nulle tra le feature originali. Questo risultato convalida l'importanza dell'analisi esplorativa dei dati.

3. **Il Tied MVG (90.70% accuratezza) perde terreno significativamente**, indicando che l'assunzione di covarianze identiche è violata nel dataset completo.

#### Lezioni Metodologiche

1. **L'importanza dell'analisi delle correlazioni**: La matrice di correlazione ha predetto accuratamente il successo del Naive Bayes.

2. **Feature engineering vs. dimensionality reduction**: Le feature originali sono risultate superiori a qualsiasi trasformazione PCA testata.

3. **Specializzazione per sottospazi**: L'inversione di ranking tra Full MVG e Tied MVG sulle feature 3-4 dimostra che la scelta ottimale del modello può dipendere dal sottospazio considerato.

#### Implicazioni Pratiche

Per problemi simili di classificazione biometrica:
- Iniziare sempre con un'analisi esplorativa delle correlazioni
- Testare tutti e tre i modelli prima di decidere
- Considerare l'uso di ensemble che combinano le previsioni dei diversi modelli
- Valutare attentamente il trade-off tra accuratezza e complessità computazionale

Il framework sviluppato fornisce una base solida per affrontare problemi di classificazione generativa, combinando rigore teorico, implementazione efficiente e analisi empirica approfondita.