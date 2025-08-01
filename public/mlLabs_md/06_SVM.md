### Argomento 6: Support Vector Machine – La Ricerca del Margine Massimo

Dopo aver esplorato modelli generativi e la Regolazione Logistica, ci addentriamo in un'altra classe di modelli discriminativi: le **Support Vector Machines (SVM)**. L'idea fondamentale delle SVM è diversa da quella della Regolazione Logistica. Invece di modellare la probabilità, le SVM cercano di trovare l'iperpiano di separazione che massimizza la "distanza" o il **margine** tra i punti più vicini delle due classi (i cosiddetti *vettori di supporto*). Questo approccio, basato sulla massimizzazione del margine, mira a creare un classificatore che sia il più robusto possibile al rumore e che generalizzi bene su dati nuovi.

#### Fondamenti Teorici: Dal Margine alla Hinge Loss

Il concetto di **margine** è centrale nelle SVM. Per un classificatore lineare definito da $\mathbf{w}^T\mathbf{x} + b = 0$, il margine geometrico di un punto $\mathbf{x}_i$ con etichetta $z_i \in \{-1, +1\}$ è dato da:

$$\text{margine}(\mathbf{x}_i) = \frac{z_i(\mathbf{w}^T\mathbf{x}_i + b)}{||\mathbf{w}||}$$

L'obiettivo delle SVM è massimizzare il margine minimo tra tutti i punti di training, equivalentemente minimizzando $||\mathbf{w}||^2$ soggetto ai vincoli $z_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1$.

Per gestire i casi in cui i dati non sono linearmente separabili, si introducono le **variabili slack** $\xi_i \geq 0$, portando alla formulazione **soft-margin**:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} ||\mathbf{w}||^2 + C \sum_{i=1}^N \xi_i$$

soggetto a:
- $z_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 - \xi_i$
- $\xi_i \geq 0$

Questa formulazione può essere riscritta utilizzando la **Hinge Loss**:

$$\mathcal{L}_{\text{hinge}}(z, s) = \max(0, 1 - zs)$$

ottenendo il problema non vincolato:

$$\min_{\mathbf{w}, b} \frac{1}{2} ||\mathbf{w}||^2 + C \sum_{i=1}^N \max(0, 1 - z_i(\mathbf{w}^T\mathbf{x}_i + b))$$

#### La Formulazione Duale e il Kernel Trick

La chiave del successo delle SVM risiede nella **formulazione duale** del problema di ottimizzazione. Utilizzando i moltiplicatori di Lagrange $\alpha_i$, il problema duale diventa:

$$\max_{\boldsymbol{\alpha}} \mathcal{L}_D(\boldsymbol{\alpha}) = \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j z_i z_j k(\mathbf{x}_i, \mathbf{x}_j)$$

con vincoli:
- $0 \leq \alpha_i \leq C$ per tutti gli $i$
- $\sum_{i=1}^N \alpha_i z_i = 0$ (per SVM con bias)

dove $k(\mathbf{x}_i, \mathbf{x}_j)$ è la **funzione kernel**. Per il caso lineare, $k(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j$.

La soluzione ottimale è data da:
$$\mathbf{w}^* = \sum_{i=1}^N \alpha_i^* z_i \mathbf{x}_i$$

e la funzione di classificazione diventa:
$$f(\mathbf{x}) = \sum_{i=1}^N \alpha_i^* z_i k(\mathbf{x}_i, \mathbf{x}) + b^*$$

```python
def train_dual_SVM_linear(DTR, LTR, C, K=1.0):
    """Train linear SVM using dual formulation with bias handling"""
    # Convert labels to {-1, +1}
    ZTR = LTR * 2.0 - 1.0
    
    # Extend data matrix to include bias term with parameter K
    DTR_EXT = np.vstack([DTR, np.ones((1, DTR.shape[1])) * K])
    
    # Construct Hessian matrix for dual problem
    H = np.dot(DTR_EXT.T, DTR_EXT) * vcol(ZTR) * vrow(ZTR)

    # Dual objective function (to minimize: -L_D)
    def fOpt(alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - np.ones(alpha.size)
        return loss, grad

    # Solve dual problem using L-BFGS-B
    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(
        fOpt, np.zeros(DTR_EXT.shape[1]), 
        bounds=[(0, C) for i in LTR], 
        factr=1e7, pgtol=1e-5
    )
    
    # Recover primal solution from dual variables
    w_hat = (vrow(alphaStar) * vrow(ZTR) * DTR_EXT).sum(1)
    w, b = w_hat[0:DTR.shape[0]], w_hat[-1] * K
    
    return w, b, alphaStar
```

#### Kernels Non-Lineari: Espandere lo Spazio delle Features

Il **kernel trick** permette di lavorare implicitamente in spazi a dimensionalità molto alta senza calcolare esplicitamente la trasformazione $\phi(\mathbf{x})$. I kernel più comuni sono:

##### 1. Kernel Polinomiale
$$k(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i^T \mathbf{x}_j + c)^d$$

Per $d=2$ e $c=1$, questo kernel cattura tutte le interazioni quadratiche tra le features:

```python
def polyKernel(degree, c):
    """Create polynomial kernel function"""
    def polyKernelFunc(D1, D2):
        return (np.dot(D1.T, D2) + c) ** degree
    return polyKernelFunc

# Usage for quadratic kernel
kernel_func = polyKernel(2, 1)
```

##### 2. Kernel RBF (Radial Basis Function)
$$k(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma ||\mathbf{x}_i - \mathbf{x}_j||^2)$$

Il kernel RBF mappa i dati in uno spazio a dimensionalità infinita:

```python
def rbfKernel(gamma):
    """Create RBF kernel function"""
    def rbfKernelFunc(D1, D2):
        # Efficient computation using broadcasting
        D1Norms = (D1**2).sum(0)
        D2Norms = (D2**2).sum(0)
        Z = vcol(D1Norms) + vrow(D2Norms) - 2 * np.dot(D1.T, D2)
        return np.exp(-gamma * Z)
    return rbfKernelFunc
```

L'implementazione per SVM con kernel generale è:

```python
def train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps=1.0):
    """Train SVM with generic kernel function"""
    ZTR = LTR * 2.0 - 1.0
    
    # Compute Gram matrix with bias term
    K = kernelFunc(DTR, DTR) + eps
    H = vcol(ZTR) * vrow(ZTR) * K

    def fOpt(alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - np.ones(alpha.size)
        return loss, grad

    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(
        fOpt, np.zeros(DTR.shape[1]), 
        bounds=[(0, C) for i in LTR], 
        factr=1e7, pgtol=1e-5
    )

    # Return scoring function for new samples
    def fScore(DTE):
        K_test = kernelFunc(DTR, DTE) + eps
        H_test = vcol(alphaStar) * vcol(ZTR) * K_test
        return H_test.sum(0)

    return fScore, alphaStar
```

#### Duality Gap e Condizioni KKT

Un aspetto cruciale nell'ottimizzazione delle SVM è il **duality gap**, che misura la differenza tra la soluzione primale e duale:

$$\text{Duality Gap} = \mathcal{L}_P(\mathbf{w}^*, b^*) - \mathcal{L}_D(\boldsymbol{\alpha}^*)$$

dove:
- $\mathcal{L}_P = \frac{1}{2}||\mathbf{w}||^2 + C\sum_{i=1}^N \max(0, 1-z_i(\mathbf{w}^T\mathbf{x}_i + b))$ (Loss Primale)
- $\mathcal{L}_D = \sum_{i=1}^N \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j z_i z_j k(\mathbf{x}_i, \mathbf{x}_j)$ (Loss Duale)

Un duality gap vicino a zero indica che l'ottimizzazione ha raggiunto la convergenza. Le **condizioni KKT** (Karush-Kuhn-Tucker) caratterizzano la soluzione ottimale:

1. $\alpha_i \geq 0$ (feasibility primale)
2. $z_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 - \xi_i$ (feasibility duale)
3. $\alpha_i[z_i(\mathbf{w}^T\mathbf{x}_i + b) - 1 + \xi_i] = 0$ (complementary slackness)

I punti con $\alpha_i > 0$ sono i **support vectors** e sono gli unici che influenzano la decisione finale.

#### Esperimento 1: SVM Lineare – Un Inizio Sottotono

Abbiamo testato una SVM lineare su dati originali e centrati, variando il parametro C secondo una progressione logaritmica $C \in [10^{-5}, 10^0]$.

```python
def analyze_linear_svm(DTR, LTR, DVAL, LVAL, centered=False):
    """Comprehensive analysis of linear SVM performance"""
    # C values - logarithmic scale for comprehensive exploration
    C_values = np.logspace(-5, 0, 11)
    
    minDCFs, actDCFs, duality_gaps = [], [], []
    
    for C in C_values:
        # Train model
        w, b, alphaStar = train_dual_SVM_linear(DTR, LTR, C)
        
        # Compute validation scores
        scores = (vrow(w) @ DVAL + b).ravel()
        
        # Evaluate using Bayes risk framework
        minDCF = bayesRisk.compute_minDCF_binary_fast(scores, LVAL, 0.1, 1.0, 1.0)
        actDCF = bayesRisk.compute_actDCF_binary_fast(scores, LVAL, 0.1, 1.0, 1.0)
        
        minDCFs.append(minDCF)
        actDCFs.append(actDCF)
    
    return C_values, minDCFs, actDCFs
```

<p align="center">
  <img src="/mlLabs_screens/06_SVM/linear_svm_Original_Data.png" width="48%" />
  <img src="/mlLabs_screens/06_SVM/linear_svm_Centered_Data.png" width="48%" />
</p>

##### Analisi Dettagliata dei Risultati Lineari

Dai risultati empirici emergono pattern interessanti:

**Performance Analysis (Grafici di Sinistra):**
- **Convergenza Rapida**: Il minDCF si stabilizza rapidamente per $C \geq 10^{-2}$, suggerendo che il modello raggiunge la sua capacità espressiva massima con regolarizzazione moderata.
- **Plateau Effect**: Il minDCF oscilla minimamente attorno a **0.3582** per un'ampia gamma di valori di C, indicando robustezza nella scelta dell'iperparametro.
- **Optimal Point**: Il minimo si ottiene per **C = 0.1**, bilanciando efficacemente regolarizzazione e fitting dei dati.

**Calibration Analysis (Grafici Centrali):**
La calibrazione delle SVM lineari rivela una problematica fondamentale:
- **Gap Enorme**: La differenza tra actDCF e minDCF supera sistematicamente 0.15, indicando che i punteggi grezzi $\mathbf{w}^T\mathbf{x} + b$ sono molto distanti dalle log-odds calibrate.
- **Natura della Hinge Loss**: A differenza della cross-entropy, la hinge loss non è progettata per produrre probabilità ben calibrate, ma solo per separare correttamente le classi.
- **Implicazioni Pratiche**: In applicazioni real-world dove le soglie di decisione possono variare, è essenziale un passo di calibrazione post-training.

**Centrature dei Dati:**
Il confronto tra dati originali e centrati rivela:
- **Invarianza Traslazionale**: Le performance sono quasi identiche (0.3582 vs 0.3590), confermando che le SVM lineari sono intrinsecamente robuste alle traslazioni dei dati.
- **Stabilità Numerica**: La stabilità dei risultati suggerisce che l'ottimizzazione è ben condizionata in entrambi i casi.

#### Esperimento 2: SVM con Kernel – Entrare nella Non Linearità

Il vero potere delle SVM si manifesta con l'uso dei kernel non-lineari, che permettono di trovare confini di separazione complessi senza aumentare esplicitamente la dimensionalità.

##### SVM Polinomiale (Grado 2)

Il kernel polinomiale $k(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i^T \mathbf{x}_j + 1)^2$ mappa implicitamente i dati in uno spazio che include tutte le interazioni quadratiche. Per un vettore di features $\mathbf{x} = [x_1, x_2, ..., x_d]^T$, la mappatura equivalente è:

$$\phi(\mathbf{x}) = [1, \sqrt{2}x_1, ..., \sqrt{2}x_d, x_1^2, ..., x_d^2, \sqrt{2}x_1x_2, ..., \sqrt{2}x_{d-1}x_d]^T$$

![](/mlLabs_screens/06_SVM/polynomial_svm_analysis.png)

**Analisi dei Risultati Polinomiali:**

Il miglioramento drammatico delle performance (minDCF da 0.3582 a **0.2455**) conferma l'importanza delle non-linearità quadratiche per questo dataset specifico. Questo risultato è consistente con quello ottenuto dalla Logistic Regression Quadratica, suggerendo che:

1. **Struttura Intrinseca**: Il dataset presenta relazioni quadratiche fondamentali tra le features
2. **Optimal Complexity**: Il grado 2 cattura la complessità necessaria senza overfitting
3. **Feature Interactions**: Le interazioni incrociate tra features sono cruciali per la separazione delle classi

L'analisi del parametro C mostra che il modello polinomiale raggiunge il suo optimum per **C = 3.2×10^{-2}**, un valore più basso rispetto al caso lineare, indicando che la maggiore espressività del kernel richiede più regolarizzazione per prevenire l'overfitting.

##### SVM con Kernel RBF: Massima Flessibilità

Il kernel RBF rappresenta il culmine della flessibilità, mappando i dati in uno spazio di Hilbert a dimensionalità infinita. La funzione:

$$k(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma ||\mathbf{x}_i - \mathbf{x}_j||^2)$$

può essere interpretata come una misura di similarità che decresce esponenzialmente con la distanza euclidea. Il parametro $\gamma$ controlla la "larghezza" delle funzioni di base radiali:
- **γ alto**: Funzioni strette, decisioni locali, rischio di overfitting
- **γ basso**: Funzioni larghe, decisioni più globali, possibile underfitting

La necessità di ottimizzare congiuntamente γ e C richiede una grid search sistematica:

```python
def analyze_rbf_svm_grid_search(DTR, LTR, DVAL, LVAL):
    """Comprehensive grid search for RBF SVM hyperparameters"""
    # Logarithmic spacing for gamma (based on feature space characteristics)
    gamma_values = [np.exp(-4), np.exp(-3), np.exp(-2), np.exp(-1)]  # [0.018, 0.050, 0.135, 0.368]
    C_values = np.logspace(-3, 2, 11)  # Wider range for C
    
    results = {}
    best_result = {'minDCF': float('inf'), 'gamma': None, 'C': None}
    
    for gamma in gamma_values:
        kernel_func = rbfKernel(gamma)
        minDCFs_gamma, actDCFs_gamma = [], []
        
        for C in C_values:
            # Train with current hyperparameters
            fScore, alphaStar = train_dual_SVM_kernel(DTR, LTR, C, kernel_func, eps=1.0)
            scores = fScore(DVAL)
            
            # Evaluate performance
            minDCF = bayesRisk.compute_minDCF_binary_fast(scores, LVAL, 0.1, 1.0, 1.0)
            actDCF = bayesRisk.compute_actDCF_binary_fast(scores, LVAL, 0.1, 1.0, 1.0)
            
            minDCFs_gamma.append(minDCF)
            actDCFs_gamma.append(actDCF)
            
            # Track global optimum
            if minDCF < best_result['minDCF']:
                best_result.update({
                    'minDCF': minDCF, 'actDCF': actDCF,
                    'gamma': gamma, 'C': C
                })
        
        results[gamma] = {
            'C_values': C_values.copy(),
            'minDCFs': minDCFs_gamma,
            'actDCFs': actDCFs_gamma
        }
    
    return results, best_result
```

![](/mlLabs_screens/06_SVM/rbf_svm_grid_search.png)

**Analisi Approfondita dei Risultati RBF:**

Il grafico della grid search rivela pattern complessi e informativi:

**Heatmap Analysis (Grafico in basso a destra):**
- **Sweet Spot**: La regione ottimale (colori più scuri) si concentra attorno a γ = 0.135 e C elevati (10-100)
- **γ Trade-off**: 
  - γ troppo basso (0.018): Performance moderate, il modello è troppo "liscio"
  - γ troppo alto (0.368): Instabilità per C bassi, overfitting per C alti
  - γ ottimale (0.135): Equilibrio perfetto tra localizzazione e generalizzazione

**C Parameter Analysis:**
- **Low C Region**: Per tutti i γ, C bassi portano a underfitting severo (minDCF → 1.0)
- **High C Region**: L'effetto di C alto dipende fortemente da γ:
  - Con γ basso: Miglioramento monotono
  - Con γ alto: Possibile degradazione per overfitting

**Optimal Configuration:**
Il punto ottimale (**γ = 0.135, C = 32.0**) raggiunge un **minDCF = 0.1735**, stabilendo un nuovo record di performance. Questa configurazione rappresenta un equilibrio ottimale:
- **Complessità Locale**: γ = 0.135 permette decisioni sufficientemente locali per catturare pattern complessi
- **Regolarizzazione**: C = 32.0 consente al modello di adattarsi ai dati mantenendo generalizzazione

#### Interpretazione Geometrica e Support Vectors

L'analisi dei support vectors fornisce insight sulla struttura del problema:

```python
def analyze_support_vectors(alphaStar, DTR, LTR, threshold=1e-6):
    """Analyze support vector distribution and characteristics"""
    # Identify support vectors (non-zero alpha values)
    sv_indices = np.where(alphaStar > threshold)[0]
    n_support_vectors = len(sv_indices)
    
    # Classify support vectors by alpha value
    bounded_sv = np.sum((alphaStar > threshold) & (alphaStar < C - threshold))
    margin_sv = np.sum(alphaStar >= C - threshold)
    
    return {
        'n_total': len(alphaStar),
        'n_support_vectors': n_support_vectors,
        'support_vector_ratio': n_support_vectors / len(alphaStar),
        'bounded_sv': bounded_sv,  # 0 < alpha < C
        'margin_sv': margin_sv     # alpha = C
    }
```

I support vectors si dividono in due categorie:
1. **Margin Support Vectors** (0 < αᵢ < C): Punti esattamente sul margine
2. **Bounded Support Vectors** (αᵢ = C): Punti violanti il margine (all'interno o misclassificati)

La distribuzione dei support vectors varia significativamente tra i kernel:
- **Linear SVM**: Pochi support vectors concentrati vicino al boundary lineare
- **RBF SVM**: Molti support vectors distribuiti per catturare la complessità non-lineare

#### Conclusioni Finali: Gerarchia delle Performance e Implicazioni

La gerarchia definitiva delle performance SVM rivela insight profondi sulla natura del problema:

| Rank | Model | minDCF | actDCF | Parameters | Caratteristiche |
|------|-------|--------|--------|------------|-----------------|
| 1 | **RBF SVM** | **0.1735** | 0.4226 | γ=0.135, C=32.0 | Massima flessibilità |
| 2 | **Polynomial SVM** | **0.2455** | 0.4654 | C=3.2×10⁻² | Interazioni quadratiche |
| 3 | **Linear SVM (Original)** | **0.3582** | 0.5152 | C=0.1 | Baseline lineare |
| 4 | **Linear SVM (Centered)** | **0.3590** | 0.5797 | C=3.2×10⁻² | Preprocessing irrilevante |

**Implicazioni Teoriche:**

1. **Non-linearità Fondamentale**: Il miglioramento drammatico dal lineare (0.3582) al polinomiale (0.2455) e infine all'RBF (0.1735) dimostra che il dataset presenta strutture non-lineari complesse che richiedono modelli sofisticati.

2. **Curse of Dimensionality Benefico**: Contrariamente all'intuizione, l'aumento implicito della dimensionalità tramite kernel migliora le performance, suggerendo che il dataset beneficia della separazione in spazi ad alta dimensionalità.

3. **Optimal Complexity**: L'RBF, pur avendo capacità espressiva infinita, non presenta overfitting severo quando appropriatamente regolarizzato, indicando che il dataset è sufficientemente ricco da sfruttare tale flessibilità.

**Problematiche Critiche:**

**Calibrazione Universalmente Scarsa:**
Tutti i modelli SVM mostrano gap di calibrazione > 0.2, un problema sistematico che deriva dalla natura della hinge loss. Questo richiede:
- **Post-processing obbligatorio**: Platt scaling o calibrazione isotonica
- **Attenzione nelle applicazioni**: I punteggi grezzi non sono affidabili per decision making probabilistico

**Computational Complexity:**
- **Training**: O(N³) per la soluzione del problema duale
- **Prediction**: O(n_sv) dove n_sv è il numero di support vectors
- **Memory**: Storage di tutti i support vectors necessario

**Hyperparameter Sensitivity:**
L'RBF SVM richiede careful tuning di γ e C, con performance che possono variare drasticamente (da 0.17 a 1.0) in base alla configurazione.

#### Prospettive Future e Raccomandazioni

**Per Applicazioni Pratiche:**
1. **Pipeline Completa**: RBF SVM + calibrazione post-training per optimal performance
2. **Fallback Strategy**: Polynomial SVM come compromesso tra performance e robustezza
3. **Baseline Comparison**: Linear SVM come punto di riferimento per valutare la necessità di non-linearità

**Per Ricerca Avanzata:**
1. **Kernel Engineering**: Sviluppo di kernels domain-specific
2. **Scalability**: Implementazioni approssimate per dataset molto grandi
3. **Multi-class Extension**: Strategie one-vs-one o one-vs-all per problemi multi-classe

L'analisi delle SVM ha dimostrato il potere della non-linearità e stabilito un nuovo benchmark di performance (minDCF = 0.1735), pur evidenziando la necessità critica di calibrazione per applicazioni pratiche. La superiorità dell'RBF SVM conferma che questo dataset richiede modelli sofisticati per rivelare completamente la sua struttura latente.