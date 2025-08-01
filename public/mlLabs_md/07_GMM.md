## Argomento 7: Gaussian Mixture Models – Modellare la Complessità del Mondo Reale

Nei capitoli precedenti abbiamo esplorato un'ampia gamma di classificatori. Siamo partiti da modelli semplici come la singola Gaussiana (MVG) e abbiamo progressivamente aumentato la complessità, passando per modelli discriminativi lineari (Regolazione Logistica) e non lineari (SVM con kernel). Ora, torniamo ai modelli generativi, ma con uno strumento molto più potente e flessibile: i **Gaussian Mixture Models (GMM)**.

L'assunzione che i dati di una classe provengano da una singola distribuzione Gaussiana è spesso una semplificazione eccessiva. Nel mondo reale, i dati possono presentare sotto-cluster, modalità multiple o forme complesse che una singola elissoide non può catturare. I GMM superano questo limite modellando la distribuzione di probabilità di una classe come una **sovrapposizione pesata di più componenti Gaussiane**. È come usare un set di "pennelli" a forma di campana di varie dimensioni e orientamenti per "dipingere" la forma complessa della distribuzione dei dati.

### Fondamenti Matematici: Dalle Singole Gaussiane alle Mixture

#### Formulazione del Modello

Un Gaussian Mixture Model descrive la densità di probabilità di un campione $\mathbf{x}$ come una somma pesata di $M$ componenti Gaussiane:

$$p(\mathbf{x}) = \sum_{m=1}^{M} w_m \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_m, \boldsymbol{\Sigma}_m)$$

Dove per ogni componente $m$:
- $w_m$ è il peso (o probabilità a priori) della componente, con il vincolo che $\sum_{m=1}^{M} w_m = 1$
- $\mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_m, \boldsymbol{\Sigma}_m)$ è una normale funzione di densità Gaussiana multivariata con media $\boldsymbol{\mu}_m$ e matrice di covarianza $\boldsymbol{\Sigma}_m$

#### Interpretazione Geometrica e Probabilistica

I GMM possono essere interpretati come un **modello di variabili latenti**. Immaginiamo che ogni punto dati sia generato da un processo a due fasi:

1. **Selezione della componente**: Scegliamo una componente $m$ con probabilità $w_m$
2. **Generazione del campione**: Generiamo $\mathbf{x}$ dalla Gaussiana $\mathcal{N}(\boldsymbol{\mu}_m, \boldsymbol{\Sigma}_m)$

Questa interpretazione introduce la **variabile latente** $z_m \in \{0,1\}$ che indica quale componente ha generato il campione:

$$P(z_m = 1) = w_m$$
$$p(\mathbf{x}|z_m = 1) = \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_m, \boldsymbol{\Sigma}_m)$$

#### Classificazione con GMM

Per la classificazione, addestriamo un GMM separato per ogni classe. Per un dato campione $\mathbf{x}$, calcoliamo la verosimiglianza sotto il modello di ogni classe e prendiamo la decisione basandoci sul **Log-Likelihood Ratio (LLR)**:

$$\text{LLR}(\mathbf{x}) = \log p(\mathbf{x} | \text{GMM}_{\text{classe 1}}) - \log p(\mathbf{x} | \text{GMM}_{\text{classe 0}})$$

Il calcolo della log-densità per un GMM viene implementato usando la stabilità numerica del `logsumexp`:

```python
def logpdf_GMM(X, gmm):
    S = []
    for w, mu, C in gmm:
        logpdf_conditional = logpdf_GAU_ND(X, mu, C)
        logpdf_joint = logpdf_conditional + numpy.log(w)
        S.append(logpdf_joint)
    
    S = numpy.vstack(S)
    logdens = scipy.special.logsumexp(S, axis=0)
    return logdens
```

### L'Algoritmo Expectation-Maximization: Il Cuore dell'Addestramento

#### Formulazione del Problema di Ottimizzazione

L'addestramento di un GMM è un problema di **Maximum Likelihood Estimation (MLE)** complesso. Data una matrice di dati $\mathbf{X} = [\mathbf{x}_1, \ldots, \mathbf{x}_N]$, vogliamo massimizzare la log-likelihood:

$$\mathcal{L}(\boldsymbol{\theta}) = \sum_{n=1}^{N} \log p(\mathbf{x}_n | \boldsymbol{\theta}) = \sum_{n=1}^{N} \log \left(\sum_{m=1}^{M} w_m \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_m, \boldsymbol{\Sigma}_m)\right)$$

dove $\boldsymbol{\theta} = \{w_m, \boldsymbol{\mu}_m, \boldsymbol{\Sigma}_m\}_{m=1}^M$ sono i parametri del modello.

Il problema è che la presenza del logaritmo della somma rende impossibile una soluzione in forma chiusa. L'algoritmo **EM** risolve questo problema introducendo le variabili latenti e iterando tra due passi complementari.

#### E-step: Calcolo delle Responsabilità

Nel passo E, calcoliamo le **responsabilità** $\gamma_{nm}$ - la probabilità a posteriori che il campione $n$ sia stato generato dalla componente $m$:

$$\gamma_{nm} = P(z_{nm} = 1 | \mathbf{x}_n, \boldsymbol{\theta}^{(t)}) = \frac{w_m^{(t)} \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_m^{(t)}, \boldsymbol{\Sigma}_m^{(t)})}{\sum_{k=1}^{M} w_k^{(t)} \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_k^{(t)}, \boldsymbol{\Sigma}_k^{(t)})}$$

Intuitivamente, $\gamma_{nm}$ ci dice "quanto" il campione $n$ appartiene alla componente $m$.

#### M-step: Aggiornamento dei Parametri

Nel passo M, aggiorniamo i parametri massimizzando la log-likelihood pesata con le responsabilità:

**Aggiornamento dei pesi**:
$$w_m^{(t+1)} = \frac{1}{N} \sum_{n=1}^{N} \gamma_{nm} = \frac{N_m^{(t)}}{N}$$

dove $N_m^{(t)} = \sum_{n=1}^{N} \gamma_{nm}$ è il "numero effettivo" di campioni assegnati alla componente $m$.

**Aggiornamento delle medie**:
$$\boldsymbol{\mu}_m^{(t+1)} = \frac{\sum_{n=1}^{N} \gamma_{nm} \mathbf{x}_n}{\sum_{n=1}^{N} \gamma_{nm}} = \frac{1}{N_m^{(t)}} \sum_{n=1}^{N} \gamma_{nm} \mathbf{x}_n$$

**Aggiornamento delle covarianze**:
$$\boldsymbol{\Sigma}_m^{(t+1)} = \frac{1}{N_m^{(t)}} \sum_{n=1}^{N} \gamma_{nm} (\mathbf{x}_n - \boldsymbol{\mu}_m^{(t+1)})(\mathbf{x}_n - \boldsymbol{\mu}_m^{(t+1)})^T$$

#### Implementazione dell'Iterazione EM

```python
def train_GMM_EM_Iteration(X, gmm, covType='Full', psiEig=None):
    # E-step: Calcolo responsabilità
    S = []
    for w, mu, C in gmm:
        logpdf_conditional = logpdf_GAU_ND(X, mu, C)
        logpdf_joint = logpdf_conditional + numpy.log(w)
        S.append(logpdf_joint)
    
    S = numpy.vstack(S)
    logdens = scipy.special.logsumexp(S, axis=0)
    gammaAllComponents = numpy.exp(S - logdens)
    
    # M-step: Aggiornamento parametri
    gmmUpd = []
    for gIdx in range(len(gmm)):
        gamma = gammaAllComponents[gIdx]
        Z = gamma.sum()  # N_m
        F = vcol((vrow(gamma) * X).sum(1))  # Somma pesata
        S = (vrow(gamma) * X) @ X.T  # Matrice di scatter pesata
        
        muUpd = F/Z
        CUpd = S/Z - muUpd @ muUpd.T
        wUpd = Z / X.shape[1]
        
        if covType.lower() == 'diagonal':
            CUpd = CUpd * numpy.eye(X.shape[0])
        
        gmmUpd.append((wUpd, muUpd, CUpd))
    
    # Gestione covarianza tied e regolarizzazione
    if covType.lower() == 'tied':
        CTied = sum(w * C for w, mu, C in gmmUpd)
        gmmUpd = [(w, mu, CTied) for w, mu, C in gmmUpd]
    
    if psiEig is not None:
        gmmUpd = [(w, mu, smooth_covariance_matrix(C, psiEig)) 
                  for w, mu, C in gmmUpd]
    
    return gmmUpd
```

### Inizializzazione LBG: Evitare Minimi Locali Cattivi

#### Il Problema dell'Inizializzazione

L'algoritmo EM è garantito per convergere a un **massimo locale** della likelihood, ma la qualità di questo massimo dipende fortemente dall'inizializzazione. Una inizializzazione casuale può portare a soluzioni di scarsa qualità.

#### L'Algoritmo Linde-Buzo-Gray (LBG)

L'algoritmo **LBG** risolve questo problema con un approccio di **divisione iterativa**:

1. **Inizializzazione**: Partire da un singolo centroide (GMM con 1 componente)
2. **Divisione**: Dividere ogni componente esistente in due componenti "gemelle"
3. **Ottimizzazione**: Eseguire EM per ottimizzare i parametri
4. **Iterazione**: Ripetere fino al numero desiderato di componenti

#### Meccanismo di Divisione

Per dividere una componente con parametri $(w, \boldsymbol{\mu}, \boldsymbol{\Sigma})$:

1. Calcolare la **direzione principale** di variazione tramite SVD: $\boldsymbol{\Sigma} = \mathbf{U}\mathbf{S}\mathbf{V}^T$
2. Calcolare il **vettore di perturbazione**: $\mathbf{d} = \alpha \sqrt{s_1} \mathbf{u}_1$, dove $s_1$ è il primo valore singolare, $\mathbf{u}_1$ il primo vettore singolare, e $\alpha$ un fattore di scala
3. Creare due componenti gemelle:
   - $(w/2, \boldsymbol{\mu} - \mathbf{d}, \boldsymbol{\Sigma})$  
   - $(w/2, \boldsymbol{\mu} + \mathbf{d}, \boldsymbol{\Sigma})$

```python
def split_GMM_LBG(gmm, alpha=0.1):
    gmmOut = []
    for (w, mu, C) in gmm:
        U, s, Vh = numpy.linalg.svd(C)
        d = U[:, 0:1] * s[0]**0.5 * alpha
        gmmOut.append((0.5 * w, mu - d, C))
        gmmOut.append((0.5 * w, mu + d, C))
    return gmmOut

def train_GMM_LBG_EM(X, numComponents, covType='Full', psiEig=None, 
                     epsLLAverage=1e-6, lbgAlpha=0.1):
    # Inizializzazione con singola Gaussiana
    mu, C = compute_mu_C(X)
    if covType.lower() == 'diagonal':
        C = C * numpy.eye(X.shape[0])
    
    gmm = [(1.0, mu, C)]
    
    # Iterazioni LBG
    while len(gmm) < numComponents:
        gmm = split_GMM_LBG(gmm, lbgAlpha)
        gmm = train_GMM_EM(X, gmm, covType=covType, psiEig=psiEig, 
                          epsLLAverage=epsLLAverage)
    return gmm
```

### Varianti del Modello di Covarianza

#### Full Covariance vs. Diagonal Covariance vs. Tied Covariance

Come per i classificatori Gaussiani classici, i GMM supportano diverse assunzioni sulla struttura della covarianza:

**Full Covariance**: Ogni componente ha la propria matrice di covarianza completa
- **Parametri per componente**: $D + \frac{D(D+1)}{2}$ (media + covarianza)
- **Flessibilità**: Massima
- **Rischio overfitting**: Alto con pochi dati

**Diagonal Covariance**: Le matrici di covarianza sono diagonali
- **Parametri per componente**: $D + D = 2D$ (media + varianze)
- **Assunzione**: Indipendenza condizionale delle features
- **Vantaggio**: Regolarizzazione naturale

**Tied Covariance**: Tutte le componenti condividono la stessa matrice di covarianza
- **Parametri**: $M \cdot D + \frac{D(D+1)}{2}$ (M medie + 1 covarianza condivisa)
- **Assunzione**: Forma identica, solo traslazione

#### Regolarizzazione delle Covarianze

Per evitare matrici singolari o mal condizionate, si applica **eigenvalue thresholding**:

```python
def smooth_covariance_matrix(C, psi):
    U, s, Vh = numpy.linalg.svd(C)
    s[s < psi] = psi  # Soglia sugli autovalori
    CUpd = U @ (vcol(s) * U.T)
    return CUpd
```

Questo garantisce che tutti gli autovalori siano almeno $\psi$, evitando degenerazioni numeriche.

### Analisi Sperimentale Approfondita

#### Setup Sperimentale

Il nostro esperimento sistematico ha testato GMM con:
- **Numero di componenti**: $\{1, 2, 4, 8, 16, 32\}$
- **Tipi di covarianza**: Full e Diagonal
- **Dataset**: 4000 campioni training, 2000 validation
- **Metrica**: minDCF e actDCF con prior target $\pi_T = 0.1$

```python
def analyze_gmm_components(DTR, LTR, DVAL, LVAL, cov_type='full', 
                          max_components=32, target_prior=0.1):
    component_values = [1, 2, 4, 8, 16, 32]
    results = []
    
    for num_components in component_values:
        # Addestra GMM per entrambe le classi
        gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], num_components, 
                               covType=cov_type, psiEig=0.01)
        gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], num_components, 
                               covType=cov_type, psiEig=0.01)
        
        # Calcola LLR e performance
        ll0 = logpdf_GMM(DVAL, gmm0)
        ll1 = logpdf_GMM(DVAL, gmm1)
        llr_scores = ll1 - ll0
        
        minDCF = bayesRisk.compute_minDCF_binary_fast(
            llr_scores, LVAL, target_prior, 1.0, 1.0)
        actDCF = bayesRisk.compute_actDCF_binary_fast(
            llr_scores, LVAL, target_prior, 1.0, 1.0)
        
        results.append({
            'components': num_components,
            'minDCF': minDCF,
            'actDCF': actDCF,
            'calibration_gap': actDCF - minDCF
        })
    
    return results
```

#### Risultati dei GMM: Il Numero di Componenti è la Chiave

I risultati sperimentali rivelano pattern chiari e interpretabili:

![](/mlLabs_screens/07_GMM/gmm_analysis_comparison.png)

Il grafico "GMM Analysis Comparison" riassume in modo eccellente i risultati di questi esperimenti.

**Andamento del minDCF (in alto a sinistra)**: Entrambi i modelli mostrano un andamento a "U". Le performance migliorano all'aumentare del numero di componenti, raggiungono un punto ottimale e poi peggiorano. Questo è il classico trade-off tra *underfitting* e *overfitting*:

- Con poche componenti, il modello è troppo semplice per catturare la vera struttura dei dati
- Con troppe componenti, il modello diventa eccessivamente complesso e inizia a modellare il rumore specifico del training set, perdendo capacità di generalizzazione

**GMM a Covarianza Completa (linea blu)**: Raggiunge il suo punto di minima spesa (minDCF) con **16 componenti**, ottenendo un eccellente **minDCF = 0.1631**.

**GMM a Covarianza Diagonale (linea rossa)**: A sorpresa, questo modello più semplice fa ancora meglio. Raggiunge il suo minimo con sole **8 componenti**, ottenendo un **minDCF di 0.1463**, il miglior risultato visto finora.

#### Interpretazione Teorica dei Risultati

##### Perché Diagonal Supera Full?

Il successo del modello diagonal è controintuitivo ma ha spiegazioni teoriche solide:

1. **Regolarizzazione Intrinseca**: Forzare le covarianze a essere diagonali agisce come una potente forma di regolarizzazione, impedendo al modello di apprendere correlazioni spurie

2. **Efficienza Parametrica**: Con 8 componenti diagonal vs 16 componenti full:
   - Diagonal: $8 \times (6 + 6) = 96$ parametri per classe
   - Full: $16 \times (6 + 21) = 432$ parametri per classe
   - Rapporto: 4.5x meno parametri!

3. **Stabilità Numerica**: Meno parametri significano stime più stabili, specialmente quando il numero di campioni per componente diminuisce

4. **Allineamento con la Struttura dei Dati**: I risultati suggeriscono che le modalità principali di variazione sono effettivamente allineate con gli assi delle features

##### Analisi del Bias-Variance Trade-off

Il comportamento a "U" della curva di performance può essere analizzato attraverso la decomposizione bias-variance:

- **Poche componenti (1-2)**: Alto bias, bassa varianza → underfitting
- **Componenti ottimali (8 diagonal, 16 full)**: Bias e varianza bilanciati
- **Troppe componenti (32+)**: Basso bias, alta varianza → overfitting

#### Calibrazione: Un Vantaggio dei Modelli Generativi

**Calibrazione (in basso a sinistra del grafico)**: Entrambi i modelli GMM dimostrano una **calibrazione eccellente**. Il gap tra actDCF e minDCF è sempre molto basso (sotto il 4%), come ci si aspetta da modelli generativi ben addestrati. I loro punteggi sono affidabili.

Questa superiorità nella calibrazione deriva dal fatto che i GMM modellano esplicitamente le densità di probabilità, fornendo stime naturalmente calibrate delle probabilità a posteriori.

### Il Verdetto Finale: Confronto Completo

Dopo un lungo percorso attraverso diversi paradigmi di classificazione, siamo pronti per la resa dei conti finale. Abbiamo messo a confronto i migliori modelli di ogni famiglia: GMM, SVM e Regolazione Logistica.

#### Ranking Quantitativo Finale

| Classificatore              | Miglior minDCF | Calibrazione | Configurazione Ottimale                  |
| :-------------------------- | :------------- | :----------- | :--------------------------------------- |
| **GMM Diagonale**           | **0.1463**     | Eccellente   | 8 componenti                             |
| **GMM Completa**            | 0.1631         | Eccellente   | 16 componenti                            |
| **SVM (RBF)**               | 0.2391         | Pessima      | C=1.0, γ=1.0                             |
| **Regolazione Logistica**   | 0.3611         | Discreta     | Standard, λ=0.01                         |

Il **GMM con 8 componenti diagonali emerge come il vincitore indiscusso**. Non solo ottiene il più basso rischio di classificazione, ma lo fa con punteggi ben calibrati.

#### Analisi Visiva Comparativa

![](/mlLabs_screens/07_GMM/all_classifiers_-_bayes_error_plot_bayes_error_plot.png)

Il grafico "All Classifiers - Bayes Error Plot" fornisce una panoramica visiva particolarmente illuminante, mostrando come le curve del minDCF (linee continue) per i modelli GMM (rossa e blu) si mantengano nettamente più in basso rispetto a quelle della Regolazione Logistica (verde) e della SVM (viola) lungo l’intero spettro applicativo. Un aspetto cruciale messo in rilievo da questo grafico è il comportamento del gap tra minDCF e actDCF: nei GMM, queste linee sono praticamente sovrapposte, mentre per la SVM il distacco è marcato—segno inconfutabile di scarsa calibrazione degli score prodotti.

Queste differenze si spiegano guardando alla natura dei modelli e, in particolare, al significato degli score che producono. I GMM, essendo autentici modelli generativi probabilistici, calcolano per costruzione vere probabilità a posteriori tramite il teorema di Bayes. I loro output, ovvero i log-likelihood ratio, riflettono direttamente la probabilità che un dato osservato appartenga a una classe piuttosto che all’altra. Questo significa che la soglia di separazione teorica data da Bayes è effettivamente ottimale senza alcuna trasformazione o “calibrazione” successiva: il sistema è calibrato per natura. Il risultato pratico è che il gap tra minDCF (rischio “potenziale” del miglior threshold) e actDCF (rischio “reale” usando il threshold teorico) resta minimo, tipicamente inferiore al 4%.

Nel caso della regressione logistica, benché anche questo modello fornisca probabilità a posteriori esplicite attraverso la funzione sigmoide, diversi fattori possono degradare la calibrazione: regolarizzazione e scelte ottimizzative possono distorcere i valori numerici degli score, producendo minDCF e actDCF più distanti. In pratica, pur mantenendo un’interpretazione probabilistica, la regressione logistica risente di un certo scostamento dalla probabilità reale, che va raramente oltre il 10%.

La situazione cambia radicalmente con i modelli totalmente discriminativi come le SVM. Qui, gli score in uscita non sono vere probabilità ma piuttosto misure geometriche (distanze dall’iperpiano) o confidenze numeriche relative alla classificazione. Non esiste corrispondenza diretta tra il valore numerico dello score SVM e la probabilità che quell’oggetto appartenga ad una classe. Scegliere la soglia tramite la teoria di Bayes, senza una calibrazione specifica (ad esempio tramite Platt scaling o regressione isotonica), comporta l’uso di un threshold che è arbitrario rispetto alla scala degli score e spesso drasticamente fuori target rispetto al rischio reale. Ecco perché nei risultati la distanza tra minDCF e actDCF nelle SVM è drammatica e spesso superiore a quella di qualsiasi altro modello osservato.

Questo fenomeno mette in evidenza, in modo molto concreto, un principio fondamentale nella scelta e valutazione dei modelli di classificazione: dove la calibrazione probabilistica degli output è necessaria all’applicazione finale, solo i modelli con interpretazione probabilistica naturale (come GMM e, se ben regolarizzata, la regressione logistica) possono garantire la coerenza tra rischio teorico e rischio reale. I modelli puramente discriminativi, invece, necessitano di step aggiuntivi di calibrazione per poter essere usati come veri “probability scorers”.

Tra le implicazioni metodologiche si conferma quindi che:
- I modelli probabilistici generativi come i GMM sono particolarmente potenti su dataset complessi e multimodali, offrendo una calibrazione naturale e score affidabili.
- I modelli discriminativi (come SVM), senza adeguata calibrazione, rischiano di fornire ottime superfici di separazione ma stime di rischio operative fuorvianti.
- La scelta del modello migliore dipende non solo dalla performance in classificazione “pura”, ma anche dalla natura del task: quando l’output probabilistico è cruciale, la disciplina probabilistica del modello diventa determinante.
- Esiste, inoltre, un bilanciamento da considerare tra capacità di modellazione, interpretabilità degli output e necessità di pre- o post-processing dei risultati numerici.

Alla luce di queste considerazioni, il GMM a 8 componenti diagonali emerge come campione non solo per prestazioni pure, ma anche per la qualità intrinseca dei suoi score in termini di affidabilità e immediatezza d’uso nelle applicazioni pratiche. In contesti reali e complessi, questa proprietà pone i modelli probabilistici generativi come scelta di riferimento e come benchmark metodologico per il confronto e l’evoluzione di futuri sistemi di pattern recognition.