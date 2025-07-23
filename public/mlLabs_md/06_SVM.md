### Argomento 6: Support Vector Machine – La Ricerca del Margine Massimo

Dopo aver esplorato modelli generativi e la Regolazione Logistica, ci addentriamo in un'altra classe di modelli discriminativi: le **Support Vector Machines (SVM)**. L'idea fondamentale delle SVM è diversa da quella della Regolazione Logistica. Invece di modellare la probabilità, le SVM cercano di trovare l'iperpiano di separazione che massimizza la "distanza" o il **margine** tra i punti più vicini delle due classi (i cosiddetti *vettori di supporto*). Questo approccio, basato sulla massimizzazione del margine, mira a creare un classificatore che sia il più robusto possibile al rumore e che generalizzi bene su dati nuovi.

#### La Matematica: Dalla Hinge Loss alla Formulazione Duale

L'obiettivo di una SVM lineare è trovare i parametri **w** e *b* che risolvono il seguente problema di ottimizzazione (formulazione primale):

$$
\min_{\mathbf{w}, b} \frac{1}{2} ||\mathbf{w}||^2 + C \sum_{i=1}^N \max(0, 1 - z_i(\mathbf{w}^T\mathbf{x}_i + b))
$$

Dove $$z_i \in \{-1, +1\}$$ sono le etichette di classe. Il primo termine, $$ \frac{1}{2} ||\mathbf{w}||^2 $$, è legato alla massimizzazione del margine. Il secondo termine è la **Hinge Loss**, una funzione di costo che penalizza i punti che si trovano dal lato sbagliato del margine o all'interno di esso. L'iperparametro **C** funge da trade-off:
*   Un **C piccolo** aumenta la regolarizzazione, privilegiando un margine più largo anche a costo di classificare erroneamente qualche punto (SVM *soft-margin*).
*   Un **C grande** diminuisce la regolarizzazione, cercando di classificare correttamente più punti possibili, anche a costo di un margine più stretto.

Per gestire modelli non lineari, si passa alla **formulazione duale** del problema, che viene espressa in termini di moltiplicatori di Lagrange $$ \alpha_i $$, uno per ogni punto di training. L'obiettivo diventa massimizzare:

$$
\mathcal{L}_D(\boldsymbol{\alpha}) = \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j z_i z_j (\mathbf{x}_i^T \mathbf{x}_j)
$$

con i vincoli $$ 0 \le \alpha_i \le C $$. Questa formulazione è cruciale perché la dipendenza dai dati avviene solo tramite il prodotto scalare $$ \mathbf{x}_i^T \mathbf{x}_j $$, il che apre la porta al "kernel trick".

Nel codice, questa ottimizzazione viene risolta usando un solutore come L-BFGS-B.

```python
def train_dual_SVM_linear(DTR, LTR, C, K=1.0):
    # Converte le label in {-1, +1}
    ZTR = LTR * 2.0 - 1.0
    # Estende i dati per includere il bias (parametro K)
    DTR_EXT = np.vstack([DTR, np.ones((1, DTR.shape[1])) * K])
    # Costruisce la matrice Hessiana H del problema duale
    H = np.dot(DTR_EXT.T, DTR_EXT) * vcol(ZTR) * vrow(ZTR)

    # Funzione obiettivo duale da minimizzare (negativo di L_D)
    def fOpt(alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - np.ones(alpha.size)
        return loss, grad

    # Ottimizzazione con L-BFGS-B
    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(
        fOpt, np.zeros(DTR_EXT.shape[1]), 
        bounds=[(0, C) for i in LTR], 
        factr=1e7, pgtol=1e-5
    )
    # ... calcolo di w e b da alphaStar ...
    return w, b, primalLoss_val, dualLoss_val, duality_gap
```

#### Esperimento 1: SVM Lineare – Un Inizio Sottotono

Abbiamo testato una SVM lineare su dati originali e centrati, variando il parametro C.

<p align="center">
  <img src="/mlLabs_screens/06_SVM/linear_svm_Original_Data.png" width="48%" />
  <img src="/mlLabs_screens/06_SVM/linear_svm_Centered_Data.png" width="48%" />
</p>

Dai grafici "Linear SVM: Original Data", emergono diversi punti:
*   **Performance (Grafico a Sinistra)**: Il minDCF (curva blu) si stabilizza rapidamente intorno a **0.3582** per un'ampia gamma di valori di C, con un ottimo per **C = 0.1**. Questo valore è peggiore di quello ottenuto con la Regolazione Logistica Lineare, suggerendo che per questo problema la Hinge Loss della SVM sia meno efficace della Cross-Entropy Loss.
*   **Calibrazione (Grafico al Centro)**: La calibrazione è **pessima**. Il gap tra actDCF e minDCF è enorme (oltre 0.15 per il C ottimale), indicando che i punteggi grezzi della SVM sono molto lontani dall'essere probabilità logaritmiche affidabili. Questo è un difetto noto delle SVM, che sono focalizzate sul margine e non sulla calibrazione dei punteggi.
*   **Centratura dei Dati**: Confrontando con i risultati su dati centrati (vedi grafico "Linear SVM: Centered Data"), le performance sono quasi identiche. Questo dimostra che, a differenza di altri modelli, la SVM lineare è intrinsecamente robusta alla traslazione dei dati.

#### Esperimento 2: SVM con Kernel – Entrare nella Non Linearità

Il vero potere delle SVM si scatena con il **kernel trick**. Sostituendo il prodotto scalare $$ \mathbf{x}_i^T \mathbf{x}_j $$ nella formulazione duale con una funzione kernel non lineare $$ k(\mathbf{x}_i, \mathbf{x}_j) $$, possiamo implicitamente mappare i dati in uno spazio a dimensionalità molto più alta e trovare confini di separazione non lineari.

```python
# La funzione di training per SVM con kernel è molto simile,
# ma costruisce la matrice H usando la funzione kernel
def train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps=1.0):
    ZTR = LTR * 2.0 - 1.0
    # Calcola la matrice di Gram usando la funzione kernel
    K = kernelFunc(DTR, DTR) + eps
    H = vcol(ZTR) * vrow(ZTR) * K
    # ... il resto dell'ottimizzazione è identico
```

##### SVM Polinomiale (Grado 2)

Usando un kernel polinomiale di grado 2, $$ k(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i^T \mathbf{x}_j + 1)^2 $$, permettiamo alla SVM di trovare confini quadratici.

![](/mlLabs_screens/06_SVM/polynomial_svm_analysis.png)

Come si vede nel grafico "Polynomial SVM analysis", le performance migliorano drasticamente. Il **minDCF scende a 0.2455**, un risultato quasi identico a quello della Regolazione Logistica Quadratica. Questo rafforza la conclusione che le interazioni quadratiche tra le feature sono fondamentali per questo dataset. Tuttavia, la calibrazione rimane molto scarsa.

##### SVM con Kernel RBF (Radial Basis Function)

Il kernel RBF, $$ k(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma ||\mathbf{x}_i - \mathbf{x}_j||^2) $$, è ancora più potente, poiché mappa implicitamente i dati in uno spazio a dimensionalità infinita. Richiede però di ottimizzare due iperparametri: **C** e **γ** (che controlla la "larghezza" delle funzioni base). Abbiamo eseguito una grid search su 44 combinazioni.

![](/mlLabs_screens/06_SVM/rbf_svm_grid_search.png)

I risultati, visualizzati nel complesso grafico "RBF SVM Grid Search", sono eccezionali:
*   **La Heatmap (in basso a destra)** mostra chiaramente come le performance (minDCF) dipendano fortemente dalla combinazione di C e γ. La regione più scura (performance migliori) si trova per valori intermedi di γ e valori alti di C.
*   Il punto ottimale, marcato con una stella rossa, è stato trovato per **γ = 0.135** e **C = 32.0**.
*   In questa configurazione, la SVM RBF ha raggiunto un **minDCF di 0.1735**, il miglior risultato ottenuto finora in tutta l'analisi, superando nettamente tutti i modelli precedenti.

#### Conclusioni Finali: il Trionfo della SVM Non Lineare

La gerarchia delle performance all'interno della famiglia SVM è chiara e inequivocabile:
1.  **RBF SVM (minDCF = 0.1735)**: Il vincitore assoluto. La sua capacità di modellare confini di separazione estremamente complessi si è rivelata la chiave per ottenere le massime performance su questo dataset.
2.  **Polynomial SVM (minDCF = 0.2455)**: Molto potente, conferma l'importanza delle non linearità quadratiche.
3.  **Linear SVM (minDCF = 0.3582)**: Nettamente inferiore, dimostrando che un confine lineare è un'approssimazione troppo grossolana per questo problema.

L'analisi delle SVM ha portato alla scoperta del modello più performante finora. La SVM con kernel RBF, grazie alla sua incredibile flessibilità, ha stabilito un nuovo benchmark di **minDCF = 0.1735**. Tuttavia, questo potere ha un costo: tutti i modelli SVM testati hanno mostrato una **calibrazione estremamente scarsa**. Questo significa che, sebbene siano eccellenti nel separare le classi, i loro punteggi non sono affidabili e richiederebbero un passo di calibrazione aggiuntivo prima di poter essere utilizzati in un'applicazione pratica che si basi su soglie di decisione variabili.