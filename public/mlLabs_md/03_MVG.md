### Argomento 3: Costruire Classificatori con Modelli Generativi Gaussiani (Analisi Dettagliata)

Nei capitoli precedenti abbiamo sezionato il problema da diverse angolazioni. Ora, mettiamo insieme tutti i pezzi per costruire dei classificatori completi e potenti. Abbandoniamo l'analisi di una singola feature alla volta per abbracciare una visione d'insieme, trattando ogni campione come un unico **vettore a 6 dimensioni**. L'idea è di costruire dei **modelli generativi**, cioè modelli che non si limitano a tracciare una linea di separazione tra le classi, ma che imparano la "ricetta" per generare i dati di ogni classe.

Nello specifico, assumeremo che i dati di ogni classe ("Genuine" e "Fake") provengano da una **distribuzione Gaussiana Multivariata (MVG)**. Imparando la forma di queste distribuzioni, potremo calcolare, per un nuovo campione, quale delle due classi ha la probabilità più alta di averlo generato.

#### La Base Matematica: la Gaussiana in Più Dimensioni

Una Gaussiana in M dimensioni non è più una semplice campana, ma un'**iper-ellissoide** nello spazio, la cui forma, orientamento e posizione sono descritti da due parametri:
*   Il **vettore delle medie** $$ \boldsymbol{\mu} $$, un vettore M-dimensionale che rappresenta il centro della nuvola di punti.
*   La **matrice di covarianza** $$ \boldsymbol{\Sigma} $$, una matrice M×M che descrive la forma della nuvola: la sua dispersione lungo ogni asse (le varianze sulla diagonale) e le correlazioni tra gli assi (i valori fuori diagonale).

La funzione di densità di probabilità (PDF) è una generalizzazione della formula 1D:
$$
p(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{\sqrt{(2\pi)^M |\det(\boldsymbol{\Sigma})|}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)
$$
Nel codice, il calcolo (in log-scala per stabilità numerica) è implementato così:

```python
def logpdf_GAU_ND(X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """Log‑density of N‑D Gaussian for all columns in X."""
    M = X.shape[0]
    XC = X - mu
    invS = np.linalg.inv(Sigma)
    log_det = np.linalg.slogdet(Sigma)[1]
    quad = np.sum(XC * (invS @ XC), axis=0)
    return -0.5 * (M * np.log(2*np.pi) + log_det + quad)
```

Per classificare, calcoliamo il **Log-Likelihood Ratio (LLR)**, che confronta la verosimiglianza di un campione **x** sotto il modello della classe 1 contro quello della classe 0. Se il risultato è positivo, scegliamo la classe 1, altrimenti la 0.

#### Tre "Sapori" di Classificatori Gaussiani: un Approfondimento

La vera differenza tra i modelli che testeremo risiede nelle assunzioni che facciamo sulla matrice di covarianza $$ \boldsymbol{\Sigma} $$. Questa non è una scelta puramente tecnica, ma una dichiarazione d'intenti sulla struttura che crediamo i nostri dati posseggano.

##### 1. Il Modello a Covarianza Completa (Full MVG): Massima Flessibilità

*   **L'Idea**: Questo è l'approccio più generale e potente. Si stima una media $$ \boldsymbol{\mu}_c $$ e una matrice di covarianza $$ \boldsymbol{\Sigma}_c $$ **distinte e complete** per ogni classe $$c$$.
*   **Le Implicazioni**: Permette a ogni classe di avere una propria forma ellissoidale, con un orientamento e una dimensione unici. È il modello ideale se le nuvole di punti delle due classi hanno "forme" diverse. Il rovescio della medaglia è che richiede di stimare molti parametri (per 6 feature, la matrice di covarianza ha $$ (6 \times 7) / 2 = 21 $$ parametri unici per classe), il che può portare a overfitting su dataset piccoli.
*   **Nel Codice**: La sua implementazione è diretta. Per ogni classe, si calcolano media e covarianza usando solo i campioni di quella classe.

```python
# Dentro il metodo train della classe GaussianClassifier
if self.classifier_type == "mvg":
    self.covs = {}
    for c in classes:
        Dc = D_tr[:, L_tr == c]  # Seleziona i dati della classe c
        self.mus[c] = mean_vector(Dc)
        self.covs[c] = covariance_matrix(Dc) # Calcola la covarianza completa
```

##### 2. Il Modello Naive Bayes: la Forza della Semplicità

*   **L'Idea**: Qui facciamo un'assunzione drastica: le feature sono **condizionalmente indipendenti** data la classe. Questo si traduce nell'imporre che le matrici di covarianza siano **diagonali**. Tutti gli elementi fuori dalla diagonale sono forzati a zero.
*   **Le Implicazioni**: Si ignora completamente ogni correlazione tra le feature. Graficamente, le ellissi di probabilità sono sempre allineate con gli assi coordinati. Il vantaggio è un crollo nel numero di parametri da stimare (solo 6 varianze per classe), rendendo il modello molto veloce e robusto all'overfitting. È un'ottima scelta quando le feature sono effettivamente poco correlate.
*   **Nel Codice**: Invece di `covariance_matrix`, si usa una funzione che calcola la covarianza e poi azzera gli elementi non diagonali.

```python
# Dentro il metodo train della classe GaussianClassifier
elif self.classifier_type == "naive_bayes":
    self.covs = {}
    for c in classes:
        Dc = D_tr[:, L_tr == c]
        self.mus[c] = mean_vector(Dc)
        self.covs[c] = diagonal_covariance_matrix(Dc) # Calcola solo le varianze
```

##### 3. Il Modello a Covarianza Condivisa (Tied MVG): la Via di Mezzo

*   **L'Idea**: Un compromesso intelligente. Si assume che le classi abbiano centri diversi ($$ \boldsymbol{\mu}_0 \neq \boldsymbol{\mu}_1 $$) ma che condividano **la stessa identica matrice di covarianza** ($$ \boldsymbol{\Sigma}_0 = \boldsymbol{\Sigma}_1 = \boldsymbol{\Sigma}_{\text{tied}} $$).
*   **Le Implicazioni**: Le ellissi di probabilità delle due classi hanno la stessa forma e lo stesso orientamento, sono solo traslate l'una rispetto all'altra. Questo modello ha meno parametri del Full MVG ma più del Naive Bayes, offrendo un buon bilanciamento. È la scelta giusta quando si crede che le due classi siano generate da processi con la stessa variabilità interna.
*   **Nel Codice**: La matrice condivisa viene calcolata come media pesata delle matrici di covarianza individuali, dove i pesi sono dati dalla numerosità di ciascuna classe.

```python
# Dentro il metodo train della classe GaussianClassifier
elif self.classifier_type == "tied":
    # ... calcolo medie e covarianze temporanee ...
    # Pooled covariance
    self.Sigma_tied = sum(Ns[c] * covs_temp[c] for c in classes) / N_tot
```

#### Risultati Sperimentali: la Prova del Nove

Dopo aver diviso i dati (4000 per il training, 2000 per la validazione), abbiamo messo alla prova i tre modelli.

##### Esperimento Chiave: Tutte le 6 Feature

Usando l'intero set di feature, i risultati parlano chiaro:

| Modello       | Tasso di Errore | Accuratezza |
| :------------ | :-------------- | :---------- |
| **MVG**       | 7.00%           | **93.00%**  |
| Naive Bayes   | 7.20%           | 92.80%      |
| Tied MVG      | 9.30%           | 90.70%      |

*   **Perché vince il Full MVG?** Con 4000 campioni, il modello ha dati a sufficienza per stimare le covarianze complete senza cadere in overfitting. La sua vittoria suggerisce che le strutture di covarianza delle classi "Genuine" e "Fake" sono effettivamente diverse, e catturare questa differenza è cruciale.
*   **Perché Naive Bayes va così bene?** La sua performance stellare (solo 0.2% in meno del modello completo) è una diretta conseguenza della bassissima correlazione tra le feature. L'analisi della matrice di correlazione mostra valori quasi nulli fuori dalla diagonale. L'assunzione di indipendenza, quindi, non era un'approssimazione grossolana, ma un riflesso fedele della realtà dei dati.
*   **Perché il Tied MVG perde terreno?** Il calo di oltre 2 punti percentuali è significativo. È la prova che l'assunzione di una covarianza condivisa è sbagliata per questo problema. Le nuvole di dati delle due classi non hanno la stessa forma, e forzarle in uno stampo comune danneggia la capacità del modello di discriminarle.

##### Altri Esperimenti e Conferme

*   **Potere delle Feature**: L'analisi sui sottoinsiemi ha confermato che le **feature 3 e 4** sono il motore della classificazione, raggiungendo da sole il 90.6% di accuratezza. Curiosamente, su questo sottospazio, il modello Tied ha funzionato molto bene, suggerendo che *in queste due dimensioni* le classi hanno una forma simile e differiscono principalmente per la posizione.
*   **L'Inutilità della PCA**: Tentare di pre-processare i dati con la PCA si è rivelato controproducente. L'accuratezza massima è leggermente scesa. Questo accade perché la PCA è agnostica rispetto alle classi e, nel cercare le direzioni di massima varianza, ha scartato piccole ma preziose informazioni che aiutavano la separazione, confermando che le feature originali erano già in una forma quasi ottimale per questo compito.

#### Sintesi Finale: Lezioni Imparate

L'analisi approfondita ci porta a una conclusione solida: il **modello generativo Gaussiano a covarianza completa (Full MVG), addestrato su tutte e 6 le feature originali, è l'approccio vincente**, con un'**accuratezza del 93.00%**.

La lezione più importante, però, va oltre il singolo risultato. Abbiamo visto come la scelta tra modelli apparentemente simili (Full, Tied, Naive) sia in realtà una profonda indagine sulla natura dei dati. Il successo del Full MVG e il fallimento del Tied ci dicono che le classi hanno variabilità intrinseche diverse. L'ottima performance del Naive Bayes ci svela la quasi totale assenza di correlazione tra le feature. In sintesi, la modellazione non è solo un esercizio di predizione, ma uno strumento per comprendere.