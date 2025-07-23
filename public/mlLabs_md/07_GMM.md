### Argomento 7: Gaussian Mixture Models – Modellare la Complessità del Mondo Reale

Nei capitoli precedenti abbiamo esplorato un'ampia gamma di classificatori. Siamo partiti da modelli semplici come la singola Gaussiana (MVG) e abbiamo progressivamente aumentato la complessità, passando per modelli discriminativi lineari (Regolazione Logistica) e non lineari (SVM con kernel). Ora, torniamo ai modelli generativi, ma con uno strumento molto più potente e flessibile: i **Gaussian Mixture Models (GMM)**.

L'assunzione che i dati di una classe provengano da una singola distribuzione Gaussiana è spesso una semplificazione eccessiva. Nel mondo reale, i dati possono presentare sotto-cluster, modalità multiple o forme complesse che una singola elissoide non può catturare. I GMM superano questo limite modellando la distribuzione di probabilità di una classe come una **sovrapposizione pesata di più componenti Gaussiane**. È come usare un set di "pennelli" a forma di campana di varie dimensioni e orientamenti per "dipingere" la forma complessa della distribuzione dei dati.

#### La Matematica: Sommare le Gaussiane per Creare la Complessità

Un GMM descrive la densità di probabilità di un campione **x** come una somma pesata di *M* componenti Gaussiane:

$$
p(\mathbf{x}) = \sum_{m=1}^{M} w_m \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_m, \boldsymbol{\Sigma}_m)
$$

Dove per ogni componente *m*:
*   $$ w_m $$ è il peso (o probabilità a priori) della componente, con il vincolo che $$ \sum_{m=1}^{M} w_m = 1 $$.
*   $$ \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_m, \boldsymbol{\Sigma}_m) $$ è una normale funzione di densità Gaussiana multivariata con media $$ \boldsymbol{\mu}_m $$ e matrice di covarianza $$ \boldsymbol{\Sigma}_m $$.

Per la classificazione, addestriamo un GMM separato per ogni classe. Quindi, per un dato campione **x**, calcoliamo la verosimiglianza sotto il modello della classe 0 e della classe 1 e, come sempre, prendiamo la nostra decisione basandoci sul Log-Likelihood Ratio (LLR):

$$
\text{LLR}(\mathbf{x}) = \log p(\mathbf{x} | \text{GMM}_{\text{classe 1}}) - \log p(\mathbf{x} | \text{GMM}_{\text{classe 0}})
$$

L'addestramento di un GMM è un compito più complesso rispetto a una singola Gaussiana e viene tipicamente eseguito con l'algoritmo **Expectation-Maximization (EM)**. L'EM itera tra due passi fino a convergenza:
1.  **E-step (Expectation)**: Si calcolano le "responsabilità" di ogni componente Gaussiana per ogni punto del dataset. In pratica, si stima quanto è probabile che un dato campione sia stato generato da ciascuna delle *M* componenti.
2.  **M-step (Maximization)**: Si aggiornano i parametri di ogni componente (pesi, medie, covarianze) usando i dati pesati per le responsabilità calcolate nel passo E.

Per evitare che l'algoritmo finisca in un minimo locale di scarsa qualità, viene spesso utilizzato un approccio di inizializzazione chiamato **LBG (Linde-Buzo-Gray)**, che parte da un singolo centroide e lo "divide" iterativamente per creare il numero desiderato di componenti.

Nel nostro codice, questo complesso processo è incapsulato nella funzione `train_GMM_LBG_EM`.

```python
def analyze_gmm_components(DTR, LTR, DVAL, LVAL, cov_type='full', max_components=32, target_prior=0.1):
    # ...
    for num_components in component_values:
        # Addestra un GMM per la classe 0
        gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], num_components, 
                                covType=cov_type, verbose=False, psiEig=0.01)
        
        # Addestra un GMM per la classe 1
        gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], num_components, 
                                covType=cov_type, verbose=False, psiEig=0.01)
        
        # Calcola le verosimiglianze e gli score LLR
        ll0 = logpdf_GMM(DVAL, gmm0)
        ll1 = logpdf_GMM(DVAL, gmm1)
        llr_scores = ll1 - ll0
        
        # Valuta le performance
        minDCF = bayesRisk.compute_minDCF_binary_fast(llr_scores, LVAL, target_prior, 1.0, 1.0)
        # ...
    return results, best_result
```

#### Analisi dei Risultati: il Numero di Componenti è la Chiave

Abbiamo testato i GMM con un numero crescente di componenti (da 1 a 32) e con due tipi di matrici di covarianza: **full** (complete) e **diagonal** (diagonali).

![](/mlLabs_screens/07_GMM/gmm_analysis_comparison.png)

Il grafico "GMM Analysis Comparison" riassume in modo eccellente i risultati di questi esperimenti.
*   **Andamento del minDCF (in alto a sinistra)**: Entrambi i modelli mostrano un andamento a "U". Le performance migliorano all'aumentare del numero di componenti, raggiungono un punto ottimale e poi peggiorano. Questo è il classico trade-off tra *underfitting* e *overfitting*.
    *   Con poche componenti, il modello è troppo semplice per catturare la vera struttura dei dati.
    *   Con troppe componenti, il modello diventa eccessivamente complesso e inizia a modellare il rumore specifico del training set, perdendo capacità di generalizzazione.
*   **GMM a Covarianza Completa (linea blu)**: Raggiunge il suo punto di minima spesa (minDCF) con **16 componenti**, ottenendo un eccellente **minDCF = 0.1631**.
*   **GMM a Covarianza Diagonale (linea rossa)**: A sorpresa, questo modello più semplice fa ancora meglio. Raggiunge il suo minimo con sole **8 componenti**, ottenendo un **minDCF di 0.1463**, il miglior risultato visto finora.

Perché un modello più vincolato (diagonale) supera un modello più flessibile (completo)? La spiegazione più probabile è che l'assunzione di indipendenza tra le feature a livello di componente agisca come una forma molto efficace di **regolarizzazione**. Forzando le covarianze a essere diagonali, si impedisce al modello di apprendere correlazioni spurie dai dati, rendendolo più robusto, specialmente quando il numero di campioni per componente diminuisce. Questo suggerisce che le principali modalità di variazione dei dati sono allineate con gli assi delle feature.

*   **Calibrazione (in basso a sinistra)**: Entrambi i modelli GMM dimostrano una **calibrazione eccellente**. Il gap tra actDCF e minDCF è sempre molto basso (sotto il 4%), come ci si aspetta da modelli generativi ben addestrati. I loro punteggi sono affidabili.

#### Il Verdetto Finale: Tutti i Classificatori a Confronto

Dopo un lungo percorso, siamo pronti per la resa dei conti finale. Abbiamo messo a confronto i migliori modelli di ogni famiglia: GMM, SVM e Regolazione Logistica.

| Classificatore              | Miglior minDCF | Calibrazione | Configurazione Ottimale                  |
| :-------------------------- | :------------- | :----------- | :--------------------------------------- |
| **GMM Diagonale**           | **0.1463**     | Buona        | 8 componenti                             |
| **GMM Completa**            | 0.1631         | Buona        | 16 componenti                            |
| **SVM (RBF)**               | 0.2391         | Pessima      | C=1.0, γ=1.0                             |
| **Regolazione Logistica**   | 0.3611         | Discreta     | Standard, λ=0.01                         |

Il **GMM con 8 componenti diagonali emerge come il vincitore indiscusso**. Non solo ottiene il più basso rischio di classificazione, ma lo fa con punteggi ben calibrati.

![](/mlLabs_screens/07_GMM/all_classifiers_-_bayes_error_plot_bayes_error_plot.png)

Il grafico "All Classifiers - Bayes Error Plot" fornisce una visione panoramica e definitiva.
*   Le curve del minDCF (linee continue) dei modelli GMM (rossa e blu) sono nettamente più basse di quelle della Regolazione Logistica (verde) e della SVM (viola) su quasi tutto lo spettro di applicazioni.
*   Il gap tra le linee continue (minDCF) e tratteggiate (actDCF) evidenzia la differenza di calibrazione: per i GMM le linee sono quasi sovrapposte, mentre per la SVM sono drammaticamente separate, a conferma della sua scarsa calibrazione.

In conclusione, l'analisi ha dimostrato che per questo dataset di dati biometrici, la cui distribuzione è evidentemente complessa e non catturabile da un singolo modello Gaussiano, i **Gaussian Mixture Models sono la soluzione superiore**. La loro capacità di modellare densità arbitrarie con una miscela di componenti, unita a una buona calibrazione, li rende l'approccio più potente e affidabile, decretando il **GMM a 8 componenti diagonali** come il campione assoluto di questa competizione.
