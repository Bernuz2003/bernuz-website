### Teoria delle Decisioni Bayesiane e Analisi del Rischio nei Classificatori Gaussiani

Questo laboratorio esplora la teoria delle decisioni bayesiane applicata alla classificazione di impronte digitali, analizzando tre varianti di classificatori Gaussiani multivariati (MVG Full, Naive Bayes, Tied Covariance) sotto diverse condizioni operative. L'obiettivo è comprendere come i parametri dell'applicazione (prior, costi) influenzino le performance e la calibrazione dei modelli.

#### Fondamenti Teorici della Teoria delle Decisioni Bayesiane

**Il Framework delle Decisioni Bayesiane**

La teoria delle decisioni bayesiane fornisce un framework matematico ottimale per la classificazione in presenza di incertezza. Per un problema di classificazione binaria, definiamo:

- $\pi = P(C_1)$: probabilità a priori della classe 1 (Genuine)
- $1-\pi = P(C_0)$: probabilità a priori della classe 0 (Fake)
- $C_{fn}$: costo di un falso negativo (classificare Genuine come Fake)
- $C_{fp}$: costo di un falso positivo (classificare Fake come Genuine)

**Rischio Bayesiano e DCF (Detection Cost Function)**

Il rischio bayesiano $R$ per una decisione $\hat{c}$ è definito come:

$$
R = \pi \cdot C_{fn} \cdot P_{fn} + (1-\pi) \cdot C_{fp} \cdot P_{fp}
$$

dove:
- $P_{fn} = P(\hat{c} = 0 | C = 1)$: probabilità di falso negativo
- $P_{fp} = P(\hat{c} = 1 | C = 0)$: probabilità di falso positivo

La **Detection Cost Function normalizzata** è:

$$
DCF = \frac{R}{\min(\pi \cdot C_{fn}, (1-\pi) \cdot C_{fp})}
$$

**Log-Likelihood Ratio e Soglia Ottimale**

Il **Log-Likelihood Ratio (LLR)** è definito come:

$$
\text{LLR}(x) = \log \frac{p(x|C_1)}{p(x|C_0)} = \log p(x|C_1) - \log p(x|C_0)
$$

La soglia di decisione ottimale secondo il criterio di Bayes è:

$$
\tau = -\log \frac{\pi \cdot C_{fn}}{(1-\pi) \cdot C_{fp}}
$$

La regola di decisione ottimale è: $\hat{c} = 1$ se $\text{LLR}(x) > \tau$, altrimenti $\hat{c} = 0$.

```python
def compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp):
    """Compute optimal Bayes decisions from LLR."""
    th = -np.log((prior * Cfn) / ((1 - prior) * Cfp))
    return np.int32(llr > th)
```

**Effective Prior**

L'**effective prior** $\tilde{\pi}$ rappresenta il prior equivalente che, con costi unitari, produce la stessa soglia di decisione:

$$
\tilde{\pi} = \frac{\pi \cdot C_{fn}}{\pi \cdot C_{fn} + (1-\pi) \cdot C_{fp}}
$$

```python
def compute_effective_prior(prior, Cfn, Cfp):
    """Compute effective prior from application parameters."""
    return (prior * Cfn) / (prior * Cfn + (1 - prior) * Cfp)
```

#### Classificatori Gaussiani Multivariati

**MVG Full Covariance**

Il classificatore MVG completo modella ogni classe con una distribuzione Gaussiana multivariata con matrice di covarianza piena:

$$
p(\mathbf{x}|C_i) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}_i|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_i)^T\boldsymbol{\Sigma}_i^{-1}(\mathbf{x}-\boldsymbol{\mu}_i)\right)
$$

La log-densità è:

$$
\log p(\mathbf{x}|C_i) = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\log|\boldsymbol{\Sigma}_i| - \frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_i)^T\boldsymbol{\Sigma}_i^{-1}(\mathbf{x}-\boldsymbol{\mu}_i)
$$

```python
def logpdf_GAU_ND(X, mu, Sigma):
    """Compute log-density for multivariate Gaussian."""
    M = X.shape[0]
    XC = X - mu
    invS = np.linalg.inv(Sigma)
    log_det = np.linalg.slogdet(Sigma)[1]
    quad = np.sum(XC * (invS @ XC), axis=0)
    return -0.5 * (M * np.log(2*np.pi) + log_det + quad)
```

**Naive Bayes Gaussian**

Il classificatore Naive Bayes assume indipendenza condizionale tra le feature, risultando in matrici di covarianza diagonali:

$$
\boldsymbol{\Sigma}_i = \text{diag}(\sigma_{i1}^2, \sigma_{i2}^2, \ldots, \sigma_{id}^2)
$$

Questo riduce il numero di parametri da $O(d^2)$ a $O(d)$, migliorando la robustezza in caso di dati limitati.

**Tied Covariance**

Il modello Tied assume che entrambe le classi condividano la stessa matrice di covarianza:

$$
\boldsymbol{\Sigma} = \frac{N_0 \boldsymbol{\Sigma}_0 + N_1 \boldsymbol{\Sigma}_1}{N_0 + N_1}
$$

dove $N_i$ è il numero di campioni nella classe $i$.

#### Metodologia Sperimentale e Setup

**Dataset e Suddivisione**

Il dataset contiene 6000 campioni (2990 Fake, 3010 Genuine) con 6 feature ciascuno. La suddivisione è 2/3 training (4000 campioni) e 1/3 validation (2000 campioni).

**Applicazioni Analizzate**

Vengono considerate cinque applicazioni con diversi parametri operativi:

| Applicazione | $\pi$ | $C_{fn}$ | $C_{fp}$ | $\tilde{\pi}$ | Descrizione |
|-------------|-------|----------|----------|---------------|-------------|
| Balanced | 0.5 | 1.0 | 1.0 | 0.500 | Costi e prior uniformi |
| High Genuine Prior | 0.9 | 1.0 | 1.0 | 0.900 | Most users legit |
| High Fake Prior | 0.1 | 1.0 | 1.0 | 0.100 | Most users impostors |
| Strong Security | 0.5 | 1.0 | 9.0 | 0.100 | High FP cost |
| Ease of Use | 0.5 | 9.0 | 1.0 | 0.900 | High FN cost |

L'analisi si concentra su tre casi principali: $\tilde{\pi} = 0.1$ (Security-focused), $\tilde{\pi} = 0.5$ (Balanced), e $\tilde{\pi} = 0.9$ (User-friendly).

#### Risultati Sperimentali e Analisi Dettagliata

**Performance dei Classificatori per Applicazione**

| Applicazione | Classificatore | actDCF | minDCF | Calibration Loss | Loss % |
|-------------|----------------|--------|--------|------------------|---------|
| **π̃ = 0.1 (Security)** | Naive Bayes | 0.302 | **0.257** | 0.045 | 17.6% |
|                        | MVG | 0.305 | 0.263 | 0.042 | 16.1% |
|                        | Tied | 0.406 | 0.363 | 0.043 | **11.9%** |
| **π̃ = 0.5 (Balanced)** | MVG | 0.140 | **0.130** | 0.010 | 7.5% |
|                        | Naive Bayes | 0.144 | 0.131 | 0.013 | 9.8% |
|                        | Tied | 0.186 | 0.181 | 0.005 | **2.6%** |
| **π̃ = 0.9 (User-friendly)** | MVG | 0.400 | **0.342** | 0.058 | 16.9% |
|                           | Naive Bayes | 0.389 | 0.351 | 0.038 | 10.9% |
|                           | Tied | 0.463 | 0.442 | 0.020 | **4.6%** |

**Interpretazione Teorica dei Risultati**

1. **Inconsistenza del Ranking**: Il ranking dei modelli varia significativamente tra le applicazioni:
   - Per $\tilde{\pi} = 0.1$: Naive Bayes > MVG > Tied
   - Per $\tilde{\pi} = 0.5$ e $\tilde{\pi} = 0.9$: MVG > Naive Bayes > Tied

   Questo fenomeno è spiegabile attraverso il **bias-variance tradeoff**:
   - **Naive Bayes**: Modello più semplice (bias elevato, varianza bassa), robusto in condizioni estreme
   - **MVG Full**: Modello complesso (bias basso, varianza alta), ottimale per condizioni bilanciate
   - **Tied**: Compromesso intermedio, ma vincoli troppo restrittivi per questo dataset

2. **Dipendenza dal Prior Effettivo**: Le performance relative cambiano al variare di $\tilde{\pi}$ a causa della **geometria delle superfici di decisione**. Per prior estremi, modelli più semplici tendono a generalizzare meglio.

**Analisi della Calibrazione**

La **calibration loss** è definita come:
$$
\text{Calibration Loss} = \text{actDCF} - \text{minDCF}
$$

Rappresenta la perdita dovuta all'uso della soglia basata sui parametri dell'applicazione invece della soglia ottimale empirica.

**Osservazioni sulla Calibrazione:**

1. **Tied Covariance**: Sistematicamente il meglio calibrato (loss medio 6.4%)
2. **MVG Full**: Migliori performance assolute ma calibrazione variabile
3. **Naive Bayes**: Calibrazione intermedia ma più stabile

La migliore calibrazione del modello Tied è dovuta alla **regolarizzazione implicita** introdotta dal vincolo di covarianza condivisa, che produce LLR più stabili.

#### Bayes Error Plots e Analisi Visuale

![](/mlLabs_screens/04_BDM/bayes_error_plots.png)

I **Bayes Error Plots** mostrano actDCF e minDCF in funzione del prior log-odds $\log\frac{\tilde{\pi}}{1-\tilde{\pi}}$ nell'intervallo $[-4, +4]$.

**Interpretazione dei Plot:**

1. **MVG Full (Plot sinistra)**:
   - minDCF più basso nella regione centrale ($\tilde{\pi} \approx 0.5$)
   - Gap significativo tra actDCF e minDCF agli estremi
   - Forma a "U" caratteristica di un classificatore ben calibrato al centro

2. **Tied Covariance (Plot centrale)**:
   - Performance inferiori in termini di minDCF
   - Gap minore tra actDCF e minDCF (migliore calibrazione)
   - Andamento più uniforme su tutto il range

3. **Naive Bayes (Plot destra)**:
   - Performance competitive con MVG nella regione $\tilde{\pi} < 0.5$
   - Degrado rapido per $\tilde{\pi}$ elevati
   - Calibrazione variabile

**Punti Markati**: I cerchi rossi e blu evidenziano le tre applicazioni focus ($\tilde{\pi} = 0.1, 0.5, 0.9$), mostrando chiaramente come il ranking dei modelli si inverta.

#### Analisi Statistica Approfondita

**Test di Significatività delle Differenze**

Per valutare la significatività statistica delle differenze osservate, consideriamo l'**intervallo di confidenza** per il DCF. Dato che il DCF è stimato su 2000 campioni, l'errore standard è approssimativamente:

$$
SE_{DCF} \approx \sqrt{\frac{DCF(1-DCF)}{N}}
$$

Le differenze tra MVG e Naive Bayes nell'applicazione Balanced (0.130 vs 0.131) sono nell'ordine dell'errore di stima, suggerendo performance equivalenti.

**Decomposizione del Rischio Bayesiano**

Il rischio può essere decomposto in:
$$
R = \underbrace{\pi \cdot C_{fn} \cdot P_{fn}}_{\text{Costo FN}} + \underbrace{(1-\pi) \cdot C_{fp} \cdot P_{fp}}_{\text{Costo FP}}
$$

Per l'applicazione Security-focused ($\tilde{\pi} = 0.1$):
- Il termine dominante è il costo FP: $(1-0.1) \times 1.0 \times P_{fp} = 0.9 \times P_{fp}$
- I modelli ottimizzano principalmente la riduzione di $P_{fp}$

Per l'applicazione User-friendly ($\tilde{\pi} = 0.9$):
- Il termine dominante è il costo FN: $0.9 \times 1.0 \times P_{fn} = 0.9 \times P_{fn}$
- I modelli ottimizzano principalmente la riduzione di $P_{fn}$

#### Implicazioni Teoriche e Pratiche

**1. Selezione del Modello Context-Aware**

I risultati dimostrano che **non esiste un modello universalmente ottimale**. La scelta deve considerare:
- **Condizioni operative**: prior e costi dell'applicazione
- **Requisiti di calibrazione**: criticità della stima di confidenza
- **Robustezza**: stabilità delle performance al variare delle condizioni

**2. Trade-off Bias-Variance nei Classificatori Gaussiani**

L'esperimento illustra chiaramente il trade-off:
- **Alta varianza (MVG)**: Eccellente in condizioni nominali, degrada agli estremi
- **Alto bias (Naive Bayes)**: Robusto ma sub-ottimale in condizioni ideali
- **Compromesso (Tied)**: Mediocre ma stabile

**3. Importanza della Calibrazione**

La calibrazione è cruciale per:
- **Sistemi multi-stadio**: dove i scores vengono utilizzati downstream
- **Decision making**: quando la confidenza della predizione è rilevante
- **Fusion di modelli**: per combinare appropriatamente i contributi

#### Analisi della Complessità Computazionale

**Complessità di Training:**
- **MVG Full**: $O(d^3)$ per l'inversione di $\boldsymbol{\Sigma}_i$
- **Tied**: $O(d^3)$ per una singola inversione di $\boldsymbol{\Sigma}$
- **Naive Bayes**: $O(d)$ solo calcoli diagonali

**Complessità di Inference:**
- **MVG Full**: $O(d^2)$ per il termine quadratico
- **Tied**: $O(d^2)$ ma con matrice pre-computata
- **Naive Bayes**: $O(d)$ calcoli elementari

#### Conclusioni e Raccomandazioni

**Sintesi dei Risultati:**

1. **Performance Globali**: MVG Full emerge come migliore compromesso complessivo (minDCF medio: 0.245)
2. **Calibrazione**: Tied Covariance è sistematicamente il meglio calibrato (loss medio: 6.4%)
3. **Robustezza**: Naive Bayes mostra la migliore robustezza in condizioni estreme

**Raccomandazioni Pratiche:**

- **Applicazioni balanced**: Utilizzare MVG Full per performance ottimali
- **Applicazioni extreme prior**: Considerare Naive Bayes per robustezza
- **Sistemi che richiedono calibrazione**: Preferire Tied Covariance
- **Risorse computazionali limitate**: Naive Bayes offre il miglior trade-off

**Sviluppi Futuri:**

1. **Regolarizzazione**: Esplorare tecniche di regolarizzazione per MVG
2. **Calibrazione post-hoc**: Implementare metodi di calibrazione (Platt scaling, isotonic regression)
3. **Ensemble methods**: Combinare i tre approcci per sfruttare i loro punti di forza complementari
4. **Analisi di sensibilità**: Studiare la robustezza ai parametri mal-specificati dell'applicazione

Questo studio dimostra l'importanza di una valutazione multi-dimensionale dei classificatori, considerando non solo le performance assolute ma anche calibrazione, robustezza e adeguatezza al contesto applicativo specifico.