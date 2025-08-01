### Stima di Densità Gaussiana Univariata: Fondamenti Teorici e Analisi Approfondita

Dopo aver esplorato i dati con PCA e LDA, facciamo un passo indietro e proviamo a capire la natura fondamentale delle singole feature attraverso la **stima di densità di probabilità**. Questo approccio ci permette di costruire un modello generativo dei dati, fondamentale per comprendere la struttura intrinseca delle distribuzioni condizionate alle classi e per sviluppare classificatori probabilistici.

#### Fondamenti Teorici della Stima di Densità

**La Distribuzione Gaussiana Univariata**

Una distribuzione Gaussiana (o Normale) univariata è completamente caratterizzata da due parametri: la media $\mu$ (parametro di localizzazione) e la varianza $\sigma^2$ (parametro di scala). La funzione di densità di probabilità (PDF) è espressa dalla celebre formula:

$$
p(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

dove:
- Il termine $\frac{1}{\sqrt{2\pi\sigma^2}}$ è la costante di normalizzazione che garantisce che $\int_{-\infty}^{\infty} p(x|\mu,\sigma^2) dx = 1$
- L'esponente $-\frac{(x-\mu)^2}{2\sigma^2}$ determina la forma a campana caratteristica

**Stima di Massima Verosimiglianza (MLE)**

Data una collezione di osservazioni indipendenti $\mathcal{D} = \{x_1, x_2, \ldots, x_N\}$, la funzione di verosimiglianza è definita come:

$$
\mathcal{L}(\mu, \sigma^2) = \prod_{i=1}^{N} p(x_i | \mu, \sigma^2)
$$

Per ragioni computazionali, si lavora tipicamente con la log-verosimiglianza:

$$
\ell(\mu, \sigma^2) = \log \mathcal{L}(\mu, \sigma^2) = \sum_{i=1}^{N} \log p(x_i | \mu, \sigma^2)
$$

Sostituendo la PDF Gaussiana:

$$
\ell(\mu, \sigma^2) = -\frac{N}{2}\log(2\pi) - \frac{N}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{N}(x_i - \mu)^2
$$

**Derivazione degli Stimatori MLE**

Per trovare i parametri ottimali, deriviamo la log-verosimiglianza rispetto a $\mu$ e $\sigma^2$ e poniamo le derivate uguali a zero:

$$
\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^{N}(x_i - \mu) = 0
$$

$$
\frac{\partial \ell}{\partial \sigma^2} = -\frac{N}{2\sigma^2} + \frac{1}{2(\sigma^2)^2}\sum_{i=1}^{N}(x_i - \mu)^2 = 0
$$

Risolvendo queste equazioni otteniamo gli stimatori MLE:

$$
\hat{\mu}_{ML} = \frac{1}{N} \sum_{i=1}^{N} x_i = \bar{x}
$$

$$
\hat{\sigma}^2_{ML} = \frac{1}{N} \sum_{i=1}^{N} (x_i - \hat{\mu}_{ML})^2
$$

```python
def compute_mu_var_1D(x):
    """Compute ML estimates for mean and variance of 1D Gaussian."""
    mu = x.mean()
    var = x.var()  # NumPy usa denominatore N, non N-1
    return mu, var

def logpdf_GAU_1D(x, mu, var):
    """Compute log-density for 1D Gaussian distribution."""
    return -0.5 * np.log(2 * np.pi * var) - 0.5 * ((x - mu) ** 2) / var

def compute_ll_1D(x, mu, var):
    """Compute log-likelihood for 1D Gaussian."""
    return logpdf_GAU_1D(x, mu, var).sum()
```

#### Implementazione e Metodologia Sperimentale

Il nostro dataset consiste in 6000 campioni bilanciati (2990 "Fake", 3010 "Genuine") con 6 feature ciascuno. Per ogni combinazione classe-feature, stimiamo i parametri Gaussiani utilizzando la MLE:

```python
def analyze_univariate_gaussian_fits(data, labels, output_dir="gaussian_fits"):
    """Perform complete univariate Gaussian analysis."""
    class_names = ['Fake', 'Genuine']
    n_features = data.shape[0]
    ml_estimates = [[] for _ in range(2)]
    
    for class_idx, class_name in enumerate(class_names):
        class_data = data[:, labels == class_idx]
        
        for feat_idx in range(n_features):
            feature_data = class_data[feat_idx, :]
            
            # Compute ML estimates
            mu_ml, var_ml = compute_mu_var_1D(feature_data)
            ml_estimates[class_idx].append((mu_ml, var_ml))
            
            # Compute log-likelihood
            ll = compute_ll_1D(feature_data, mu_ml, var_ml)
            
            print(f"   Feature {feat_idx + 1}: μ={mu_ml:.4f}, σ²={var_ml:.4f}, LL={ll:.2f}")
    
    return ml_estimates
```

#### Analisi Dettagliata dei Risultati Sperimentali

**Parametri Stimati per Classe**

| Classe | Feature | Media ($\hat{\mu}$) | Varianza ($\hat{\sigma}^2$) | Log-Likelihood |
|--------|---------|---------------------|----------------------------|----------------|
| **Fake** | 1 | 0.0029 | 0.5696 | -3401.16 |
|          | 2 | 0.0187 | 1.4209 | -4767.77 |
|          | 3 | -0.6809 | 0.5500 | -3348.80 |
|          | 4 | 0.6708 | 0.5360 | -3310.43 |
|          | 5 | 0.0280 | 0.6801 | -3666.22 |
|          | 6 | -0.0058 | 0.7050 | -3720.12 |
| **Genuine** | 1 | 0.0005 | 1.4302 | -4809.55 |
|             | 2 | -0.0085 | 0.5783 | -3446.72 |
|             | 3 | 0.6652 | 0.5489 | -3368.25 |
|             | 4 | -0.6642 | 0.5533 | -3380.38 |
|             | 5 | -0.0417 | 1.3178 | -4686.29 |
|             | 6 | 0.0239 | 1.2870 | -4650.77 |

**Interpretazione Statistica Approfondita**

1. **Feature 3 e 4: Potere Discriminante Elevato**
   
   Queste feature mostrano un pattern di **separazione per inversione di segno**:
   - Feature 3: $\mu_{Fake} = -0.681$, $\mu_{Genuine} = 0.665$ (differenza = 1.346)
   - Feature 4: $\mu_{Fake} = 0.671$, $\mu_{Genuine} = -0.664$ (differenza = 1.335)
   
   La **distanza di Bhattacharyya** tra le distribuzioni, definita come:
   $$
   D_B = \frac{1}{8}(\mu_1 - \mu_2)^2 \left(\frac{2}{\sigma_1^2 + \sigma_2^2}\right) + \frac{1}{2}\ln\left(\frac{\sigma_1^2 + \sigma_2^2}{2\sqrt{\sigma_1^2\sigma_2^2}}\right)
   $$
   
   risulta particolarmente elevata per queste feature, indicando una buona separabilità.

2. **Feature 1, 5, 6: Discriminazione per Varianza**
   
   Queste feature mostrano medie simili tra le classi ma varianze molto diverse:
   - Feature 1: $\sigma^2_{Fake} = 0.570$, $\sigma^2_{Genuine} = 1.430$ (rapporto = 2.51)
   - Feature 5: $\sigma^2_{Fake} = 0.680$, $\sigma^2_{Genuine} = 1.318$ (rapporto = 1.94)
   - Feature 6: $\sigma^2_{Fake} = 0.705$, $\sigma^2_{Genuine} = 1.287$ (rapporto = 1.83)
   
   Questo suggerisce che la classe "Genuine" presenta maggiore variabilità interna in queste dimensioni.

3. **Feature 2: Caso Intermedio**
   
   Presenta sia differenza nelle medie che nelle varianze, ma con pattern opposto rispetto alle feature 1, 5, 6.

**Analisi della Log-Verosimiglianza**

La log-verosimiglianza fornisce una misura della "bontà di adattamento" del modello Gaussiano ai dati. Valori meno negativi indicano un migliore adattamento:

- **Migliori fit**: Feature 4 (Fake: -3310.43), Feature 3 (Fake: -3348.80)
- **Peggiori fit**: Feature 2 (Fake: -4767.77), Feature 1 (Genuine: -4809.55)

La correlazione inversa tra log-verosimiglianza e varianza è evidente: feature con varianze elevate tendono ad avere log-verosimiglianze più negative.

![](/mlLabs_screens/02_DE/all_features_summary-1.png)

#### Validazione dell'Assunzione Gaussiana

**Test di Goodness-of-Fit**

Per validare l'appropriatezza del modello Gaussiano, implementiamo diversi test statistici:

```python
def analyze_gaussian_fit_quality(feature_data, mu_ml, var_ml):
    """Analyze quality of Gaussian fit using various metrics."""
    
    # Compute log-likelihood
    ll = compute_ll_1D(feature_data, mu_ml, var_ml)
    
    # Skewness test (dovrebbe essere ≈ 0 per Gaussiana)
    mean_centered = feature_data - mu_ml
    skewness = np.mean((mean_centered / np.sqrt(var_ml)) ** 3)
    
    # Kurtosis test (dovrebbe essere ≈ 3 per Gaussiana)
    kurtosis = np.mean((mean_centered / np.sqrt(var_ml)) ** 4)
    
    # Test Kolmogorov-Smirnov semplificato
    sorted_data = np.sort(feature_data)
    n = len(sorted_data)
    empirical_cdf = np.arange(1, n+1) / n
    
    from scipy.stats import norm
    theoretical_cdf = norm.cdf(sorted_data, mu_ml, np.sqrt(var_ml))
    max_diff = np.max(np.abs(empirical_cdf - theoretical_cdf))
    
    return {
        'log_likelihood': ll,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'max_cdf_diff': max_diff
    }
```

**Interpretazione dei Momenti Statistici**

1. **Skewness (Asimmetria)**: Misura l'asimmetria della distribuzione
   - $S = 0$: distribuzione simmetrica (ideale per Gaussiana)
   - $S > 0$: coda destra più pesante
   - $S < 0$: coda sinistra più pesante

2. **Kurtosis (Curtosi)**: Misura la "pesantezza" delle code
   - $K = 3$: curtosi normale (Gaussiana)
   - $K > 3$: code più pesanti (leptocurtica)
   - $K < 3$: code più leggere (platicurtica)

#### Implicazioni per la Classificazione

**Costruzione di un Classificatore Naive Bayes**

I parametri stimati possono essere utilizzati per costruire un classificatore Naive Bayes Gaussiano. Per un nuovo campione $\mathbf{x} = [x_1, x_2, \ldots, x_6]^T$, la probabilità a posteriori della classe $k$ è:

$$
P(C_k | \mathbf{x}) \propto P(C_k) \prod_{j=1}^{6} p(x_j | C_k)
$$

dove $p(x_j | C_k) = \mathcal{N}(x_j; \hat{\mu}_{jk}, \hat{\sigma}^2_{jk})$ sono le densità Gaussiane stimate.

**Analisi del Potere Discriminante per Feature**

Calcoliamo il **rapporto di verosimiglianza** per ogni feature:

$$
\Lambda_j(x) = \frac{p(x_j | C_{Genuine})}{p(x_j | C_{Fake})}
$$

Le feature con maggiore potere discriminante saranno quelle per cui questo rapporto varia più drasticamente al variare di $x_j$.

**Considerazioni sulla Complessità del Modello**

Il modello Gaussiano univariato per classificazione binaria richiede:
- **12 parametri totali**: 2 parametri × 6 feature × 2 classi
- **Assunzione di indipendenza**: le feature sono considerate indipendenti (Naive Bayes)
- **Complessità computazionale**: $O(d)$ per la classificazione di un singolo campione

#### Limitazioni e Sviluppi Futuri

**Limitazioni del Modello Univariato**

1. **Perdita di Informazione**: Ignorando le correlazioni tra feature, il modello può perdere informazioni discriminanti importanti
2. **Assunzione di Normalità**: Se i dati non seguono effettivamente una distribuzione Gaussiana, le performance possono degradare
3. **Sensibilità agli Outlier**: La MLE è sensibile a valori estremi che possono distorcere le stime

**Estensioni Possibili**

1. **Modello Multivariato**: Considerare la matrice di covarianza completa
2. **Mixture Models**: Utilizzare miscele di Gaussiane per catturare distribuzioni multimodali
3. **Kernel Density Estimation**: Approccio non-parametrico per evitare assunzioni distributive

#### Conclusioni

L'analisi della stima di densità Gaussiana univariata ha rivelato diverse caratteristiche fondamentali del dataset:

1. **Feature Discriminanti**: Le feature 3 e 4 mostrano il pattern più promettente per la classificazione, con inversione delle medie tra le classi e varianze simili.

2. **Variabilità Intrinseca**: Le classi mostrano pattern diversi di variabilità, con la classe "Genuine" generalmente più dispersa nelle feature 1, 5, e 6.

3. **Qualità del Fit**: Il modello Gaussiano sembra appropriato per la maggior parte delle feature, come evidenziato dalle log-verosimiglianze e dall'ispezione visiva.

4. **Fondamenta per Classificatori Generativi**: I parametri stimati forniscono una base solida per lo sviluppo di classificatori probabilistici basati su assunzioni Gaussiane.

Questa analisi costituisce un passaggio fondamentale verso la comprensione della struttura probabilistica dei dati e prepara il terreno per lo sviluppo di modelli di classificazione più sofisticati, come i classificatori Gaussiani multivariati che saranno esplorati nei laboratori successivi.