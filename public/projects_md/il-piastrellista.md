# Il Piastrellista - Puzzle Solver

![Sorting Visualizer](https://github.com/Bernuz2003/Il-Piastrellista/blob/master/media/il_piastrellista_img.png)

> Un'applicazione interattiva per visualizzare e comprendere il funzionamento degli algoritmi di ordinamento.

## **Descrizione**
Benvenuto in **Il Piastrellista**, un’applicazione React che simula un puzzle di “saltelli” su una griglia NxN. Lo scopo è riempire la griglia con numeri progressivi (da 1 fino a `N * N`) partendo da una cella iniziale, spostandosi secondo una serie di mosse predefinite (simili alle mosse di un cavallo negli scacchi ma con saltelli personalizzati).

## Funzionalità

- **Scelta dimensioni griglia**: puoi impostare il numero di righe (rows) e colonne (cols).
- **Posizione di partenza**: specifica la cella di partenza (startRow, startCol) in cui verrà posizionato il numero 1.
- **Algoritmo di ricerca**:
  - Usa backtracking per esplorare ricorsivamente tutte le possibili sequenze di mosse.
  - Integra un’euristica di riordino delle mosse (Warnsdorff-like) per ridurre i vicoli ciechi.
  - Applica un controllo preliminare di raggiungibilità globale (globalReachabilityCheck).
  - Esegue un pruning basato sulla connettività delle celle vuote per interrompere i rami non promettenti.
- **Visualizzazione soluzioni**: mostra in tempo reale lo stato della ricerca e permette di scorrere tra le varie soluzioni trovate.
- **Metriche di ricerca**: conta il numero di soluzioni, il numero di mosse effettuate e il tempo di ricerca.

## Requisiti

- **Node.js** (versione 14 o superiore, preferibilmente LTS).
- **NPM** o **Yarn** (per la gestione dei pacchetti).

## Installazione ed Esecuzione in Locale

### 1. Clona il repository

Apri il terminale (o prompt dei comandi) e digita:

```bash
git clone https://github.com/Bernuz2003/Il-Piastrellista.git
```

Entra nella cartella del progetto:

```bash
cd Il-Piastrellista
```

### 2. Installa le dipendenze

All’interno della cartella del progetto, installa le dipendenze con:

```bash
npm install
```

*(In alternativa, se preferisci Yarn, puoi usare `yarn install`.)*

### 3. Avvia il server di sviluppo

Per avviare l’app in modalità sviluppo, esegui:

```bash
npm run dev
```

Al termine della compilazione, dovresti vedere un messaggio nel terminale con l’indirizzo locale su cui l’applicazione è in ascolto, ad esempio:

```
  VITE v4.0.0  ready in 123 ms
  ➜  Local:   http://localhost:5173/
```

Apri [http://localhost:5173/](http://localhost:5173/) (o la porta indicata nel terminale) nel browser per usare l’applicazione.

### 4. Utilizzo dell’App

1. **Imposta dimensioni**: seleziona il numero di righe (`rows`) e di colonne (`cols`).
2. **Scegli la cella di partenza** (`startRow`, `startCol`).
3. **Avvia la ricerca** con il pulsante “Start” (o simile).  
4. **Osserva i risultati**:
   - Segui la ricerca in tempo reale sul “mini-board” di stato.
   - Monitora il numero di soluzioni, di mosse effettuate e il tempo trascorso.
   - Se vengono trovate più soluzioni, puoi navigare tra di esse con i pulsanti Avanti/Indietro.
5. **Ferma la ricerca** con il pulsante “Stop” se desideri interromperla in anticipo.

### 5. Build per la produzione (opzionale)

Se vuoi generare una build ottimizzata per la produzione, puoi eseguire:

```bash
npm run build
```

Questo comando crea una cartella `dist` con i file pronti per essere distribuiti su un server.

## Struttura Principale del Codice

- **`src/utils/solver.ts`**: Contiene la logica di backtracking, incluse l’euristica di riordino, i controlli preliminari e il pruning.
- **`src/components`**: Contiene i vari componenti React (Matrix, Controls, SearchStatus, ecc.) che gestiscono l’interfaccia grafica.
- **`src/App.tsx`**: Entry point dell’app React. Gestisce lo stato globale della ricerca, il timer, e le interazioni con i componenti.

---


## Contribuire

Se vuoi contribuire:
1. Fai un fork del repository.
2. Crea un branch con il tuo contributo: `git checkout -b feature/tua-feature`.
3. Manda una Pull Request.  

---

**Buon divertimento con Il Piastrellista!** Se hai domande o riscontri problemi, non esitare ad aprire una issue o inviare una PR. Buone partite!
