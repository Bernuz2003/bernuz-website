# Pathfinding Visualizer

Questo è un progetto web interattivo creato per visualizzare il funzionamento di vari algoritmi di ricerca del percorso (pathfinding) su una griglia bidimensionale. L'applicazione permette agli utenti di costruire scenari complessi con muri e ostacoli, per poi osservare in tempo reale come gli algoritmi esplorano la griglia per trovare il cammino più breve tra un punto di partenza e uno di arrivo.

## 📜 Descrizione

Pathfinding Visualizer è uno strumento educativo e dimostrativo che rende accessibile la comprensione di algoritmi complessi. Gli utenti possono:
- Impostare un punto di **partenza** e uno di **arrivo**.
- Disegnare **muri** sulla griglia per creare ostacoli.
- Scegliere tra diversi algoritmi classici di pathfinding.
- Avviare la visualizzazione e osservare l'animazione che mostra le celle visitate dall'algoritmo e il percorso finale trovato.
- Analizzare le statistiche di performance di ogni algoritmo, come il numero di celle visitate, la lunghezza del percorso e il tempo di esecuzione.

L'interfaccia è stata progettata per essere intuitiva e reattiva, con una griglia che si adatta dinamicamente allo spazio disponibile.

## ✨ Caratteristiche

- **Griglia Interattiva**: Pieno controllo sulla configurazione della griglia tramite strumenti per posizionare/rimuovere start, end e muri.
- **Visualizzazione Animata**: Le animazioni mostrano passo dopo passo il processo di esplorazione di ogni algoritmo, aiutando a capirne la logica interna.
- **Algoritmi Multipli**: Supporto per i seguenti algoritmi:
  - **A* (A-Star)**: Un algoritmo informato che usa euristiche per trovare il percorso più breve in modo efficiente.
  - **Dijkstra**: Trova il percorso più breve tra i nodi in un grafo, ottimo per grafi con pesi diversi (in questo caso, i pesi sono uniformi).
  - **Breadth-First Search (BFS)**: Un algoritmo non informato che garantisce di trovare il percorso più breve in grafi non pesati.
  - **Depth-First Search (DFS)**: Un algoritmo non informato che esplora il più profondamente possibile lungo ogni ramo prima di tornare indietro (non garantisce il percorso più breve).
  - **Greedy Best-First Search**: Un algoritmo informato che si espande verso il nodo che sembra essere più vicino all'obiettivo, basandosi solo sull'euristica.
- **Controlli Completi**: Possibilità di resettare la griglia, cambiare la sua dimensione e selezionare gli strumenti con facilità.
- **Statistiche di Performance**: Dopo ogni esecuzione, vengono mostrate statistiche dettagliate per confrontare l'efficienza dei diversi algoritmi.
- **Design Moderno e Responsivo**: L'interfaccia utente è pulita e si adatta per offrire la migliore esperienza visiva, con le celle della griglia che si ridimensionano per riempire lo spazio.

## 🚀 Installazione ed Avvio Locale

Per eseguire il progetto in locale, segui questi passaggi:

1.  **Clona la repository**
    ```bash
    git clone https://github.com/tuo-username/Pathfinding-Visualizer.git
    ```

2.  **Entra nella cartella del progetto**
    ```bash
    cd Pathfinding-Visualizer
    ```

3.  **Installa le dipendenze**
    Il progetto utilizza `npm` per la gestione dei pacchetti.
    ```bash
    npm install
    ```

4.  **Avvia il server di sviluppo**
    Questo comando avvierà l'applicazione in modalità sviluppo con Vite.
    ```bash
    npm run dev
    ```
    Apri il browser e naviga all'indirizzo `http://localhost:5173` (o quello indicato nel terminale) per vedere l'applicazione in funzione.

## 🛠️ Tecnologie Utilizzate

-   **Linguaggio**: [TypeScript](https://www.typescriptlang.org/)
-   **Framework Frontend**: [React](https://reactjs.org/)
-   **Build Tool**: [Vite](https://vitejs.dev/)
-   **State Management**: React Hooks (`useReducer` e `useContext`) per una gestione dello stato centralizzata e prevedibile.
-   **Styling**: CSS puro con variabili CSS per una facile personalizzazione e manutenibilità. Nessun framework UI esterno.
