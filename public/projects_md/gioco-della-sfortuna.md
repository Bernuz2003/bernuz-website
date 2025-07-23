## React Client Application Routes

- Route `/`: Homepage principale con regolamento completo del gioco che spiega obiettivi, meccaniche di gioco e sistema di punteggio. Presenta una sezione regolamento dettagliata in formato card Bootstrap e include un pulsante dinamico CTA che avvia partita completa per utenti autenticati o demo per visitatori anonimi, utilizzando API.startGame() per l'inizializzazione.

- Route `/play`: Pagina principale di gioco dove si svolge l'intera partita interattiva in tempo reale. Layout a due colonne: sinistra mostra griglia responsive delle carte possedute ordinate per indice sfortuna, destra presenta la carta misteriosa da indovinare con timer integrato di 30 secondi, progress bar animata e bottoni dinamici per posizionamento. Gestisce completamente la logica di round, contatore errori per utenti loggati, stati loading/error e transizioni tra round con modal di conferma risultato.

- Route `/result/:gameId`: Pagina risultato immediato mostrata subito dopo il completamento di qualsiasi partita (demo o normale). Visualizza esito finale con badge colorato (verde vittoria, rosso sconfitta), griglia delle carte raccolte con componente OwnedCards riutilizzabile, e fornisce azioni contestuali differenziate: nuova partita + cronologia per utenti loggati, invito al login per demo.

- Route `/history`: Lista cronologia completa delle partite dell'utente autenticato in formato tabella Bootstrap responsive. Richiede autenticazione obbligatoria con redirect automatico a /login se non loggato. Include colonne ordinate: numero progressivo, data formattata DD/MM/YYYY HH:mm, badge colorato per l'esito e link "View" per accedere ai dettagli. Gestisce stato vuoto con messaggio di invito al gioco.

- Route `/history/:gameId`: Pagina dettagli approfonditi di una partita specifica dalla cronologia utente. Layout strutturato in sezioni card Bootstrap: riepilogo generale (data, esito, carte totali), sezione carte iniziali ordinate per indice sfortuna, cronologia completa round con risultati vinta/persa. Include navigazione breadcrumb per tornare al profilo e bottone per nuove partite.

- Route `/login`: Form di autenticazione semplificato con validazione client-side, campi username e password required, gestione errori con Alert Bootstrap dismissible. Dopo login di successo reindirizza automaticamente alla homepage con stato aggiornato, mentre gli utenti già autenticati vengono automaticamente reindirizzati alla home tramite Navigate component.

## API Server

- POST `/api/sessions`
  - Endpoint di autenticazione che riceve credenziali utente, verifica con database SQLite e crea sessione Passport.js persistente
  - Request body: 
  ```json
  {
    "username": "admin",
    "password": "password123"
  }
  ```
  - Response body (success):
  ```json
  {
    "id": 1,
    "username": "admin"
  }
  ```
  - Response body (error): `401 Unauthorized` con messaggio errore


- GET `/api/sessions/current`
  - Verifica stato autenticazione corrente utilizzando sessione Passport e ritorna dati utente se loggato
  - Request: Nessun body, utilizza cookie di sessione
  - Response body (authenticated):
  ```json
  {
    "id": 1,
    "username": "admin"
  }
  ```
  - Response body (not authenticated): `401 Unauthorized` con `{ "error": "Not authenticated" }`


- DELETE `/api/sessions/current`
  - Termina sessione utente corrente, pulisce cookie di autenticazione e invalida sessione Passport
  - Request: Nessun body
  - Response: Status 200 con body vuoto

- POST `/api/games`
  - Crea nuova partita con 3 carte iniziali casuali ordinate per indice sfortuna. Determina automaticamente tipo: demo per utenti anonimi (userId=null), partite complete per utenti loggati (userId valorizzato)
  - Request body: Vuoto (il server determina automaticamente il tipo basandosi sull'autenticazione)
  - Response body:
  ```json
  {
    "gameId": 123,
    "initialCards": [
      {
        "id": 1,
        "name": "Volo cancellato",
        "imagePath": "/images/volo.jpg",
        "unluckIdx": 15
      },
      {
        "id": 5,
        "name": "Bagaglio perso",
        "imagePath": "/images/bagaglio.jpg", 
        "unluckIdx": 42
      },
      {
        "id": 8,
        "name": "Hotel overbooking",
        "imagePath": "/images/hotel.jpg",
        "unluckIdx": 78
      }
    ]
  }
  ```

- GET `/api/games/:id`
  - Ottiene riepilogo completo partita per pagine risultato e dettagli cronologia. Include dati partita, tutte le carte giocate con round e risultati. Verifica autorizzazioni per partite normali (solo proprietario), mentre demo (userId=null) sono accessibili pubblicamente
  - Request: gameId come parametro URL
  - Response body:
  ```json
  {
    "game": {
      "id": 123,
      "userId": 1,
      "startTS": "2025-01-15T10:30:00.000Z",
      "outcome": "win"
    },
    "gameCards": [
      {
        "gameId": 123,
        "cardId": 1,
        "round": 0,
        "gained": true,
        "card": {
          "id": 1,
          "name": "Volo cancellato",
          "imagePath": "/images/volo.jpg",
          "unluckIdx": 15
        }
      }
    ]
  }
  ```

- GET `/api/games`
  - Richiede autenticazione obbligatoria. Lista cronologia partite utente ordinata per data decrescente per pagina profilo
  - Request: Nessun body, richiede cookie autenticazione
  - Response body:
  ```json
  [
    {
      "id": 123,
      "userId": 1,
      "startTS": "2025-01-15T10:30:00.000Z",
      "outcome": "win"
    },
    {
      "id": 122,
      "userId": 1,
      "startTS": "2025-01-14T15:20:00.000Z",
      "outcome": "lose"
    }
  ]
  ```

- GET `/api/games/:id/next-card`
  - Ottiene prossima carta casuale da indovinare per round corrente, escludendo carte già giocate. Include token sicurezza temporizzato e carte possedute aggiornate
  - Response body:
  ```json
  {
    "cardId": 12,
    "name": "Treno in ritardo",
    "imagePath": "/images/img1.jpg",
    "roundToken": "eyJnYW1lSWQiOjEyMy...",
    "ownedCards": [
      {
        "id": 1,
        "name": "Volo cancellato",
        "imagePath": "/images/img2.jpg",
        "unluckIdx": 15
      }
    ]
  }
  ```

- POST `/api/games/:id/guess`
  - Riceve tentativo posizionamento carta con verifica token sicurezza temporizzato (30 secondi timeout). Calcola correttezza confrontando posizione scelta con ordinamento teorico per indice sfortuna. Registra risultato nel database ma non gestisce logica fine partita (delegata al client)
  - Request body: 
  ```json
  {
    "position": 2,
    "roundToken": "eyJnYW1lSWQiOjEyMywiY2FyZElkIjoxMiwidGltZXN0YW1wIjoxNjczNzgwNDAwMDAwfQ==",
    "cardId": 12
  }
  ```
  - Response body:
  ```json
  {
    "correct": true,
    "card": {
      "id": 12,
      "name": "Treno in ritardo",
      "imagePath": "/images/treno.jpg",
      "unluckIdx": 45
    },
    "timeExpired": false
  }
  ```

- POST `/api/games/:id/complete`
  - Finalizza partita aggiornando database con esito finale e timestamp conclusione. Chiamato dal client dopo logica vittoria/sconfitta locale per persistenza dati
  - Request body: 
  ```json
  {
    "outcome": "win",
    "errors": 2
  }
  ```
  - Response body:
  ```json
  {
    "success": true
  }
  ```

## Database Tables

- Table `users` - Gestisce utenti pre-inseriti staticamente nel database con credenziali hardcoded. Include id, username univoco per login e password in plaintext. Non supporta registrazione dinamica - gli utenti vengono inseriti manualmente nel database SQLite per testing.

- Table `cards` - Contiene collezione completa di 20+ carte tema viaggi/turismo con id, name descrittivo della disgrazia, imagePath per risorse statiche servite da Express e unluckIdx (1-100) che determina ordinamento sfortuna e difficoltà posizionamento nel gameplay.

- Table `games` - Traccia partite individuali con id auto-incrementale, userId (null per demo anonime), startTS timestamp inizio partita e outcome finale ('win'/'lose'/'demo-win'/'demo-lose'). Supporta sia partite complete (max 6 carte, 3 errori) che demo (1 carta, 1 tentativo).

- Table `game_cards` - Tabella relazionale che collega ogni partita alle carte giocate/possedute. Include gameId, cardId, round (0 per carte iniziali, 1+ per carte indovinate), gained boolean per tracciare successo/fallimento posizionamento nel round specifico.

## Main React Components

- `Home` (in `Home.jsx`): Homepage principale con regolamento dettagliato del gioco, spiegazione obiettivi e meccaniche. Include CTA dinamico che adatta il messaggio e comportamento in base allo stato autenticazione utente.

- `Play` (in `Play.jsx`): Cuore dell'applicazione che gestisce tutta la logica di gioco. Coordina timer 30 secondi, contatore errori per utenti loggati, comunicazione API per carte e round, gestione stati loading/error e transizioni tra round con modal conferma.

- `GuessForm` (in `GuessForm.jsx`): Componente carta da indovinare con timer integrato visuale (progress bar). Genera dinamicamente bottoni posizione basati su carte possedute, gestisce selezione utente e disabilita interfaccia durante invio tentativo.

- `OwnedCards` (in `OwnedCards.jsx`): Griglia responsive per visualizzare carte possedute con layout adattivo. Automaticamente organizza 1-3 carte in riga singola, 4-6 carte in griglia 2x3, mostrando immagine, nome, indice sfortuna e opzionalmente round acquisizione.

- `Result` (in `Result.jsx`): Pagina risultato immediato post-partita con badge esito colorato. Distingue tra demo (messaggio invito login) e partite complete, mostra carte vinte con componente OwnedCards e fornisce azioni contestuali per proseguire esperienza.

- `GameHistory` (in `GameHistory.jsx`): Pagina dettagli cronologia con layout strutturato in card Bootstrap. Separa visivamente carte iniziali da carte round, mostra cronologia completa tentativi con badge successo/fallimento e include navigazione verso nuove partite.

- `Profile` (in `Profile.jsx`): Lista cronologia partite in formato tabella responsive con colonne data, risultato badge colorato e link dettagli. Gestisce stati loading/error e mostra messaggio vuoto per utenti senza partite precedenti.

- `RoundConfirm` (in `RoundConfirm.jsx`): Modal Bootstrap per conferma risultato round con animazioni e feedback visuale. Mostra carta vinta in caso successo con dimensioni ottimizzate per popup, messaggi errore per fallimenti e bottone continuazione verso round successivo.

- `NavHeader` (in `NavHeader.jsx`): Barra navigazione responsive con menu contestuale per stato autenticazione. Mostra link differenziati per utenti loggati (Gioca, Cronologia, Logout) vs visitatori (solo Login), include saluto personalizzato con username.

## Screenshot

![Screenshot](./client/public/game_running.png)
![Screenshot](./client/public/history.png)
![Screenshot](./client/public/game_details.png)


## Users Credentials
(username, password)
- user1, webapp
- user2, webapp 
