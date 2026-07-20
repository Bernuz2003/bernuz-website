import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export default function TerminalDrawer() {
  const [isOpen, setIsOpen] = useState(false);
  const [history, setHistory] = useState<{ type: 'input' | 'output', text: string }[]>([
    { type: 'output', text: 'Benvenuto nel terminale nascosto. Digita "help" per i comandi.' }
  ]);
  const [input, setInput] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') return;
      if (e.key === '`' || e.key === '~' || e.key === '§') {
        e.preventDefault();
        setIsOpen(prev => !prev);
      }
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, []);

  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [isOpen]);

  const handleCommand = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      const cmd = input.trim();
      setInput('');
      
      const newHistory = [...history, { type: 'input' as const, text: cmd }];
      
      if (cmd === 'clear') {
        setHistory([]);
        return;
      }
      
      let output = '';
      switch(cmd.toLowerCase()) {
        case 'help': output = 'Comandi disponibili: whoami, ls, clear, matrix, exit'; break;
        case 'whoami': output = 'Emanuele Bernucci - Studente di Ingegneria Informatica e ML Enthusiast'; break;
        case 'ls': output = 'projects/  ml-labs/  resume.pdf  secret.txt'; break;
        case 'exit': setIsOpen(false); break;
        case 'matrix': 
          document.body.classList.toggle('matrix-rain-active');
          output = 'Wake up, Neo...';
          break;
        case '': break;
        default: output = `Comando non trovato: ${cmd}`; break;
      }

      if (output) newHistory.push({ type: 'output' as const, text: output });
      setHistory(newHistory);
    }
  };

  return (
    <>
      <AnimatePresence>
        {!isOpen && (
          <motion.div
            initial={{ y: 20 }}
            animate={{ y: 0 }}
            exit={{ y: 20 }}
            onClick={() => setIsOpen(true)}
            className="cursor-pointer font-mono text-secondary"
            style={{
              position: 'fixed',
              bottom: 0,
              left: '50%',
              transform: 'translateX(-50%)',
              background: 'rgba(5, 5, 5, 0.8)',
              padding: '0.2rem 1rem',
              borderTopLeftRadius: '8px',
              borderTopRightRadius: '8px',
              border: '1px solid rgba(13, 202, 240, 0.2)',
              borderBottom: 'none',
              zIndex: 9998,
              fontSize: '0.85rem'
            }}
          >
            &gt;_ terminal
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {isOpen && (
        <motion.div
          initial={{ y: '100%' }}
          animate={{ y: 0 }}
          exit={{ y: '100%' }}
          transition={{ type: 'spring', damping: 25, stiffness: 200 }}
          style={{
            position: 'fixed',
            bottom: 0,
            left: 0,
            right: 0,
            height: '30vh',
            minHeight: '200px',
            backgroundColor: 'rgba(5, 5, 5, 0.95)',
            backdropFilter: 'blur(10px)',
            borderTop: '1px solid rgba(13, 202, 240, 0.2)',
            zIndex: 9999,
            padding: '1rem',
            fontFamily: 'monospace',
            display: 'flex',
            flexDirection: 'column',
            boxShadow: '0 -10px 40px rgba(0,0,0,0.5)'
          }}
        >
          <div className="d-flex justify-content-between mb-2 pb-2 border-bottom border-secondary">
            <span className="text-secondary small">~/portfolio/hidden-shell</span>
            <span className="text-secondary cursor-pointer" onClick={() => setIsOpen(false)}>✕</span>
          </div>
          
          <div className="flex-grow-1 overflow-auto d-flex flex-column gap-1" style={{ fontSize: '0.9rem' }}>
            {history.map((h, i) => (
              <div key={i} className={h.type === 'input' ? 'text-light' : 'text-info'}>
                {h.type === 'input' ? <span className="text-secondary me-2">$</span> : null}
                {h.text}
              </div>
            ))}
          </div>
          
          <div className="d-flex mt-2">
            <span className="text-secondary me-2">$</span>
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleCommand}
              className="bg-transparent border-0 text-light flex-grow-1 outline-none"
              style={{ outline: 'none' }}
              spellCheck={false}
            />
          </div>
        </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
