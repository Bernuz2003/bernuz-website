import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { workItems } from '../data/workItems';

export default function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const navigate = useNavigate();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setOpen((o) => !o);
      }
      if (e.key === 'Escape') {
        setOpen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  useEffect(() => {
    setSelectedIndex(0);
  }, [query]);

  if (!open) return null;

  const isMatrix = document.body.classList.contains('matrix-rain-active');

  const staticCommands = [
    { name: 'Home', action: () => navigate('/') },
    { name: 'Portfolio (Work)', action: () => navigate('/work') },
    { name: 'Contatti', action: () => { navigate('/'); setTimeout(() => document.getElementById('contact')?.scrollIntoView({ behavior: 'smooth'}), 100) } },
    { name: 'Toggle CRT Mode', action: () => { 
      document.body.classList.toggle('crt-mode'); 
      window.dispatchEvent(new Event('crtToggled'));
      setOpen(false); 
    } },
    { name: '> help', action: () => {
      setQuery('> theme');
    } },
    { name: `> theme: matrix ${isMatrix ? '[ATTIVO]' : ''}`, action: () => {
      document.body.classList.toggle('matrix-rain-active');
      document.body.classList.toggle('matrix-theme');
      setOpen(false);
    } }
  ];

  const projectCommands = workItems.map(w => ({
    name: `Work: ${w.title} (${w.domain})`,
    action: () => navigate(`/work/${w.slug}`)
  }));

  const allCommands = [...staticCommands, ...projectCommands];

  // Mostra solo i comandi principali all'avvio (query vuota)
  const commandsToShow = query.trim() === '' 
    ? staticCommands 
    : allCommands;

  const filtered = commandsToShow.filter(c => c.name.toLowerCase().includes(query.toLowerCase()));

  const handleInputKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex(prev => Math.min(prev + 1, filtered.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex(prev => Math.max(prev - 1, 0));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (filtered[selectedIndex]) {
        filtered[selectedIndex].action();
        setOpen(false);
      }
    }
  };

  return (
    <div className="fixed-top w-100 h-100 d-flex align-items-center justify-content-center" style={{ zIndex: 1050, background: 'rgba(0,0,0,0.5)', backdropFilter: 'blur(4px)' }} onClick={() => setOpen(false)}>
      <div className="glass-panel p-3 w-100" style={{ maxWidth: '500px' }} onClick={e => e.stopPropagation()}>
        <input 
          type="text" 
          autoFocus
          className="form-control bg-transparent text-white border-secondary font-mono mb-3" 
          placeholder="Type a command or search..."
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={handleInputKeyDown}
        />
        <div className="d-flex flex-column gap-2">
          {filtered.map((cmd, i) => (
            <button 
              key={i}
              className={`btn text-start font-mono border-0 text-white hover-white px-3 py-2 ${i === selectedIndex ? 'bg-secondary bg-opacity-25' : 'btn-outline-secondary'}`}
              onClick={() => { cmd.action(); setOpen(false); }}
              onMouseEnter={() => setSelectedIndex(i)}
            >
              {cmd.name}
            </button>
          ))}
          {filtered.length === 0 && <span className="text-secondary p-2">Nessun comando trovato.</span>}
        </div>
      </div>
    </div>
  );
}
