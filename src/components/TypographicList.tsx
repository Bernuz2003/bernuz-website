import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/TypographicList.css';
import ScrollReveal from './ScrollReveal';

export interface TypographicItem {
  title: string;
  slug: string;
  description?: string;
  techStack?: string[];
  link: string;
}

interface Props {
  items: TypographicItem[];
  limit?: number;
}

export default function TypographicList({ items, limit }: Props) {
  const navigate = useNavigate();
  const displayItems = limit ? items.slice(0, limit) : items;

  // Stato per l'immagine "fluttuante" al passaggio del mouse
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  return (
    <div className="typographic-list-container">
      {displayItems.map((item, idx) => (
        <ScrollReveal key={item.slug} delay={idx * 0.1}>
          <div 
            className="typographic-row d-flex flex-column flex-md-row align-items-md-center justify-content-between py-4 cursor-pointer"
            onClick={() => navigate(item.link)}
            onMouseEnter={() => setHoverIndex(idx)}
            onMouseLeave={() => setHoverIndex(null)}
          >
            <div className="d-flex flex-column">
              <h3 className="h4 mb-1 text-light fw-semibold title-transition">{item.title}</h3>
              {item.description && (
                <p className="text-secondary mb-0 small" style={{ maxWidth: '600px' }}>
                  {item.description.length > 100 ? item.description.substring(0, 100) + '...' : item.description}
                </p>
              )}
            </div>
            
            <div className="d-flex gap-2 mt-3 mt-md-0 ms-md-4 flex-wrap">
              {item.techStack?.slice(0, 3).map(tech => (
                <span key={tech} className="badge rounded-pill font-mono bg-dark text-secondary border border-secondary">
                  {tech}
                </span>
              ))}
            </div>
          </div>
        </ScrollReveal>
      ))}
    </div>
  );
}
