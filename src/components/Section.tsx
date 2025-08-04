import type { ReactNode } from 'react';
import { Container } from 'react-bootstrap';

interface Props { 
  id?: string; 
  title?: string; 
  lead?: string; 
  children: ReactNode; 
  className?: string;
  titleAction?: ReactNode;
}

export default function Section({id,title,lead,children,className,titleAction}:Props){
  return (
    <section id={id} className={`section-padding ${className||''}`}> 
      <Container>
        {title && (
          <div className="d-flex justify-content-between align-items-center mb-3">
            <h2 className="h2 fw-bold mb-0">{title}</h2>
            {titleAction}
          </div>
        )}
        {lead && <p className="text-secondary mb-4" style={{maxWidth:'52ch'}}>{lead}</p>}
        {children}
      </Container>
    </section>
  );
}