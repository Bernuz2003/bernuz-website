import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeHighlight from 'rehype-highlight';
import rehypeRaw from 'rehype-raw';
import 'highlight.js/styles/github-dark.css';
import '../styles/Markdown.css';
import { useState } from 'react';
import type { ComponentPropsWithoutRef, ReactNode } from 'react';

// Extract text from nested react elements
const extractText = (node: any): string => {
  if (typeof node === 'string') return node;
  if (Array.isArray(node)) return node.map(extractText).join('');
  if (node?.props?.children) return extractText(node.props.children);
  return '';
};

function PreBlock({ children, ...props }: ComponentPropsWithoutRef<'pre'> & { children?: ReactNode }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    const text = extractText(children);
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="position-relative code-block-wrapper group">
      <button 
        onClick={handleCopy}
        className={`btn btn-sm position-absolute top-0 end-0 m-2 font-mono ${copied ? 'btn-success' : 'btn-outline-secondary'}`}
        style={{ opacity: copied ? 1 : 0.4, transition: 'opacity 0.2s', zIndex: 10 }}
      >
        {copied ? 'Copiato!' : 'Copy'}
      </button>
      <pre {...props}>
        {children}
      </pre>
    </div>
  );
}

export default function Markdown({ source }: { source: string }) {
  return (
    <article className="markdown-body bg-dark border border-secondary rounded-2 p-4">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeRaw, rehypeKatex, rehypeHighlight]}
        components={{
          table: (props) => (
            <div className="md-table-wrap">
              <table className="md-table" {...props} />
            </div>
          ),
          pre: PreBlock
        }}
      >
        {source}
      </ReactMarkdown>
    </article>
  );
}
