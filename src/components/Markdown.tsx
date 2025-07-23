import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeHighlight from 'rehype-highlight';
import rehypeRaw from 'rehype-raw';
import 'highlight.js/styles/github-dark.css';
import '../styles/Markdown.css';

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
        }}
      >
        {source}
      </ReactMarkdown>
    </article>
  );
}
