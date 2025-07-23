export default function SkillBadge({ label }: { label: string }) {
  return <span className="badge badge-tech rounded-pill px-3 py-2 me-2 mb-2">{label}</span>;
}