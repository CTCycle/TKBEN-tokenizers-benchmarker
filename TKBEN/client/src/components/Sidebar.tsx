import { useLocation, useNavigate } from 'react-router-dom';
import { useState } from 'react';
import HFAccessKeyManager from './HFAccessKeyManager';

const navItems = [
  {
    to: '/dataset',
    label: 'Datasets',
    icon: (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <rect x="3" y="4" width="18" height="4" rx="1.5" />
        <rect x="3" y="10" width="18" height="4" rx="1.5" />
        <rect x="3" y="16" width="18" height="4" rx="1.5" />
      </svg>
    ),
  },
  {
    to: '/tokenizers',
    label: 'Tokenizers',
    icon: (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <circle cx="8" cy="8" r="3" />
        <circle cx="16" cy="8" r="3" />
        <rect x="6" y="13" width="12" height="7" rx="2" />
      </svg>
    ),
  },
  {
    to: '/cross-benchmark',
    label: 'Cross Benchmark',
    icon: (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <rect x="4" y="4" width="7" height="7" rx="1.5" />
        <rect x="13" y="4" width="7" height="7" rx="1.5" />
        <rect x="4" y="13" width="7" height="7" rx="1.5" />
        <rect x="13" y="13" width="7" height="7" rx="1.5" />
      </svg>
    ),
  },
];

const Sidebar = () => {
  const [isKeyManagerOpen, setIsKeyManagerOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();

  const isNavItemActive = (targetPath: string): boolean =>
    location.pathname === targetPath || location.pathname.startsWith(`${targetPath}/`);

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">TK</div>
      <nav className="sidebar-nav">
        {navItems.map((item) => (
          <button
            key={item.to}
            type="button"
            className={`sidebar-link${isNavItemActive(item.to) ? ' sidebar-link--active' : ''}`}
            aria-label={item.label}
            aria-current={isNavItemActive(item.to) ? 'page' : undefined}
            onClick={() => {
              if (!isNavItemActive(item.to)) {
                navigate(item.to);
              }
            }}
          >
            {item.icon}
          </button>
        ))}
        <button
          type="button"
          className={`sidebar-link sidebar-link--utility${isKeyManagerOpen ? ' sidebar-link--active' : ''}`}
          aria-label="Manage Hugging Face keys"
          onClick={() => setIsKeyManagerOpen((value) => !value)}
        >
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <circle cx="8" cy="12" r="3" fill="none" strokeWidth="2" />
            <path
              d="M11 12h10M18 12v3M21 12v2"
              strokeWidth="2"
              strokeLinecap="round"
              fill="none"
            />
          </svg>
        </button>
      </nav>
      <HFAccessKeyManager
        isOpen={isKeyManagerOpen}
        onClose={() => setIsKeyManagerOpen(false)}
      />
    </aside>
  );
};

export default Sidebar;
