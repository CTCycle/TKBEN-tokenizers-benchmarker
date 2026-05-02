import { useState } from 'react';
import { Outlet, useLocation, useNavigate } from 'react-router-dom';
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

const AppShell = () => {
  const [isKeyManagerOpen, setIsKeyManagerOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();

  const isNavItemActive = (targetPath: string): boolean =>
    location.pathname === targetPath || location.pathname.startsWith(`${targetPath}/`);

  return (
    <div className="app-shell">
      <header className="app-header-bar">
        <h1 className="app-header-title">TKBEN Benchmarker</h1>
        <button
          type="button"
          className={`icon-button subtle app-header-key-button${isKeyManagerOpen ? ' accent' : ''}`}
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
      </header>
      <nav className="app-tab-nav" aria-label="Primary">
        {navItems.map((item) => (
          <button
            key={item.to}
            type="button"
            className={`app-tab${isNavItemActive(item.to) ? ' app-tab--active' : ''}`}
            aria-current={isNavItemActive(item.to) ? 'page' : undefined}
            onClick={() => {
              if (!isNavItemActive(item.to)) {
                navigate(item.to);
              }
            }}
          >
            <span className="app-tab-icon">{item.icon}</span>
            <span>{item.label}</span>
          </button>
        ))}
      </nav>
      <div className="app-main">
        <section className="app-content" key={location.pathname}>
          <Outlet />
        </section>
      </div>
      <HFAccessKeyManager
        isOpen={isKeyManagerOpen}
        onClose={() => setIsKeyManagerOpen(false)}
      />
    </div>
  );
};

export default AppShell;
