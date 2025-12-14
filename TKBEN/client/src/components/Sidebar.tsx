import { NavLink } from 'react-router-dom';

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
        <circle cx="12" cy="12" r="9" />
        <path d="M12 5v14M5 12h14" strokeWidth="2" strokeLinecap="round" />
      </svg>
    ),
  },
  {
    to: '/database',
    label: 'Database Browser',
    icon: (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <ellipse cx="12" cy="6" rx="8" ry="3" />
        <path d="M4 6v6c0 1.66 3.58 3 8 3s8-1.34 8-3V6" strokeWidth="1.5" fill="none" stroke="currentColor" />
        <path d="M4 12v6c0 1.66 3.58 3 8 3s8-1.34 8-3v-6" strokeWidth="1.5" fill="none" stroke="currentColor" />
      </svg>
    ),
  },
];

const Sidebar = () => (
  <aside className="sidebar">
    <div className="sidebar-logo">TK</div>
    <nav className="sidebar-nav">
      {navItems.map((item) => (
        <NavLink
          key={item.to}
          to={item.to}
          className={({ isActive }) =>
            `sidebar-link${isActive ? ' sidebar-link--active' : ''}`
          }
          aria-label={item.label}
        >
          {item.icon}
        </NavLink>
      ))}
    </nav>
  </aside>
);

export default Sidebar;
