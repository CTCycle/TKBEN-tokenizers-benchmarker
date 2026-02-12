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
