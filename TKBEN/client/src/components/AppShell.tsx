import { Outlet, useLocation } from 'react-router-dom';
import Sidebar from './Sidebar';

const AppShell = () => {
  const location = useLocation();

  return (
    <div className="app-shell">
      <Sidebar />
      <div className="app-main">
        <section className="app-content" key={location.pathname}>
          <Outlet />
        </section>
      </div>
    </div>
  );
};

export default AppShell;
