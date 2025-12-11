import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';

const AppShell = () => (
  <div className="app-shell">
    <Sidebar />
    <div className="app-main">
      <header className="app-header">
        <div>
          <h1>TKBEN Dashboard</h1>
          <p>Benchmark datasets and tokenizers directly from the browser.</p>
        </div>
      </header>
      <section className="app-content">
        <Outlet />
      </section>
    </div>
  </div>
);

export default AppShell;
