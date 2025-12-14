import { Navigate, Route, Routes } from 'react-router-dom';
import './App.css';
import AppShell from './components/AppShell';
import DatasetPage from './pages/DatasetPage';
import TokenizersPage from './pages/TokenizersPage';
import DatabaseBrowserPage from './pages/DatabaseBrowserPage';

const App = () => (
  <Routes>
    <Route element={<AppShell />}>
      <Route index element={<Navigate to="/dataset" replace />} />
      <Route path="/dataset" element={<DatasetPage />} />
      <Route path="/tokenizers" element={<TokenizersPage />} />
      <Route path="/database" element={<DatabaseBrowserPage />} />
      <Route path="*" element={<Navigate to="/dataset" replace />} />
    </Route>
  </Routes>
);

export default App;
