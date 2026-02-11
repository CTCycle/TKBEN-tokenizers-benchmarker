import { Navigate, Route, Routes } from 'react-router-dom';
import './App.css';
import AppShell from './components/AppShell';
import DatasetTokenizersPage from './pages/DatasetTokenizersPage';

const App = () => (
  <Routes>
    <Route element={<AppShell />}>
      <Route index element={<Navigate to="/dataset" replace />} />
      <Route path="/dataset" element={<DatasetTokenizersPage />} />
      <Route path="/tokenizers" element={<DatasetTokenizersPage />} />
      <Route path="*" element={<Navigate to="/dataset" replace />} />
    </Route>
  </Routes>
);

export default App;
