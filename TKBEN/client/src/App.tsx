import { Navigate, Route, Routes } from 'react-router-dom';
import './App.css';
import AppShell from './components/AppShell';
import CrossBenchmarkPage from './pages/CrossBenchmarkPage';
import DatasetPage from './pages/DatasetPage';
import TokenizerExaminationPage from './pages/TokenizerExaminationPage';

const App = () => (
  <Routes>
    <Route element={<AppShell />}>
      <Route index element={<Navigate to="/dataset" replace />} />
      <Route path="/dataset" element={<DatasetPage />} />
      <Route path="/tokenizers" element={<TokenizerExaminationPage />} />
      <Route path="/cross-benchmark" element={<CrossBenchmarkPage />} />
      <Route path="*" element={<Navigate to="/dataset" replace />} />
    </Route>
  </Routes>
);

export default App;
