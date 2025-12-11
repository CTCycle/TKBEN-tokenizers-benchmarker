import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import './index.css';
import App from './App.tsx';
import { DatasetProvider } from './contexts/DatasetContext';
import { TokenizersProvider } from './contexts/TokenizersContext';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <DatasetProvider>
        <TokenizersProvider>
          <App />
        </TokenizersProvider>
      </DatasetProvider>
    </BrowserRouter>
  </StrictMode>,
);
