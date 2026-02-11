import DatasetPage from './DatasetPage';
import TokenizersPage from './TokenizersPage';

const DatasetTokenizersPage = () => (
  <div className="page-scroll">
    <div className="merged-page-layout">
      <div className="merged-page-row">
        <DatasetPage showDashboard={false} embedded />
      </div>
      <div className="merged-page-row">
        <TokenizersPage showDashboard={false} embedded />
      </div>
    </div>
  </div>
);

export default DatasetTokenizersPage;
