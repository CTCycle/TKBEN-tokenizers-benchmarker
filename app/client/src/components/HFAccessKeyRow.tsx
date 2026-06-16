import type { HFAccessKeyListItem } from '../types/api';

type HFAccessKeyRowProps = {
  inProgress: boolean;
  isVisible: boolean;
  keyItem: HFAccessKeyListItem;
  preview: string;
  onDelete: (key: HFAccessKeyListItem) => void;
  onToggleActivation: (keyId: number, isActive: boolean) => void;
  onToggleReveal: (keyId: number) => void;
};

const HFAccessKeyRow = ({
  inProgress,
  isVisible,
  keyItem,
  preview,
  onDelete,
  onToggleActivation,
  onToggleReveal,
}: HFAccessKeyRowProps) => (
  <div className="key-manager-row">
    <div className="key-manager-preview">{preview}</div>
    <div className="key-manager-actions">
      <button
        type="button"
        className="icon-button subtle"
        aria-label={isVisible ? 'Hide key' : 'Reveal key'}
        onClick={() => onToggleReveal(keyItem.id)}
        disabled={inProgress}
      >
        <svg viewBox="0 0 24 24" aria-hidden="true">
          {isVisible ? (
            <>
              <path d="M3 3l18 18" strokeWidth="2" strokeLinecap="round" />
              <path
                d="M9.9 5.1A10.9 10.9 0 0 1 12 5c5 0 9 4 10 7-0.4 1.3-1.5 2.8-3.1 4.1M6.2 8.2C4.4 9.5 3.3 11 2.9 12 3.9 15 7 19 12 19c0.9 0 1.8-0.1 2.6-0.3"
                strokeWidth="2"
                strokeLinecap="round"
                fill="none"
              />
            </>
          ) : (
            <>
              <path
                d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7-10-7-10-7z"
                fill="none"
                strokeWidth="2"
              />
              <circle cx="12" cy="12" r="3" />
            </>
          )}
        </svg>
      </button>
      <button
        type="button"
        className={`icon-button subtle${keyItem.is_active ? ' accent' : ''}`}
        aria-label={keyItem.is_active ? 'Deactivate key (set locked)' : 'Activate key (locked)'}
        onClick={() => onToggleActivation(keyItem.id, keyItem.is_active)}
        disabled={inProgress}
      >
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <rect x="6" y="11" width="12" height="9" rx="2" fill="none" strokeWidth="2" />
          {keyItem.is_active ? (
            <path
              d="M9 11V8a3 3 0 0 1 5.5-1.7"
              fill="none"
              strokeWidth="2"
              strokeLinecap="round"
            />
          ) : (
            <path
              d="M8 11V8a4 4 0 0 1 8 0v3"
              fill="none"
              strokeWidth="2"
              strokeLinecap="round"
            />
          )}
        </svg>
      </button>
      <button
        type="button"
        className="icon-button danger"
        aria-label="Delete key"
        onClick={() => onDelete(keyItem)}
        disabled={inProgress || keyItem.is_active}
      >
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M6 6l12 12M18 6L6 18" strokeWidth="2" strokeLinecap="round" />
        </svg>
      </button>
    </div>
  </div>
);

export default HFAccessKeyRow;
