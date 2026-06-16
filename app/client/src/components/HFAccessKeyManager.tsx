import { useState } from 'react';
import DismissibleBanner from './DismissibleBanner';
import HFAccessKeyRow from './HFAccessKeyRow';
import { useHFAccessKeys } from '../hooks/useHFAccessKeys';

type HFAccessKeyManagerProps = {
  isOpen: boolean;
  onClose: () => void;
};

const HFAccessKeyManager = ({ isOpen, onClose }: HFAccessKeyManagerProps) => {
  const [newKeyValue, setNewKeyValue] = useState('');
  const {
    actionKeyId,
    addKey,
    clearError,
    deleteKey,
    error,
    keys,
    loading,
    revealedValues,
    submitting,
    toggleActivation,
    toggleReveal,
    visibleRows,
  } = useHFAccessKeys(isOpen);

  const handleAddKey = async (): Promise<void> => {
    const added = await addKey(newKeyValue);
    if (added) {
      setNewKeyValue('');
    }
  };

  if (!isOpen) {
    return null;
  }

  return (
    <>
      <button
        type="button"
        className="key-manager-backdrop"
        aria-label="Close key manager"
        onClick={onClose}
      />
      <div className="key-manager-popover" role="dialog" aria-modal="true" aria-label="Hugging Face keys">
        <header className="key-manager-header">
          <p className="panel-label">Hugging Face Keys</p>
          <button
            type="button"
            className="icon-button subtle"
            onClick={onClose}
            aria-label="Close key manager"
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M6 6l12 12M18 6L6 18" strokeWidth="2" strokeLinecap="round" />
            </svg>
          </button>
        </header>

        <div className="key-manager-add-row">
          <input
            type="text"
            className="text-input"
            placeholder="Enter Hugging Face key"
            value={newKeyValue}
            onChange={(event) => setNewKeyValue(event.target.value)}
          />
          <button
            type="button"
            className="primary-button"
            onClick={() => void handleAddKey()}
            disabled={submitting}
          >
            Add
          </button>
        </div>

        {error && (
          <DismissibleBanner message={error} onDismiss={clearError} />
        )}

        <div className="key-manager-list">
          {loading ? (
            <div className="key-manager-empty">Loading keys...</div>
          ) : keys.length === 0 ? (
            <div className="key-manager-empty">No keys stored.</div>
          ) : (
            keys.map((key) => {
              const isVisible = Boolean(visibleRows[key.id]);
              const preview = isVisible
                ? (revealedValues[key.id] || key.masked_preview)
                : key.masked_preview;
              const inProgress = actionKeyId === key.id;

              return (
                <HFAccessKeyRow
                  key={key.id}
                  inProgress={inProgress}
                  isVisible={isVisible}
                  keyItem={key}
                  preview={preview}
                  onDelete={(keyItem) => void deleteKey(keyItem)}
                  onToggleActivation={(keyId, isActive) => void toggleActivation(keyId, isActive)}
                  onToggleReveal={(keyId) => void toggleReveal(keyId)}
                />
              );
            })
          )}
        </div>
      </div>
    </>
  );
};

export default HFAccessKeyManager;
