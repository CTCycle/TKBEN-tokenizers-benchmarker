import { useCallback, useEffect, useMemo, useState } from 'react';
import type { HFAccessKeyListItem } from '../types/api';
import {
  activateHFAccessKey,
  addHFAccessKey,
  deactivateHFAccessKey,
  deleteHFAccessKey,
  fetchHFAccessKeys,
  revealHFAccessKey,
} from '../services/keysApi';

type HFAccessKeyManagerProps = {
  isOpen: boolean;
  onClose: () => void;
};

const HFAccessKeyManager = ({ isOpen, onClose }: HFAccessKeyManagerProps) => {
  const [newKeyValue, setNewKeyValue] = useState('');
  const [keys, setKeys] = useState<HFAccessKeyListItem[]>([]);
  const [revealedValues, setRevealedValues] = useState<Record<number, string>>({});
  const [visibleRows, setVisibleRows] = useState<Record<number, boolean>>({});
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [actionKeyId, setActionKeyId] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadKeys = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetchHFAccessKeys();
      setKeys(response.keys);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load keys.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    void loadKeys();
  }, [isOpen, loadKeys]);

  const hasKeys = useMemo(() => keys.length > 0, [keys.length]);

  const handleAddKey = async () => {
    const normalizedValue = newKeyValue.trim();
    if (!normalizedValue) {
      setError('Key cannot be empty.');
      return;
    }

    setSubmitting(true);
    setError(null);
    try {
      await addHFAccessKey(normalizedValue);
      setNewKeyValue('');
      setVisibleRows({});
      setRevealedValues({});
      await loadKeys();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to add key.');
    } finally {
      setSubmitting(false);
    }
  };

  const handleToggleReveal = async (keyId: number) => {
    if (visibleRows[keyId]) {
      setVisibleRows((current) => ({ ...current, [keyId]: false }));
      return;
    }

    setError(null);
    if (revealedValues[keyId]) {
      setVisibleRows((current) => ({ ...current, [keyId]: true }));
      return;
    }

    setActionKeyId(keyId);
    try {
      const response = await revealHFAccessKey(keyId);
      setRevealedValues((current) => ({ ...current, [keyId]: response.key_value }));
      setVisibleRows((current) => ({ ...current, [keyId]: true }));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reveal key.');
    } finally {
      setActionKeyId(null);
    }
  };

  const handleActivate = async (keyId: number, isActive: boolean) => {
    setActionKeyId(keyId);
    setError(null);
    try {
      if (isActive) {
        await deactivateHFAccessKey(keyId);
      } else {
        await activateHFAccessKey(keyId);
      }
      await loadKeys();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update key state.');
    } finally {
      setActionKeyId(null);
    }
  };

  const handleDelete = async (key: HFAccessKeyListItem) => {
    if (key.is_active) {
      setError('The active key cannot be deleted.');
      return;
    }
    const confirmed = window.confirm('Delete this key?');
    if (!confirmed) {
      return;
    }

    setActionKeyId(key.id);
    setError(null);
    try {
      await deleteHFAccessKey(key.id);
      setVisibleRows((current) => ({ ...current, [key.id]: false }));
      setRevealedValues((current) => {
        const next = { ...current };
        delete next[key.id];
        return next;
      });
      await loadKeys();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete key.');
    } finally {
      setActionKeyId(null);
    }
  };

  if (!isOpen) {
    return null;
  }

  return (
    <div className="key-manager-popover" role="dialog" aria-label="Hugging Face keys">
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
        <div className="error-banner" role="alert">
          <span>{error}</span>
          <button type="button" aria-label="Dismiss" onClick={() => setError(null)}>
            ×
          </button>
        </div>
      )}

      <div className="key-manager-list">
        {loading ? (
          <div className="key-manager-empty">Loading keys...</div>
        ) : !hasKeys ? (
          <div className="key-manager-empty">No keys stored.</div>
        ) : (
          keys.map((key) => {
            const isVisible = Boolean(visibleRows[key.id]);
            const preview = isVisible
              ? (revealedValues[key.id] || key.masked_preview)
              : key.masked_preview;
            const inProgress = actionKeyId === key.id;

            return (
              <div className="key-manager-row" key={key.id}>
                <div className="key-manager-preview">{preview}</div>
                <div className="key-manager-actions">
                  <button
                    type="button"
                    className="icon-button subtle"
                    aria-label={isVisible ? 'Hide key' : 'Reveal key'}
                    onClick={() => void handleToggleReveal(key.id)}
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
                    className={`icon-button subtle${key.is_active ? ' accent' : ''}`}
                    aria-label={key.is_active ? 'Deactivate key (set locked)' : 'Activate key (locked)'}
                    onClick={() => void handleActivate(key.id, key.is_active)}
                    disabled={inProgress}
                  >
                    <svg viewBox="0 0 24 24" aria-hidden="true">
                      <rect x="6" y="11" width="12" height="9" rx="2" fill="none" strokeWidth="2" />
                      {key.is_active ? (
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
                    onClick={() => void handleDelete(key)}
                    disabled={inProgress || key.is_active}
                  >
                    <svg viewBox="0 0 24 24" aria-hidden="true">
                      <path d="M6 6l12 12M18 6L6 18" strokeWidth="2" strokeLinecap="round" />
                    </svg>
                  </button>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};

export default HFAccessKeyManager;
