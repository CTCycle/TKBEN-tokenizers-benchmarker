import { useCallback, useEffect, useState } from 'react';
import {
  activateHFAccessKey,
  addHFAccessKey,
  deactivateHFAccessKey,
  deleteHFAccessKey,
  fetchHFAccessKeys,
  revealHFAccessKey,
} from '../services/keysApi';
import type { HFAccessKeyListItem } from '../types/api';

type UseHFAccessKeysResult = {
  actionKeyId: number | null;
  error: string | null;
  keys: HFAccessKeyListItem[];
  loading: boolean;
  revealedValues: Record<number, string>;
  submitting: boolean;
  visibleRows: Record<number, boolean>;
  addKey: (keyValue: string) => Promise<boolean>;
  clearError: () => void;
  deleteKey: (key: HFAccessKeyListItem) => Promise<void>;
  toggleActivation: (keyId: number, isActive: boolean) => Promise<void>;
  toggleReveal: (keyId: number) => Promise<void>;
  clearRevealedValues: () => void;
};

export const useHFAccessKeys = (isOpen: boolean): UseHFAccessKeysResult => {
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
      return undefined;
    }

    const timeoutId = window.setTimeout(() => {
      void loadKeys();
    }, 0);

    return () => window.clearTimeout(timeoutId);
  }, [isOpen, loadKeys]);

  const addKey = useCallback(async (keyValue: string) => {
    const normalizedValue = keyValue.trim();
    if (!normalizedValue) {
      setError('Key cannot be empty.');
      return false;
    }

    setSubmitting(true);
    setError(null);
    try {
      await addHFAccessKey(normalizedValue);
      setVisibleRows({});
      setRevealedValues({});
      await loadKeys();
      return true;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to add key.');
      return false;
    } finally {
      setSubmitting(false);
    }
  }, [loadKeys]);

  const toggleReveal = useCallback(async (keyId: number) => {
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
  }, [revealedValues, visibleRows]);

  const toggleActivation = useCallback(async (keyId: number, isActive: boolean) => {
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
  }, [loadKeys]);

  const deleteKey = useCallback(async (key: HFAccessKeyListItem) => {
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
  }, [loadKeys]);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const clearRevealedValues = useCallback(() => {
    setVisibleRows({});
    setRevealedValues({});
  }, []);

  return {
    actionKeyId,
    error,
    keys,
    loading,
    revealedValues,
    submitting,
    visibleRows,
    addKey,
    clearError,
    deleteKey,
    toggleActivation,
    toggleReveal,
    clearRevealedValues,
  };
};
