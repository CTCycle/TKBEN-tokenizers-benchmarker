import { API_ENDPOINTS } from '../constants';
import type {
    HFAccessKeyListItem,
    HFAccessKeyListResponse,
    HFAccessKeyRevealResponse,
} from '../types/api';

/**
 * Fetch all stored Hugging Face keys (masked previews only).
 */
export async function fetchHFAccessKeys(): Promise<HFAccessKeyListResponse> {
    const response = await fetch(API_ENDPOINTS.KEYS);
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Failed to fetch keys: ${response.status}`);
    }
    return response.json();
}

/**
 * Add a new Hugging Face key.
 */
export async function addHFAccessKey(rawKey: string): Promise<HFAccessKeyListItem> {
    const response = await fetch(API_ENDPOINTS.KEYS, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ key_value: rawKey }),
    });
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Failed to add key: ${response.status}`);
    }
    return response.json();
}

/**
 * Toggle selected key active state (active -> inactive, inactive -> active).
 */
export async function activateHFAccessKey(keyId: number): Promise<void> {
    const response = await fetch(`${API_ENDPOINTS.KEYS}/${keyId}/activate`, {
        method: 'POST',
    });
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Failed to activate key: ${response.status}`);
    }
}

/**
 * Explicitly deactivate selected key (sets it to inactive).
 */
export async function deactivateHFAccessKey(keyId: number): Promise<void> {
    const response = await fetch(`${API_ENDPOINTS.KEYS}/${keyId}/deactivate`, {
        method: 'POST',
    });
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Failed to deactivate key: ${response.status}`);
    }
}

/**
 * Request full stored encrypted key value for a single row.
 */
export async function revealHFAccessKey(keyId: number): Promise<HFAccessKeyRevealResponse> {
    const response = await fetch(`${API_ENDPOINTS.KEYS}/${keyId}/reveal`, {
        method: 'POST',
    });
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Failed to reveal key: ${response.status}`);
    }
    return response.json();
}

/**
 * Delete a stored key with explicit confirmation.
 */
export async function deleteHFAccessKey(keyId: number): Promise<void> {
    const response = await fetch(`${API_ENDPOINTS.KEYS}/${keyId}?confirm=true`, {
        method: 'DELETE',
    });
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Failed to delete key: ${response.status}`);
    }
}
