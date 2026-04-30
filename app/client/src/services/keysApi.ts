import { API_ENDPOINTS } from '../common/constants/api';
import type {
    HFAccessKeyListItem,
    HFAccessKeyListResponse,
    HFAccessKeyRevealResponse,
} from '../types/api';

const formatApiErrorDetail = (detail: unknown): string | null => {
    if (typeof detail === 'string' && detail.trim()) {
        return detail;
    }
    if (Array.isArray(detail)) {
        const messages = detail
            .map((item) => {
                if (!item || typeof item !== 'object') {
                    return null;
                }
                const payload = item as Record<string, unknown>;
                const field = Array.isArray(payload.loc)
                    ? payload.loc.filter((part) => typeof part === 'string' || typeof part === 'number').join('.')
                    : '';
                const message = typeof payload.msg === 'string' ? payload.msg : null;
                if (!message) {
                    return null;
                }
                return field ? `${field}: ${message}` : message;
            })
            .filter((message): message is string => message !== null);
        return messages.length > 0 ? messages.join(' ') : null;
    }
    if (detail && typeof detail === 'object') {
        const payload = detail as Record<string, unknown>;
        if (typeof payload.message === 'string') {
            return payload.message;
        }
        if (typeof payload.msg === 'string') {
            return payload.msg;
        }
    }
    return null;
};

const readKeyApiError = async (response: Response, fallback: string): Promise<Error> => {
    const errorData = await response.json().catch((): { detail: string } => ({ detail: 'Unknown error' }));
    const detail = errorData && typeof errorData === 'object'
        ? (errorData as Record<string, unknown>).detail
        : errorData;
    return new Error(formatApiErrorDetail(detail) || `${fallback}: ${response.status}`);
};

/**
 * Fetch all stored Hugging Face keys (masked previews only).
 */
export async function fetchHFAccessKeys(): Promise<HFAccessKeyListResponse> {
    const response = await fetch(API_ENDPOINTS.KEYS);
    if (!response.ok) {
        throw await readKeyApiError(response, 'Failed to fetch keys');
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
        throw await readKeyApiError(response, 'Failed to add key');
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
        throw await readKeyApiError(response, 'Failed to activate key');
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
        throw await readKeyApiError(response, 'Failed to deactivate key');
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
        throw await readKeyApiError(response, 'Failed to reveal key');
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
        throw await readKeyApiError(response, 'Failed to delete key');
    }
}
