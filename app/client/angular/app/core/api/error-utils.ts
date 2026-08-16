type JsonRecord = Record<string, unknown>;

const isRecord = (value: unknown): value is JsonRecord =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

export function formatApiError(detail: unknown, fallback: string): string {
  if (typeof detail === 'string' && detail.trim()) return detail;
  if (!Array.isArray(detail)) return fallback;

  const messages = detail.flatMap((item) => {
    if (!isRecord(item) || typeof item['msg'] !== 'string') return [];
    const location = Array.isArray(item['loc'])
      ? item['loc'].filter((part): part is string | number => typeof part === 'string' || typeof part === 'number').join('.')
      : '';
    return [location ? `${location}: ${item['msg']}` : item['msg']];
  });
  return messages.length > 0 ? messages.join(' ') : fallback;
}

export function errorMessage(error: unknown, fallback: string): string {
  if (isRecord(error) && isRecord(error['error'])) {
    return formatApiError(error['error']['detail'], fallback);
  }
  if (error instanceof Error && error.message.trim()) return error.message;
  return fallback;
}

export async function errorMessageAsync(error: unknown, fallback: string): Promise<string> {
  if (isRecord(error) && error['error'] instanceof Blob) {
    try {
      const raw = await (error['error'] as Blob).text();
      if (raw.trim()) return errorMessage({ error: JSON.parse(raw) }, fallback);
    } catch {
      // A non-JSON blob has no structured detail; keep the caller's fallback.
    }
  }
  return errorMessage(error, fallback);
}
