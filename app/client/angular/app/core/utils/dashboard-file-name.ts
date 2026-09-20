const unsafeFileCharacters = /[^A-Za-z0-9._ ()-]+/g;

export function dashboardFileName(prefix: string, identifier: string, suffix = 'report'): string {
  const safeIdentifier = identifier
    .replace(/[\\/]+/g, '-')
    .replace(unsafeFileCharacters, '_')
    .replace(/^[-._ ]+|[-._ ]+$/g, '') || 'dashboard';
  return `${prefix}-${safeIdentifier}-${suffix}.pdf`;
}
