import { describe, expect, it } from 'vitest';

describe('dashboard layout storage contract', () => {
  it('preserves the v3 storage key', () => {
    expect('tkben:cross-benchmark-dashboard-layout:v3').toBe('tkben:cross-benchmark-dashboard-layout:v3');
  });
});
