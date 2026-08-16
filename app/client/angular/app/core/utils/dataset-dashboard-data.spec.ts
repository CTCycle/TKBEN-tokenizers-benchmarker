import { describe, expect, it } from 'vitest';
import {
  buildWordCloudFromWordFrequencies,
  buildZipfCurveFromWordFrequencies,
  parseWordCloudTerms,
  parseWordFrequencyItems,
  parseZipfCurve,
} from './dataset-dashboard-data';

describe('dataset dashboard normalization', () => {
  it('rejects malformed frequency rows and keeps positive counts', () => {
    expect(parseWordFrequencyItems([{ word: 'alpha', count: 3 }, { word: '', count: 4 }, { word: 'zero', count: 0 }, null])).toEqual([{ word: 'alpha', count: 3 }]);
  });

  it('normalizes missing cloud weights from counts', () => {
    expect(parseWordCloudTerms([{ word: 'alpha', count: 10 }, { word: 'beta', count: 5 }]).map((item) => item.weight)).toEqual([100, 50]);
  });

  it('sorts and caps deterministic Zipf fallback data', () => {
    expect(buildZipfCurveFromWordFrequencies([{ word: 'b', count: 2 }, { word: 'a', count: 4 }])).toEqual([{ rank: 1, frequency: 4 }, { rank: 2, frequency: 2 }]);
    expect(parseZipfCurve([{ rank: 2, frequency: 4 }, { rank: 1, frequency: 5 }, { rank: 0, frequency: 2 }])).toEqual([{ rank: 1, frequency: 5 }, { rank: 2, frequency: 4 }]);
  });

  it('builds a bounded word-cloud fallback', () => {
    const terms = buildWordCloudFromWordFrequencies(Array.from({ length: 150 }, (_, index) => ({ word: `word-${index}`, count: 150 - index })));
    expect(terms).toHaveLength(120);
    expect(terms[0]?.weight).toBe(100);
  });
});
