import { useEffect, useRef, useState } from 'react';
import type { RefObject } from 'react';
import type { WordCloudTerm } from '../types/api';

type WordCloudLayoutTerm = WordCloudTerm & {
  x: number;
  y: number;
  rotate: number;
  fontSize: number;
};

type WordCloudWorkerOutput = {
  terms: WordCloudLayoutTerm[];
};

type WordCloudSize = {
  width: number;
  height: number;
};

type UseWordCloudLayoutResult = {
  wordCloudRef: RefObject<HTMLDivElement | null>;
  wordCloudLayout: WordCloudLayoutTerm[];
};

const buildFallbackLayout = (
  terms: WordCloudTerm[],
  width: number,
  height: number,
): WordCloudLayoutTerm[] => {
  const visibleTerms = terms.slice(0, 48);
  const columns = Math.max(1, Math.floor(width / 150));
  const rowHeight = Math.max(34, Math.min(48, height / Math.max(1, Math.ceil(visibleTerms.length / columns))));

  return visibleTerms.map((term, index) => {
    const column = index % columns;
    const row = Math.floor(index / columns);
    return {
      ...term,
      x: Math.round(((column + 1) / (columns + 1)) * width),
      y: Math.round(Math.min(height - 18, 24 + row * rowHeight)),
      rotate: index % 5 === 0 ? -6 : index % 7 === 0 ? 6 : 0,
      fontSize: Math.max(12, Math.min(36, 12 + Math.round(term.weight * 0.18))),
    };
  });
};

export const useWordCloudLayout = (terms: WordCloudTerm[]): UseWordCloudLayoutResult => {
  const [wordCloudLayout, setWordCloudLayout] = useState<WordCloudLayoutTerm[]>([]);
  const [wordCloudSize, setWordCloudSize] = useState<WordCloudSize>({ width: 0, height: 0 });
  const wordCloudRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const node = wordCloudRef.current;
    if (!node) {
      return;
    }

    const observer = new ResizeObserver((entries) => {
      const first = entries[0];
      if (!first) {
        return;
      }
      setWordCloudSize({
        width: Math.max(260, Math.round(first.contentRect.width)),
        height: Math.max(240, Math.round(first.contentRect.height)),
      });
    });

    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!terms.length || wordCloudSize.width <= 0 || wordCloudSize.height <= 0) {
      return;
    }

    const worker = new Worker(new URL('../workers/wordCloudWorker.ts', import.meta.url), {
      type: 'module',
    });
    worker.onmessage = (event: MessageEvent<WordCloudWorkerOutput>) => {
      const nextLayout = event.data?.terms ?? [];
      setWordCloudLayout(nextLayout.length > 0
        ? nextLayout
        : buildFallbackLayout(terms, wordCloudSize.width, wordCloudSize.height));
      worker.terminate();
    };
    worker.onerror = () => {
      setWordCloudLayout(buildFallbackLayout(terms, wordCloudSize.width, wordCloudSize.height));
      worker.terminate();
    };
    worker.postMessage({
      terms,
      width: wordCloudSize.width,
      height: wordCloudSize.height,
    });

    return () => worker.terminate();
  }, [terms, wordCloudSize.height, wordCloudSize.width]);

  const normalizedLayout = !terms.length || wordCloudSize.width <= 0 || wordCloudSize.height <= 0
    ? []
    : wordCloudLayout;

  return {
    wordCloudRef,
    wordCloudLayout: normalizedLayout,
  };
};
