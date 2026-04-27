import { useEffect, useMemo, useRef, useState } from 'react';
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
      setWordCloudLayout(event.data?.terms ?? []);
      worker.terminate();
    };
    worker.postMessage({
      terms,
      width: wordCloudSize.width,
      height: wordCloudSize.height,
    });

    return () => worker.terminate();
  }, [terms, wordCloudSize.height, wordCloudSize.width]);

  const normalizedLayout = useMemo(() => {
    if (!terms.length || wordCloudSize.width <= 0 || wordCloudSize.height <= 0) {
      return [];
    }
    return wordCloudLayout;
  }, [terms, wordCloudLayout, wordCloudSize.height, wordCloudSize.width]);

  return {
    wordCloudRef,
    wordCloudLayout: normalizedLayout,
  };
};
