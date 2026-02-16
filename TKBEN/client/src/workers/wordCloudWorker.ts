type WordCloudInput = {
  terms: Array<{ word: string; count: number; weight: number }>;
  width: number;
  height: number;
};

type WordCloudOutput = {
  terms: Array<{
    word: string;
    count: number;
    weight: number;
    x: number;
    y: number;
    rotate: number;
    fontSize: number;
  }>;
};

type Box = {
  left: number;
  right: number;
  top: number;
  bottom: number;
};

const intersects = (a: Box, b: Box): boolean =>
  !(a.right < b.left || a.left > b.right || a.bottom < b.top || a.top > b.bottom);

const computeLayout = (input: WordCloudInput): WordCloudOutput => {
  const width = Math.max(240, Math.round(input.width || 0));
  const height = Math.max(220, Math.round(input.height || 0));
  const sorted = [...input.terms]
    .sort((a, b) => b.weight - a.weight || a.word.localeCompare(b.word))
    .slice(0, 140);

  const result: WordCloudOutput['terms'] = [];
  const placed: Box[] = [];
  const centerX = width / 2;
  const centerY = height / 2;

  sorted.forEach((term, index) => {
    const fontSize = Math.max(12, Math.min(46, 10 + Math.round(term.weight * 0.28)));
    const wordWidth = Math.max(fontSize, Math.round(term.word.length * fontSize * 0.56));
    const wordHeight = Math.round(fontSize * 1.2);
    const rotation = index % 7 === 0 ? -8 : index % 9 === 0 ? 8 : 0;

    let placedX = centerX;
    let placedY = centerY;
    let found = false;
    for (let step = 0; step < 420; step += 1) {
      const angle = step * 0.41;
      const radius = 2 + step * 1.55;
      const candidateX = Math.round(centerX + Math.cos(angle) * radius);
      const candidateY = Math.round(centerY + Math.sin(angle) * radius);
      const box: Box = {
        left: candidateX - wordWidth / 2,
        right: candidateX + wordWidth / 2,
        top: candidateY - wordHeight / 2,
        bottom: candidateY + wordHeight / 2,
      };
      if (box.left < 0 || box.right > width || box.top < 0 || box.bottom > height) {
        continue;
      }
      if (placed.every((item) => !intersects(item, box))) {
        placedX = candidateX;
        placedY = candidateY;
        placed.push(box);
        found = true;
        break;
      }
    }

    if (!found) {
      const fallback: Box = {
        left: Math.max(0, centerX - wordWidth / 2),
        right: Math.min(width, centerX + wordWidth / 2),
        top: Math.max(0, centerY - wordHeight / 2),
        bottom: Math.min(height, centerY + wordHeight / 2),
      };
      placed.push(fallback);
    }

    result.push({
      ...term,
      x: placedX,
      y: placedY,
      rotate: rotation,
      fontSize,
    });
  });

  return { terms: result };
};

self.onmessage = (event: MessageEvent<WordCloudInput>) => {
  const payload = computeLayout(event.data);
  self.postMessage(payload);
};

export {};
