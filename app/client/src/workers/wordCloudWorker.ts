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
const inflateBox = (box: Box, padding: number): Box => ({
  left: box.left - padding,
  right: box.right + padding,
  top: box.top - padding,
  bottom: box.bottom + padding,
});

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
  const centerExclusionRadius = Math.max(28, Math.min(width, height) * 0.14);

  sorted.forEach((term, index) => {
    const fontSize = Math.max(12, Math.min(44, 10 + Math.round(term.weight * 0.24)));
    const wordWidth = Math.max(fontSize, Math.round(term.word.length * fontSize * 0.56));
    const wordHeight = Math.max(12, Math.round(fontSize * 1.16));
    const rotation = index % 7 === 0 ? -8 : index % 9 === 0 ? 8 : 0;
    const collisionPadding = Math.max(3, Math.round(fontSize * 0.12));

    let placedX = centerX;
    let placedY = centerY;
    let found = false;
    const startAngle = index * 0.57;
    for (let step = 0; step < 960; step += 1) {
      const angle = startAngle + step * 0.33;
      const radius = 4 + step * 1.2;
      const candidateX = Math.round(centerX + Math.cos(angle) * radius);
      const candidateY = Math.round(centerY + Math.sin(angle) * radius);
      const centerDistance = Math.hypot(candidateX - centerX, candidateY - centerY);
      if (index > 0 && centerDistance < centerExclusionRadius) {
        continue;
      }
      const box: Box = {
        left: candidateX - wordWidth / 2,
        right: candidateX + wordWidth / 2,
        top: candidateY - wordHeight / 2,
        bottom: candidateY + wordHeight / 2,
      };
      if (box.left < 0 || box.right > width || box.top < 0 || box.bottom > height) {
        continue;
      }
      const paddedBox = inflateBox(box, collisionPadding);
      if (placed.every((item) => !intersects(item, paddedBox))) {
        placedX = candidateX;
        placedY = candidateY;
        placed.push(paddedBox);
        found = true;
        break;
      }
    }

    if (!found) {
      return;
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
