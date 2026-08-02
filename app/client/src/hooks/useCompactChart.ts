import { useEffect, useState } from 'react';

export const useCompactChart = (): boolean => {
  const [compact, setCompact] = useState(
    () => typeof window !== 'undefined' && window.innerWidth <= 700,
  );

  useEffect(() => {
    const update = () => setCompact(window.innerWidth <= 700);
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, []);

  return compact;
};
