import { useCallback, useRef } from 'react';

type UseFileInputControlResult = {
  inputRef: React.RefObject<HTMLInputElement | null>;
  openFileDialog: () => void;
  resetFileInput: () => void;
};

export const useFileInputControl = (): UseFileInputControlResult => {
  const inputRef = useRef<HTMLInputElement | null>(null);

  const openFileDialog = useCallback(() => {
    inputRef.current?.click();
  }, []);

  const resetFileInput = useCallback(() => {
    if (inputRef.current) {
      inputRef.current.value = '';
    }
  }, []);

  return {
    inputRef,
    openFileDialog,
    resetFileInput,
  };
};
