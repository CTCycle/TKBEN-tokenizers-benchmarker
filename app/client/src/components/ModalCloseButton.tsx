import type { ReactElement } from 'react';

type ModalCloseButtonProps = {
  ariaLabel: string;
  onClick: () => void;
  title?: string;
  disabled?: boolean;
};

const ModalCloseButton = ({
  ariaLabel,
  onClick,
  title,
  disabled = false,
}: ModalCloseButtonProps): ReactElement => (
  <button
    type="button"
    className="icon-button subtle modal-close-button"
    onClick={onClick}
    aria-label={ariaLabel}
    title={title}
    disabled={disabled}
  >
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M6 6l12 12M18 6L6 18" strokeWidth="2" strokeLinecap="round" />
    </svg>
  </button>
);

export default ModalCloseButton;
