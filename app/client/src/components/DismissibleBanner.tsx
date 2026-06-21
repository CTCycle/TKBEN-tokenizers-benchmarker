type DismissibleBannerProps = {
  message: string;
  onDismiss: () => void;
  role?: 'alert' | 'status';
  className?: string;
  dismissLabel?: string;
};

const DismissibleBanner = ({
  message,
  onDismiss,
  role = 'alert',
  className = '',
  dismissLabel = 'Dismiss',
}: DismissibleBannerProps) => {
  const classes = className ? `error-banner ${className}` : 'error-banner';

  return (
    <div className={classes} role={role}>
      <span>{message}</span>
      <button type="button" className="banner-dismiss-button" aria-label={dismissLabel} onClick={onDismiss}>
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M6 6l12 12M18 6L6 18" strokeWidth="2" strokeLinecap="round" />
        </svg>
      </button>
    </div>
  );
};

export default DismissibleBanner;
