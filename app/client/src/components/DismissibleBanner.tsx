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
      <button type="button" aria-label={dismissLabel} onClick={onDismiss}>
        ×
      </button>
    </div>
  );
};

export default DismissibleBanner;
