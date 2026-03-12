import DismissibleBanner from './DismissibleBanner';

type TokenizerStatusBaseProps = {
  scanError: string | null;
  benchmarkError: string | null;
  onDismissScanError: () => void;
  onDismissBenchmarkError: () => void;
  containerClassName?: string;
};

type TokenizerStatusWithWarningProps = {
  warningMode: 'with-warning';
  downloadWarning: string | null;
  onDismissDownloadWarning: () => void;
  warningClassName?: string;
};

type TokenizerStatusWithoutWarningProps = {
  warningMode: 'without-warning';
};

type TokenizerStatusBannersProps = TokenizerStatusBaseProps & (
  | TokenizerStatusWithWarningProps
  | TokenizerStatusWithoutWarningProps
);

const TokenizerStatusBanners = ({
  scanError,
  benchmarkError,
  onDismissScanError,
  onDismissBenchmarkError,
  containerClassName,
  ...warningProps
}: TokenizerStatusBannersProps) => {
  const showDownloadWarning = warningProps.warningMode === 'with-warning' && Boolean(warningProps.downloadWarning);
  const hasAnyMessage = showDownloadWarning || Boolean(scanError) || Boolean(benchmarkError);

  if (!hasAnyMessage) {
    return null;
  }

  const content = (
    <>
      {warningProps.warningMode === 'with-warning' && warningProps.downloadWarning && (
        <DismissibleBanner
          message={warningProps.downloadWarning}
          onDismiss={warningProps.onDismissDownloadWarning}
          role="status"
          className={warningProps.warningClassName}
        />
      )}
      {scanError && (
        <DismissibleBanner message={scanError} onDismiss={onDismissScanError} />
      )}
      {benchmarkError && (
        <DismissibleBanner message={benchmarkError} onDismiss={onDismissBenchmarkError} />
      )}
    </>
  );

  if (containerClassName) {
    return <div className={containerClassName}>{content}</div>;
  }

  return content;
};

export default TokenizerStatusBanners;
