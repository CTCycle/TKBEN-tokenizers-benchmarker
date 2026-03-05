type ChartPlaceholderProps = {
  message: string;
  detail?: string;
  className?: string;
};

const ChartPlaceholder = ({ message, detail, className }: ChartPlaceholderProps) => {
  const classes = className ? `chart-placeholder ${className}` : 'chart-placeholder';

  return (
    <div className={classes}>
      <p>{message}</p>
      {detail ? <span>{detail}</span> : null}
    </div>
  );
};

export default ChartPlaceholder;
