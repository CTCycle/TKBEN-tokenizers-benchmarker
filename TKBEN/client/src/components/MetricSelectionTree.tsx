import type { MetricSelectionCategory } from '../hooks/useMetricSelection';

type MetricSelectionTreeProps = {
  categories: MetricSelectionCategory[];
  selectedMetricKeys: string[];
  onToggleMetric: (metricKey: string, enabled: boolean) => void;
  onToggleCategory: (metricKeys: string[], enabled: boolean) => void;
  rootClassName?: string;
  categoryClassName: string;
  childrenClassName: string;
};

const MetricSelectionTree = ({
  categories,
  selectedMetricKeys,
  onToggleMetric,
  onToggleCategory,
  rootClassName,
  categoryClassName,
  childrenClassName,
}: MetricSelectionTreeProps) => {
  const content = categories.map((category) => {
      const categoryKeys = category.metrics.map((metric) => metric.key);
      const selectedCount = categoryKeys.filter((key) => selectedMetricKeys.includes(key)).length;
      const categoryChecked = categoryKeys.length > 0 && selectedCount === categoryKeys.length;

      return (
        <div key={category.category_key} className={categoryClassName}>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={categoryChecked}
              onChange={(event) => onToggleCategory(categoryKeys, event.target.checked)}
            />
            <span>{category.category_label} ({selectedCount}/{categoryKeys.length})</span>
          </label>
          <div className={childrenClassName}>
            {category.metrics.map((metric) => (
              <label key={metric.key} className="checkbox">
                <input
                  type="checkbox"
                  checked={selectedMetricKeys.includes(metric.key)}
                  onChange={(event) => onToggleMetric(metric.key, event.target.checked)}
                />
                <span>{metric.label}</span>
              </label>
            ))}
          </div>
        </div>
      );
    });

  if (rootClassName) {
    return <div className={rootClassName}>{content}</div>;
  }
  return <>{content}</>;
};

export default MetricSelectionTree;
