import type { ReactElement, ReactNode } from 'react';

export type CatalogOption<TValue extends string = string> = {
  value: TValue;
  label: string;
};

export type CatalogFilterToolbarProps<
  TSource extends string,
  TNumericOperator extends string,
> = {
  accessibleName: string;
  searchLabel: string;
  searchValue: string;
  searchPlaceholder: string;
  onSearchChange: (value: string) => void;
  sourceLabel: string;
  sourceValue: TSource;
  sourceOptions: readonly CatalogOption<TSource>[];
  onSourceChange: (value: TSource) => void;
  numericLabel: string;
  numericValue: string;
  numericOperator: TNumericOperator;
  numericPlaceholder: string;
  onNumericValueChange: (value: string) => void;
  onNumericOperatorChange: (value: TNumericOperator) => void;
  addButtonLabel: string;
  addButtonTitle: string;
  onAdd: () => void;
  numericOperatorOptions?: readonly CatalogOption<TNumericOperator>[];
  addIcon?: ReactNode;
};

const DEFAULT_NUMERIC_OPERATOR_OPTIONS: readonly CatalogOption<'at_least' | 'at_most'>[] = [
  { value: 'at_least', label: 'At least' },
  { value: 'at_most', label: 'At most' },
];

const getNumericOperatorOptions = <TNumericOperator extends string>(
  options?: readonly CatalogOption<TNumericOperator>[],
): readonly CatalogOption<TNumericOperator>[] =>
  options ?? DEFAULT_NUMERIC_OPERATOR_OPTIONS as readonly CatalogOption<TNumericOperator>[];

const CatalogFilterToolbar = <TSource extends string, TNumericOperator extends string>({
  accessibleName, searchLabel, searchValue, searchPlaceholder, onSearchChange,
  sourceLabel, sourceValue, sourceOptions, onSourceChange, numericLabel, numericValue,
  numericOperator, numericPlaceholder, onNumericValueChange, onNumericOperatorChange,
  addButtonLabel, addButtonTitle, onAdd, numericOperatorOptions, addIcon,
}: CatalogFilterToolbarProps<TSource, TNumericOperator>): ReactElement => {
  const sourceValueFromEvent = (value: string): TSource | undefined =>
    sourceOptions.find((option) => option.value === value)?.value;
  const operatorOptions = getNumericOperatorOptions(numericOperatorOptions);
  const numericOperatorValueFromEvent = (value: string): TNumericOperator | undefined =>
    operatorOptions.find((option) => option.value === value)?.value;

  return (
    <div className="catalog-filter-toolbar" aria-label={accessibleName}>
    <label className="catalog-filter-field catalog-filter-field--search">
      <span className="field-label">{searchLabel}</span>
      <input type="search" className="text-input" value={searchValue} onChange={(event) => onSearchChange(event.target.value)} placeholder={searchPlaceholder} />
    </label>
    <label className="catalog-filter-field">
      <span className="field-label">{sourceLabel}</span>
      <select className="text-input" value={sourceValue} onChange={(event) => {
        const nextValue = sourceValueFromEvent(event.target.value);
        if (nextValue !== undefined) onSourceChange(nextValue);
      }}>
        {sourceOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
      </select>
    </label>
    <div className="catalog-filter-field">
      <span className="field-label">{numericLabel}</span>
      <div className="catalog-number-filter-control">
        <select className="text-input" aria-label={`${numericLabel} comparison`} value={numericOperator} onChange={(event) => {
          const nextValue = numericOperatorValueFromEvent(event.target.value);
          if (nextValue !== undefined) onNumericOperatorChange(nextValue);
        }}>
          {operatorOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
        </select>
        <input type="number" className="text-input" value={numericValue} onChange={(event) => onNumericValueChange(event.target.value)} placeholder={numericPlaceholder} min={0} />
      </div>
    </div>
    <button type="button" className="catalog-add-button" onClick={onAdd} aria-label={addButtonLabel} title={addButtonTitle}>
      <span aria-hidden="true">{addIcon ?? '+'}</span>
      <span className="sr-only">{addButtonLabel}</span>
    </button>
    </div>
  );
};

export default CatalogFilterToolbar;
