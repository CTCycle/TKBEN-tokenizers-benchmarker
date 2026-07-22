import type { ReactNode } from 'react';

type CatalogOption = { value: string; label: string };
type Props = {
  accessibleName: string;
  searchLabel: string;
  searchValue: string;
  searchPlaceholder: string;
  onSearchChange: (value: string) => void;
  sourceLabel: string;
  sourceValue: string;
  sourceOptions: CatalogOption[];
  onSourceChange: (value: string) => void;
  numericLabel: string;
  numericValue: string;
  numericOperator: string;
  numericPlaceholder: string;
  onNumericValueChange: (value: string) => void;
  onNumericOperatorChange: (value: string) => void;
  addButtonLabel: string;
  addButtonTitle: string;
  onAdd: () => void;
  numericOperatorOptions?: CatalogOption[];
  addIcon?: ReactNode;
};

const CatalogFilterToolbar = ({
  accessibleName, searchLabel, searchValue, searchPlaceholder, onSearchChange,
  sourceLabel, sourceValue, sourceOptions, onSourceChange, numericLabel, numericValue,
  numericOperator, numericPlaceholder, onNumericValueChange, onNumericOperatorChange,
  addButtonLabel, addButtonTitle, onAdd, numericOperatorOptions = [
    { value: 'at_least', label: 'At least' }, { value: 'at_most', label: 'At most' },
  ], addIcon,
}: Props) => (
  <div className="catalog-filter-toolbar" aria-label={accessibleName}>
    <label className="catalog-filter-field catalog-filter-field--search">
      <span className="field-label">{searchLabel}</span>
      <input type="search" className="text-input" value={searchValue} onChange={(event) => onSearchChange(event.target.value)} placeholder={searchPlaceholder} />
    </label>
    <label className="catalog-filter-field">
      <span className="field-label">{sourceLabel}</span>
      <select className="text-input" value={sourceValue} onChange={(event) => onSourceChange(event.target.value)}>
        {sourceOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
      </select>
    </label>
    <div className="catalog-filter-field">
      <span className="field-label">{numericLabel}</span>
      <div className="catalog-number-filter-control">
        <select className="text-input" aria-label={`${numericLabel} comparison`} value={numericOperator} onChange={(event) => onNumericOperatorChange(event.target.value)}>
          {numericOperatorOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
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

export default CatalogFilterToolbar;
