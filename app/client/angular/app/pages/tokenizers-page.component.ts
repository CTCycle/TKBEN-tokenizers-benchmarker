import { Component, DestroyRef, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormControl, FormGroup, ReactiveFormsModule } from '@angular/forms';
import { debounceTime } from 'rxjs';
import { TokenizersStore } from '../core/state/tokenizers.store';
import { ModalA11yDirective } from '../core/ui/modal-a11y.directive';

@Component({
  selector: 'app-tokenizers-page',
  imports: [ReactiveFormsModule, ModalA11yDirective],
  templateUrl: './tokenizers-page.component.html',
})
export class TokenizersPageComponent {
  protected readonly store = inject(TokenizersStore);
  protected readonly addTokenizerOpen = signal(false);
  protected readonly manualTokenizerInput = signal('');
  protected readonly scanQuery = signal('');
  protected readonly manualTokenizerIds = computed(() => this.manualTokenizerInput().split(/\r?\n|,/).map((item) => item.trim()).filter(Boolean));
  protected readonly selectedScannedTokenizers = signal<readonly string[]>([]);
  protected readonly filteredScannedTokenizers = computed(() => {
    const query = this.scanQuery().trim().toLowerCase();
    const values = this.store.scannedTokenizers();
    return query ? values.filter((tokenizer) => tokenizer.toLowerCase().includes(query)) : values;
  });
  protected readonly filters = new FormGroup({
    search: new FormControl('', { nonNullable: true }),
    source: new FormControl('', { nonNullable: true }),
    vocabularyOperator: new FormControl<'at_least' | 'at_most'>('at_least', { nonNullable: true }),
    vocabulary: new FormControl<number | null>(null),
  });

  constructor() {
    const destroyRef = inject(DestroyRef);
    this.filters.valueChanges.pipe(debounceTime(10), takeUntilDestroyed(destroyRef)).subscribe(() => this.refresh());
  }

  protected refresh(): void {
    const value = this.filters.getRawValue();
    this.store.refresh({
      search: value.search,
      source: value.source === 'hugging_face' ? 'huggingface' : value.source === 'custom' ? 'custom' : undefined,
      vocabularyOperator: value.vocabularyOperator,
      vocabulary: value.vocabulary ?? undefined,
    });
  }

  protected selectTokenizer(name: string): void {
    this.store.select(name);
  }

  protected openAddTokenizer(): void { this.addTokenizerOpen.set(true); }
  protected closeAddTokenizer(): void { this.addTokenizerOpen.set(false); }
  protected scanTokenizers(): void { this.store.scan(); }
  protected toggleScannedTokenizer(tokenizer: string, enabled: boolean): void {
    const next = new Set(this.selectedScannedTokenizers());
    if (enabled) next.add(tokenizer); else next.delete(tokenizer);
    this.selectedScannedTokenizers.set([...next]);
  }
  protected downloadManualTokenizers(): void {
    const tokenizers = this.manualTokenizerInput().split(/\r?\n|,/).map((item) => item.trim()).filter(Boolean);
    if (tokenizers.length) this.store.download({ tokenizers });
  }
  protected downloadScannedTokenizers(): void {
    const tokenizers = [...this.selectedScannedTokenizers()];
    if (tokenizers.length) this.store.download({ tokenizers });
  }
  protected generateReport(name: string): void { this.store.generateReport(name); }
  protected loadLatest(name: string): void { this.store.loadLatest(name); }
  protected removeTokenizer(name: string): void { if (window.confirm(`Remove ${name} from the database?`)) this.store.remove(name); }
  protected uploadFile(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (file) { this.store.upload(file); this.addTokenizerOpen.set(false); input.value = ''; }
  }
}
