import { Component, DestroyRef, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormControl, FormGroup, ReactiveFormsModule } from '@angular/forms';
import { debounceTime } from 'rxjs';
import { TokenizersStore } from '../core/state/tokenizers.store';

@Component({
  selector: 'app-tokenizers-page',
  imports: [ReactiveFormsModule],
  templateUrl: './tokenizers-page.component.html',
})
export class TokenizersPageComponent {
  protected readonly store = inject(TokenizersStore);
  protected readonly addTokenizerOpen = signal(false);
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
  protected generateReport(name: string): void { this.store.generateReport(name); }
  protected loadLatest(name: string): void { this.store.loadLatest(name); }
  protected removeTokenizer(name: string): void { if (window.confirm(`Remove ${name} from the database?`)) this.store.remove(name); }
  protected uploadFile(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (file) { this.store.upload(file); this.addTokenizerOpen.set(false); input.value = ''; }
  }
}
