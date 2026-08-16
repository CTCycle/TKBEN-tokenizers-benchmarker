import { Component, EventEmitter, Output, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { DestroyRef } from '@angular/core';
import { FormControl, ReactiveFormsModule, Validators } from '@angular/forms';
import { KeysApiService } from '../core/api/keys-api.service';
import { errorMessage } from '../core/api/error-utils';
import type { HFAccessKeyListItem } from '../core/api/api.models';
import { ModalA11yDirective } from '../core/ui/modal-a11y.directive';

@Component({
  selector: 'app-hf-access-key-manager',
  imports: [ReactiveFormsModule, ModalA11yDirective],
  templateUrl: './hf-access-key-manager.component.html',
})
export class HfAccessKeyManagerComponent {
  private readonly api = inject(KeysApiService);
  private readonly destroyRef = inject(DestroyRef);
  @Output() readonly closed = new EventEmitter<void>();
  protected readonly key = new FormControl('', { nonNullable: true, validators: [Validators.required] });
  protected readonly keys = signal<readonly HFAccessKeyListItem[]>([]);
  protected readonly revealed = signal<Record<number, string>>({});
  protected readonly loading = signal(true);
  protected readonly submitting = signal(false);
  protected readonly actionKeyId = signal<number | null>(null);
  protected readonly error = signal<string | null>(null);

  constructor() { this.load(); }

  private load(): void {
    this.loading.set(true);
    this.api.list().pipe(takeUntilDestroyed(this.destroyRef)).subscribe({ next: (response) => { this.keys.set(response.keys); this.loading.set(false); }, error: (error: unknown) => { this.error.set(errorMessage(error, 'Failed to load keys.')); this.loading.set(false); } });
  }

  protected addKey(): void {
    if (this.key.invalid) {
      this.key.markAsTouched();
      return;
    }
    this.submitting.set(true);
    this.api.add(this.key.value).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({ next: () => { this.key.reset(); this.submitting.set(false); this.load(); }, error: (error: unknown) => { this.error.set(errorMessage(error, 'Failed to add key.')); this.submitting.set(false); } });
  }

  protected toggleReveal(item: HFAccessKeyListItem): void {
    const existing = this.revealed()[item.id];
    if (existing) { const next = { ...this.revealed() }; delete next[item.id]; this.revealed.set(next); return; }
    this.actionKeyId.set(item.id);
    this.api.reveal(item.id).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({ next: (response) => { this.revealed.set({ ...this.revealed(), [item.id]: response.key_value }); this.actionKeyId.set(null); }, error: (error: unknown) => { this.error.set(errorMessage(error, 'Failed to reveal key.')); this.actionKeyId.set(null); } });
  }

  protected toggleActivation(item: HFAccessKeyListItem): void {
    this.actionKeyId.set(item.id);
    const request$ = item.is_active ? this.api.deactivate(item.id) : this.api.activate(item.id);
    request$.pipe(takeUntilDestroyed(this.destroyRef)).subscribe({ next: () => { this.actionKeyId.set(null); this.load(); }, error: (error: unknown) => { this.error.set(errorMessage(error, 'Failed to update key.')); this.actionKeyId.set(null); } });
  }

  protected deleteKey(item: HFAccessKeyListItem): void {
    if (!window.confirm(`Delete ${item.masked_preview}?`)) return;
    this.actionKeyId.set(item.id);
    this.api.remove(item.id).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({ next: () => { this.actionKeyId.set(null); this.load(); }, error: (error: unknown) => { this.error.set(errorMessage(error, 'Failed to delete key.')); this.actionKeyId.set(null); } });
  }

  protected close(): void {
    this.closed.emit();
  }
}
