import { DOCUMENT } from '@angular/common';
import { LiveAnnouncer } from '@angular/cdk/a11y';
import { Directive, ElementRef, EventEmitter, Input, OnDestroy, OnInit, Output, inject } from '@angular/core';

@Directive({ selector: '[appModalA11y]' })
export class ModalA11yDirective implements OnInit, OnDestroy {
  @Input({ alias: 'appModalA11y' }) modalA11y = true;
  @Output() readonly modalEscape = new EventEmitter<void>();
  private active = false;
  private previousFocused: HTMLElement | null = null;
  private previousOverflow = '';
  private removeKeydown: (() => void) | null = null;

  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  private readonly document = inject(DOCUMENT);
  private readonly announcer = inject(LiveAnnouncer);

  ngOnInit(): void {
    if (this.modalA11y) this.activate();
  }

  ngOnDestroy(): void {
    this.deactivate();
  }

  private activate(): void {
    if (this.active) return;
    this.active = true;
    this.previousFocused = this.document.activeElement instanceof HTMLElement ? this.document.activeElement : null;
    this.previousOverflow = this.document.body.style.overflow;
    this.document.body.style.overflow = 'hidden';
    this.removeKeydown = this.listenKeydown();
    queueMicrotask(() => this.focusInitialElement());
    void this.announcer.announce('Dialog opened', 'polite');
  }

  private deactivate(): void {
    if (!this.active) return;
    this.active = false;
    this.removeKeydown?.();
    this.removeKeydown = null;
    this.document.body.style.overflow = this.previousOverflow;
    const target = this.previousFocused;
    this.previousFocused = null;
    if (target && typeof target.focus === 'function') queueMicrotask(() => target.focus());
    void this.announcer.announce('Dialog closed', 'polite');
  }

  private focusableElements(): HTMLElement[] {
    const selector = 'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';
    return Array.from(this.host.nativeElement.querySelectorAll<HTMLElement>(selector)).filter((element) => !element.hasAttribute('aria-hidden'));
  }

  private focusInitialElement(): void {
    const target = this.host.nativeElement.querySelector<HTMLElement>('[autofocus], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled])');
    target?.focus();
  }

  private listenKeydown(): () => void {
    const listener = (event: KeyboardEvent): void => {
      if (!this.active) return;
      if (event.key === 'Escape') {
        event.preventDefault();
        this.modalEscape.emit();
        return;
      }
      if (event.key !== 'Tab') return;
      const elements = this.focusableElements();
      if (!elements.length) {
        event.preventDefault();
        this.host.nativeElement.focus();
        return;
      }
      const first = elements[0];
      const last = elements[elements.length - 1];
      if (event.shiftKey && this.document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && this.document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    this.document.addEventListener('keydown', listener);
    return () => this.document.removeEventListener('keydown', listener);
  }
}
