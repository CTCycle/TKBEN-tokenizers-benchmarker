import { useMemo, useState } from 'react';

import { exportDashboardPdf } from '../services/exportApi';
import type { DashboardType } from '../types/api';

interface FileSystemWritableFileStreamLike {
    write(data: Blob | BufferSource | string): Promise<void>;
    close(): Promise<void>;
}

interface FileSystemFileHandleLike {
    name?: string;
    createWritable(): Promise<FileSystemWritableFileStreamLike>;
}

interface FileSystemSavePickerOptionsLike {
    suggestedName?: string;
    types?: Array<{
        description?: string;
        accept: Record<string, string[]>;
    }>;
}

type SavePickerWindow = Window & {
    showSaveFilePicker?: (options?: FileSystemSavePickerOptionsLike) => Promise<FileSystemFileHandleLike>;
};

type DashboardExportButtonProps = {
    dashboardType: DashboardType;
    reportName: string;
    dashboardPayload: Record<string, unknown> | null;
};

const DEFAULT_FILE_STEMS: Record<DashboardType, string> = {
    dataset: 'dataset-report',
    tokenizer: 'tokenizer-report',
    benchmark: 'benchmark-report',
};

const sanitizeFileStem = (value: string, fallback: string): string => {
    const normalized = value.trim().replace(/\.pdf$/i, '');
    const cleaned = normalized
        .replace(/[\\/]/g, '_')
        .replace(/[^A-Za-z0-9._ ()-]+/g, '_')
        .replace(/_+/g, '_')
        .replace(/^[\s._-]+|[\s._-]+$/g, '');
    return cleaned || fallback;
};

const toPdfFileName = (value: string): string =>
    value.toLowerCase().endsWith('.pdf') ? value : `${value}.pdf`;

const downloadPdfBlob = (blob: Blob, fileName: string) => {
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = fileName;
    link.click();
    URL.revokeObjectURL(url);
};

const DashboardExportButton = ({
    dashboardType,
    reportName,
    dashboardPayload,
}: DashboardExportButtonProps) => {
    const fallbackStem = DEFAULT_FILE_STEMS[dashboardType];
    const defaultStem = useMemo(
        () => sanitizeFileStem(reportName, fallbackStem),
        [fallbackStem, reportName],
    );

    const [submitting, setSubmitting] = useState(false);
    const handleExport = async () => {
        if (!dashboardPayload) {
            window.alert('No dashboard report data available to export.');
            return;
        }
        const normalizedFileName = toPdfFileName(defaultStem);

        setSubmitting(true);
        try {
            const pickerWindow = window as SavePickerWindow;
            const response = await exportDashboardPdf({
                dashboardType,
                reportName,
                fileName: normalizedFileName,
                dashboardPayload,
            });

            if (!pickerWindow.showSaveFilePicker) {
                downloadPdfBlob(response.blob, response.fileName);
                return;
            }

            const fileHandle = await pickerWindow.showSaveFilePicker({
                suggestedName: response.fileName,
                types: [
                    {
                        description: 'PDF document',
                        accept: { 'application/pdf': ['.pdf'] },
                    },
                ],
            });
            const writable = await fileHandle.createWritable();
            await writable.write(response.blob);
            await writable.close();
        } catch (submissionError) {
            if (submissionError instanceof DOMException && submissionError.name === 'AbortError') {
                return;
            }
            const message = submissionError instanceof Error
                ? submissionError.message
                : 'Failed to export dashboard.';
            window.alert(message);
        } finally {
            setSubmitting(false);
        }
    };

    return (
        <button
            type="button"
            className="icon-button subtle dashboard-export-trigger"
            onClick={() => { void handleExport(); }}
            disabled={submitting || !dashboardPayload}
            title="Export dashboard report as PDF"
            aria-label="Export dashboard report as PDF"
        >
            <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M12 3v10" strokeWidth="2" strokeLinecap="round" />
                <path d="M8 9l4 4 4-4" strokeWidth="2" strokeLinecap="round" />
                <path d="M5 16.5V20h14v-3.5" strokeWidth="2" strokeLinecap="round" />
            </svg>
        </button>
    );
};

export default DashboardExportButton;

