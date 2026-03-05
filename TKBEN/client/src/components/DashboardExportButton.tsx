import { useMemo, useState } from 'react';
import { createPortal } from 'react-dom';

import { exportDashboardPdf } from '../services/exportApi';
import type { DashboardType } from '../types/api';

type FileSystemPermissionMode = 'read' | 'readwrite';
type FileSystemPermissionState = 'granted' | 'denied' | 'prompt';

interface FileSystemWritableFileStreamLike {
    write(data: Blob | BufferSource | string): Promise<void>;
    close(): Promise<void>;
}

interface FileSystemFileHandleLike {
    createWritable(): Promise<FileSystemWritableFileStreamLike>;
}

interface FileSystemDirectoryHandleLike {
    name: string;
    getFileHandle(name: string, options?: { create?: boolean }): Promise<FileSystemFileHandleLike>;
    queryPermission?: (options?: { mode?: FileSystemPermissionMode }) => Promise<FileSystemPermissionState>;
    requestPermission?: (options?: { mode?: FileSystemPermissionMode }) => Promise<FileSystemPermissionState>;
}

type DirectoryPickerWindow = Window & {
    showDirectoryPicker?: () => Promise<FileSystemDirectoryHandleLike>;
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

const ensureWritePermission = async (directoryHandle: FileSystemDirectoryHandleLike): Promise<void> => {
    if (directoryHandle.queryPermission) {
        const state = await directoryHandle.queryPermission({ mode: 'readwrite' });
        if (state === 'granted') {
            return;
        }
    }
    if (directoryHandle.requestPermission) {
        const requestState = await directoryHandle.requestPermission({ mode: 'readwrite' });
        if (requestState === 'granted') {
            return;
        }
    }
    throw new Error('Write permission to selected folder was denied.');
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

    const [isDialogOpen, setIsDialogOpen] = useState(false);
    const [fileName, setFileName] = useState(defaultStem);
    const [directoryName, setDirectoryName] = useState<string>('');
    const [directoryHandle, setDirectoryHandle] = useState<FileSystemDirectoryHandleLike | null>(null);
    const [submitting, setSubmitting] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [successMessage, setSuccessMessage] = useState<string | null>(null);

    const resetFeedback = () => {
        setError(null);
        setSuccessMessage(null);
    };

    const openDialog = () => {
        setFileName(defaultStem);
        resetFeedback();
        setIsDialogOpen(true);
    };

    const closeDialog = () => {
        setIsDialogOpen(false);
        resetFeedback();
    };

    const pickFolder = async () => {
        resetFeedback();
        const pickerWindow = window as DirectoryPickerWindow;
        if (!pickerWindow.showDirectoryPicker) {
            setError('Folder picker is not supported in this browser/runtime.');
            return;
        }
        try {
            const pickedDirectory = await pickerWindow.showDirectoryPicker();
            setDirectoryHandle(pickedDirectory);
            setDirectoryName(pickedDirectory.name);
        } catch (pickError) {
            if (pickError instanceof DOMException && pickError.name === 'AbortError') {
                return;
            }
            setError(pickError instanceof Error ? pickError.message : 'Failed to open folder picker.');
        }
    };

    const handleExport = async () => {
        if (!dashboardPayload) {
            setError('No dashboard report data available to export.');
            return;
        }
        if (!directoryHandle) {
            setError('Please choose an output folder before exporting.');
            return;
        }

        const normalizedFileName = toPdfFileName(sanitizeFileStem(fileName, defaultStem));

        setSubmitting(true);
        resetFeedback();
        try {
            const response = await exportDashboardPdf({
                dashboardType,
                reportName,
                fileName: normalizedFileName,
                dashboardPayload,
            });
            await ensureWritePermission(directoryHandle);
            const fileHandle = await directoryHandle.getFileHandle(response.fileName, { create: true });
            const writable = await fileHandle.createWritable();
            await writable.write(response.blob);
            await writable.close();
            setSuccessMessage(
                `Report saved to ${directoryName || directoryHandle.name}\\${response.fileName} (${response.pageCount} pages).`,
            );
        } catch (submissionError) {
            const message = submissionError instanceof Error
                ? submissionError.message
                : 'Failed to export dashboard.';
            setError(message);
        } finally {
            setSubmitting(false);
        }
    };

    const fallbackDownload = async () => {
        if (!dashboardPayload) {
            setError('No dashboard report data available to export.');
            return;
        }
        const normalizedFileName = toPdfFileName(sanitizeFileStem(fileName, defaultStem));
        setSubmitting(true);
        resetFeedback();
        try {
            const response = await exportDashboardPdf({
                dashboardType,
                reportName,
                fileName: normalizedFileName,
                dashboardPayload,
            });
            downloadPdfBlob(response.blob, response.fileName);
            setSuccessMessage(`Report downloaded as ${response.fileName}.`);
        } catch (downloadError) {
            setError(downloadError instanceof Error ? downloadError.message : 'Failed to download dashboard PDF.');
        } finally {
            setSubmitting(false);
        }
    };

    return (
        <>
            <button
                type="button"
                className="icon-button subtle dashboard-export-trigger"
                onClick={openDialog}
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

            {isDialogOpen && createPortal(
                <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby={`export-${dashboardType}-title`}>
                    <div className="modal-card dashboard-export-modal">
                        <header className="dashboard-export-header">
                            <div>
                                <p id={`export-${dashboardType}-title`} className="panel-label">Export Dashboard PDF</p>
                                <p className="panel-description">
                                    Generate a fresh PDF report and save it into a selected folder.
                                </p>
                            </div>
                        </header>

                        <div className="dashboard-export-form">
                            <div className="dashboard-export-folder-row">
                                <label className="input-stack dashboard-export-folder-input" htmlFor={`export-folder-${dashboardType}`}>
                                    <span className="field-label">Output Folder</span>
                                    <input
                                        id={`export-folder-${dashboardType}`}
                                        className="text-input"
                                        value={directoryName}
                                        readOnly
                                        placeholder="No folder selected"
                                        disabled={submitting}
                                    />
                                </label>
                                <button
                                    type="button"
                                    className="secondary-button dashboard-export-folder-button"
                                    onClick={() => { void pickFolder(); }}
                                    disabled={submitting}
                                >
                                    Browse...
                                </button>
                            </div>

                            <label className="input-stack" htmlFor={`export-file-name-${dashboardType}`}>
                                <span className="field-label">File Name</span>
                                <input
                                    id={`export-file-name-${dashboardType}`}
                                    className="text-input"
                                    value={fileName}
                                    onChange={(event) => setFileName(event.target.value)}
                                    placeholder={defaultStem}
                                    disabled={submitting}
                                />
                            </label>

                            {error && (
                                <div className="error-banner dashboard-export-feedback">
                                    <span>{error}</span>
                                </div>
                            )}
                            {successMessage && (
                                <div className="dashboard-export-success">
                                    {successMessage}
                                </div>
                            )}
                        </div>

                        <div className="modal-footer dashboard-export-footer">
                            <button
                                type="button"
                                className="secondary-button"
                                onClick={closeDialog}
                                disabled={submitting}
                            >
                                Close
                            </button>
                            {!((window as DirectoryPickerWindow).showDirectoryPicker) && (
                                <button
                                    type="button"
                                    className="secondary-button"
                                    onClick={() => { void fallbackDownload(); }}
                                    disabled={submitting}
                                >
                                    Download PDF
                                </button>
                            )}
                            <button
                                type="button"
                                className="primary-button"
                                onClick={() => { void handleExport(); }}
                                disabled={submitting}
                            >
                                {submitting ? 'Exporting...' : 'Export PDF'}
                            </button>
                        </div>
                    </div>
                </div>,
                document.body,
            )}
        </>
    );
};

export default DashboardExportButton;

