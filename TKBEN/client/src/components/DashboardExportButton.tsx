import { useMemo, useState } from 'react';
import type { RefObject } from 'react';
import html2canvas from 'html2canvas';
import { createPortal } from 'react-dom';

import { exportDashboardPdf } from '../services/exportApi';
import type { DashboardType } from '../types/api';

type DashboardExportButtonProps = {
    dashboardType: DashboardType;
    reportName: string;
    targetRef: RefObject<HTMLElement | null>;
};

const DEFAULT_OUTPUT_DIR = 'output/pdf';
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

const toPngBlob = async (canvas: HTMLCanvasElement): Promise<Blob> =>
    new Promise<Blob>((resolve, reject) => {
        canvas.toBlob((blob) => {
            if (blob) {
                resolve(blob);
                return;
            }
            reject(new Error('Failed to capture dashboard image.'));
        }, 'image/png', 1);
    });

const DashboardExportButton = ({
    dashboardType,
    reportName,
    targetRef,
}: DashboardExportButtonProps) => {
    const fallbackStem = DEFAULT_FILE_STEMS[dashboardType];
    const defaultStem = useMemo(
        () => sanitizeFileStem(reportName, fallbackStem),
        [fallbackStem, reportName],
    );

    const [isDialogOpen, setIsDialogOpen] = useState(false);
    const [outputDir, setOutputDir] = useState(DEFAULT_OUTPUT_DIR);
    const [fileName, setFileName] = useState(defaultStem);
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

    const handleExport = async () => {
        const target = targetRef.current;
        if (!target) {
            setError('Dashboard content is not available for export.');
            return;
        }

        const normalizedOutputDir = outputDir.trim();
        const normalizedFileName = sanitizeFileStem(fileName, defaultStem);
        if (!normalizedOutputDir) {
            setError('Output path is required.');
            return;
        }

        setSubmitting(true);
        resetFeedback();
        try {
            const width = Math.max(target.scrollWidth, target.clientWidth, 1);
            const height = Math.max(target.scrollHeight, target.clientHeight, 1);

            const canvas = await html2canvas(target, {
                backgroundColor: '#161a1d',
                width,
                height,
                windowWidth: width,
                windowHeight: height,
                scale: 2,
                useCORS: true,
                logging: false,
            });
            const imagePng = await toPngBlob(canvas);
            const response = await exportDashboardPdf({
                dashboardType,
                reportName,
                outputDir: normalizedOutputDir,
                fileName: normalizedFileName,
                imagePng,
            });
            setSuccessMessage(`Report saved to ${response.output_path}`);
        } catch (submissionError) {
            setError(
                submissionError instanceof Error
                    ? submissionError.message
                    : 'Failed to export dashboard.',
            );
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
                disabled={submitting}
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
                                    Capture the current dashboard layout and save it as a PDF report.
                                </p>
                            </div>
                        </header>

                        <div className="dashboard-export-form">
                            <label className="input-stack" htmlFor={`export-output-dir-${dashboardType}`}>
                                <span className="field-label">Output Path</span>
                                <input
                                    id={`export-output-dir-${dashboardType}`}
                                    className="text-input"
                                    value={outputDir}
                                    onChange={(event) => setOutputDir(event.target.value)}
                                    placeholder="output/pdf"
                                    disabled={submitting}
                                />
                            </label>
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
