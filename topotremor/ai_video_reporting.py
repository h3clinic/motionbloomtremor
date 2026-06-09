"""
AI Video Harness Reporting Utilities

Generate automated reports from QA and ingestion results.

Reports produced:
- ai_video_qa.csv: Per-video QA metrics and validity assessment
- ai_video_ingestion_summary.json: Batch ingestion results and distributions
- generator_artifact_failures.csv: Videos that failed QA
- hard_negative_false_positive_windows.csv: Hard negatives with detected tremor
- weak_tremor_candidate_windows.csv: Weak tremor windows that passed QA
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def generate_qa_report(
    qa_results_json: str,
    output_csv: str
) -> None:
    """
    Generate per-video QA report.
    
    Args:
        qa_results_json: Path to QA results JSON (list of QA metrics dicts)
        output_csv: Path to output CSV
    """
    with open(qa_results_json, 'r') as f:
        results = json.load(f)
    
    # Flatten to CSV
    with open(output_csv, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    logger.info(f"Wrote {len(results)} QA results to {output_csv}")
    
    # Print summary
    passed = sum(1 for r in results if r.get('qa_passed', False))
    artifacts = sum(1 for r in results if r.get('validity_label') == 'GENERATOR_ARTIFACT')
    print(f"\nQA Report Summary:")
    print(f"  Total videos: {len(results)}")
    print(f"  QA passed: {passed} ({100*passed/len(results):.1f}%)")
    print(f"  Generator artifacts: {artifacts} ({100*artifacts/len(results):.1f}%)")


def generate_ingestion_summary(
    ingestion_results_dir: str,
    output_json: str
) -> None:
    """
    Generate batch ingestion summary report.
    
    Args:
        ingestion_results_dir: Directory containing window JSONs from ingestion
        output_json: Path to output summary JSON
    """
    ingestion_dir = Path(ingestion_results_dir)
    window_files = list(ingestion_dir.glob("*_windows.json"))
    
    all_windows = []
    for window_file in window_files:
        with open(window_file, 'r') as f:
            windows = json.load(f)
            all_windows.extend(windows)
    
    # Generate statistics
    summary = {
        'total_windows': len(all_windows),
        'timestamp': datetime.utcnow().isoformat(),
        
        'distribution_by_source_type': _count_field(all_windows, 'source_type'),
        'distribution_by_label_origin': _count_field(all_windows, 'label_origin'),
        'distribution_by_validity': _count_field(all_windows, 'validity_label'),
        'distribution_by_prompt_category': _count_field(all_windows, 'prompt_category'),
        'distribution_by_tremor_label': {
            'tremor_0': sum(1 for w in all_windows if w.get('tremor_present') == 0),
            'tremor_1': sum(1 for w in all_windows if w.get('tremor_present') == 1),
        },
        
        'qa_metrics': {
            'qa_passed': sum(1 for w in all_windows if w.get('qa_passed', False)),
            'anatomy_valid': sum(1 for w in all_windows if w.get('anatomy_valid', False)),
            'temporal_valid': sum(1 for w in all_windows if w.get('temporal_valid', False)),
            'avg_valid_dot_ratio': _avg_field(all_windows, 'valid_dot_ratio'),
            'avg_artifact_score': _avg_field(all_windows, 'artifact_score'),
            'avg_label_confidence': _avg_field(all_windows, 'label_confidence'),
        },
        
        'frequency_statistics': {
            'windows_with_frequency': sum(
                1 for w in all_windows 
                if w.get('global_dominant_frequency_hz') is not None
            ),
            'median_frequency_hz': _median_field(all_windows, 'global_dominant_frequency_hz'),
            'median_amplitude_px': _median_field(all_windows, 'global_amplitude_px'),
        },
    }
    
    # Write summary
    with open(output_json, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Ingestion summary written to {output_json}")
    
    # Print summary
    print(f"\nIngestion Summary:")
    print(f"  Total windows: {summary['total_windows']}")
    print(f"  QA passed: {summary['qa_metrics']['qa_passed']} ({100*summary['qa_metrics']['qa_passed']/len(all_windows):.1f}%)" if all_windows else "  (no windows)")
    print(f"  By source type: {summary['distribution_by_source_type']}")


def generate_artifact_failures_report(
    qa_results: List[Dict],
    output_csv: str
) -> None:
    """
    Generate report of videos that failed QA (artifacts).
    
    Args:
        qa_results: List of QA metrics dicts
        output_csv: Path to output CSV
    """
    artifacts = [
        r for r in qa_results
        if r.get('validity_label') == 'GENERATOR_ARTIFACT'
    ]
    
    # Select key fields
    keys_to_keep = [
        'video_path',
        'frame_count',
        'fps',
        'visible_hand_ratio',
        'finger_count_estimate',
        'hand_area_cv',
        'edge_consistency_score',
        'optical_flow_chaos_score',
        'temporal_texture_drift_score',
        'generator_artifact_reason',
        'artifact_score',
        'validity_label',
    ]
    
    artifacts_filtered = [
        {k: r.get(k) for k in keys_to_keep}
        for r in artifacts
    ]
    
    # Write CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys_to_keep)
        writer.writeheader()
        writer.writerows(artifacts_filtered)
    
    logger.info(f"Wrote {len(artifacts)} artifact failures to {output_csv}")
    print(f"  Artifact failures: {len(artifacts)}")


def generate_hard_negative_false_positives_report(
    windows: List[Dict],
    output_csv: str
) -> None:
    """
    Generate report of hard negative windows with detected tremor.
    
    Hard negatives should have tremor_present=0, but may show residual oscillation
    (e.g., tapping motion). These are useful for understanding false positive rates.
    
    Args:
        windows: List of ingestion window dicts
        output_csv: Path to output CSV
    """
    # Filter: hard negatives (prompt_category contains 'hard_negative')
    # that show measured frequency/amplitude
    hard_neg_windows = [
        w for w in windows
        if 'hard_negative' in (w.get('prompt_category') or '')
        and w.get('tremor_present') == 0
        and w.get('global_dominant_frequency_hz') is not None
    ]
    
    # Select key fields
    keys_to_keep = [
        'window_id',
        'video_path',
        'frame_range',
        'prompt_category',
        'tremor_present',
        'global_dominant_frequency_hz',
        'global_amplitude_px',
        'global_coherence',
        'valid_dot_count',
        'tracking_quality_mean',
        'label_confidence',
        'validity_label',
    ]
    
    windows_filtered = [
        {k: w.get(k) for k in keys_to_keep}
        for w in hard_neg_windows
    ]
    
    # Write CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys_to_keep)
        writer.writeheader()
        writer.writerows(windows_filtered)
    
    logger.info(f"Wrote {len(hard_neg_windows)} hard negative false positives to {output_csv}")
    print(f"  Hard negative windows with measured oscillation: {len(hard_neg_windows)}")


def generate_weak_tremor_candidates_report(
    windows: List[Dict],
    output_csv: str
) -> None:
    """
    Generate report of weak tremor windows that passed QA.
    
    These are useful for inspection and label confidence calibration.
    
    Args:
        windows: List of ingestion window dicts
        output_csv: Path to output CSV
    """
    # Filter: weak tremor category, tremor_present=1, QA passed
    weak_tremor_windows = [
        w for w in windows
        if 'weak_tremor' in (w.get('prompt_category') or '')
        and w.get('tremor_present') == 1
        and w.get('qa_passed', False)
    ]
    
    # Select key fields
    keys_to_keep = [
        'window_id',
        'video_path',
        'frame_range',
        'prompt_category',
        'tremor_present',
        'global_dominant_frequency_hz',
        'global_amplitude_px',
        'global_band_power',
        'global_coherence',
        'valid_dot_count',
        'valid_dot_ratio',
        'tracking_quality_mean',
        'label_confidence',
        'validity_label',
    ]
    
    windows_filtered = [
        {k: w.get(k) for k in keys_to_keep}
        for w in weak_tremor_windows
    ]
    
    # Write CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys_to_keep)
        writer.writeheader()
        writer.writerows(windows_filtered)
    
    logger.info(f"Wrote {len(weak_tremor_windows)} weak tremor candidates to {output_csv}")
    print(f"  Weak tremor candidate windows: {len(weak_tremor_windows)}")


def _count_field(items: List[Dict], field: str) -> Dict[str, int]:
    """Count occurrences of each value in a field."""
    counts = defaultdict(int)
    for item in items:
        value = item.get(field)
        counts[str(value)] += 1
    return dict(counts)


def _avg_field(items: List[Dict], field: str) -> Optional[float]:
    """Compute average of a numeric field."""
    values = [item.get(field) for item in items if isinstance(item.get(field), (int, float))]
    return sum(values) / len(values) if values else None


def _median_field(items: List[Dict], field: str) -> Optional[float]:
    """Compute median of a numeric field."""
    import statistics
    values = [item.get(field) for item in items if isinstance(item.get(field), (int, float))]
    return statistics.median(values) if values else None


def batch_generate_all_reports(
    qa_results_json: str,
    ingestion_results_dir: str,
    output_dir: str = 'reports'
) -> None:
    """
    Generate all reports from QA and ingestion results.
    
    Args:
        qa_results_json: Path to QA results JSON
        ingestion_results_dir: Directory containing ingestion window JSONs
        output_dir: Directory to write reports
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load QA results
    with open(qa_results_json, 'r') as f:
        qa_results = json.load(f)
    
    # Load ingestion windows
    all_windows = []
    ingestion_dir = Path(ingestion_results_dir)
    for window_file in ingestion_dir.glob("*_windows.json"):
        with open(window_file, 'r') as f:
            windows = json.load(f)
            all_windows.extend(windows)
    
    # Generate reports
    print(f"\nGenerating AI Video Harness Reports:")
    print(f"=" * 60)
    
    # 1. QA report
    qa_csv = output_dir / 'ai_video_qa.csv'
    generate_qa_report(qa_results_json, str(qa_csv))
    
    # 2. Ingestion summary
    summary_json = output_dir / 'ai_video_ingestion_summary.json'
    generate_ingestion_summary(ingestion_results_dir, str(summary_json))
    
    # 3. Artifact failures
    artifact_csv = output_dir / 'generator_artifact_failures.csv'
    generate_artifact_failures_report(qa_results, str(artifact_csv))
    
    # 4. Hard negative false positives
    fp_csv = output_dir / 'hard_negative_false_positive_windows.csv'
    generate_hard_negative_false_positives_report(all_windows, str(fp_csv))
    
    # 5. Weak tremor candidates
    weak_csv = output_dir / 'weak_tremor_candidate_windows.csv'
    generate_weak_tremor_candidates_report(all_windows, str(weak_csv))
    
    print(f"=" * 60)
    print(f"All reports written to {output_dir}")


if __name__ == '__main__':
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 3:
        print("Usage: python ai_video_reporting.py <qa_results.json> <ingestion_results_dir> [output_dir]")
        sys.exit(1)
    
    qa_json = sys.argv[1]
    ingestion_dir = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else 'reports'
    
    batch_generate_all_reports(qa_json, ingestion_dir, output_dir)
