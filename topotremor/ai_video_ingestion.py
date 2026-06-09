"""
AI Video Ingestion Pipeline for TopoTremor

Process AI-generated videos through the full data extraction pipeline:
1. Hand detection and bounding box
2. Dense optical-flow dot placement and tracking
3. Macro motion removal
4. Residual micro-motion feature extraction
5. 3-second window segmentation
6. Weak label attachment with QA metrics

Input: Video file + QA metrics + prompt label
Output: Training-ready dataset with weak labels and confidence scores
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Tuple, List, Dict, Optional
import json
from datetime import datetime
import logging
import mediapipe as mp
from scipy import signal, fft
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class DotTrack:
    """Single dot's 3-second trajectory and features."""
    dot_id: int
    x_trajectory: List[float]
    y_trajectory: List[float]
    frame_ids: List[int]
    
    # Macro motion (removed)
    macro_x: List[float]
    macro_y: List[float]
    
    # Residual micro-motion
    residual_x: List[float]
    residual_y: List[float]
    
    # Computed features
    residual_amplitude_px: float
    dominant_frequency_hz: Optional[float]
    band_power_3_12hz: Optional[float]
    band_power_ratio: Optional[float]
    coherence_with_neighbors: Optional[float]
    tracking_quality: float
    
    # QA flags
    valid_for_training: bool
    invalid_reason: Optional[str] = None


@dataclass
class WindowData:
    """3-second window of dense-field data with weak labels."""
    window_id: str
    video_path: str
    frame_range: Tuple[int, int]
    duration_sec: float
    fps: float
    
    # Hand ROI
    hand_bbox: Tuple[float, float, float, float]
    hand_area_px: float
    
    # Dot field
    dot_tracks: List[DotTrack]
    valid_dot_count: int
    valid_dot_ratio: float
    
    # Measurements (from residual micro-motion)
    global_dominant_frequency_hz: Optional[float]
    global_amplitude_px: float
    global_band_power: Optional[float]
    global_coherence: Optional[float]
    tracking_quality_mean: float
    
    # Weak labels from prompt/QA
    source_type: str  # AI_GENERATED_VIDEO, REAL_WEBCAM, CONTROLLED_PHYSICAL_HARNESS, etc.
    tremor_present: int  # 0 or 1 (weak label)
    label_origin: str  # WEAK_PROMPT_LABEL, OPERATOR_PROTOCOL, AI_QA_REJECTED, DEBUG_SYNTHETIC_RENDER
    label_confidence: float  # [0, 1] confidence in label
    prompt_category: Optional[str]  # no_tremor_still, weak_tremor_fingertip, hard_negative_finger_tap, etc.
    
    # QA metrics from ai_video_qa.py
    qa_passed: bool
    anatomy_valid: bool
    temporal_valid: bool
    artifact_score: float
    validity_label: str  # VALID, WEAK, INVALID, GENERATOR_ARTIFACT
    
    # Metadata
    created_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class AIVideoIngestion:
    """Ingest AI-generated videos into training dataset with weak labels."""
    
    # Optical flow and dot tracking parameters
    FEATURE_PARAMS = dict(maxCorners=500, qualityLevel=0.01, minDistance=7, blockSize=7)
    LK_PARAMS = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Feature extraction parameters
    TREMOR_FREQ_MIN = 3.0  # Hz
    TREMOR_FREQ_MAX = 12.0  # Hz
    WINDOW_DURATION_SEC = 3.0
    
    def __init__(self, fps: float = 30.0):
        """
        Initialize ingestion pipeline.
        
        Args:
            fps: Frames per second (default 30)
        """
        self.fps = fps
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def ingest_video(
        self,
        video_path: str,
        qa_metrics: Dict,
        prompt_category: str,
        source_type: str = "AI_GENERATED_VIDEO",
        tremor_present: int = 0,
        label_confidence: float = 0.5,
        output_json: Optional[str] = None
    ) -> List[WindowData]:
        """
        Process a single AI video and extract windows with weak labels.
        
        Args:
            video_path: Path to input video
            qa_metrics: Dict of QA metrics (from ai_video_qa.py)
            prompt_category: Category (no_tremor_still, weak_tremor_fingertip, etc.)
            source_type: Data source type (default AI_GENERATED_VIDEO)
            tremor_present: Weak label (0 or 1)
            label_confidence: Confidence in label [0, 1]
            output_json: Optional path to write windows as JSON
            
        Returns:
            List of WindowData objects (one per 3-second window)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Ingesting {video_path}: {frame_count} frames @ {fps:.1f} FPS")
        
        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        
        if len(frames) < 2:
            logger.warning(f"Video too short ({len(frames)} frames)")
            return []
        
        # Determine label_origin based on qa_metrics and tremor_present
        if qa_metrics.get('validity_label') == 'GENERATOR_ARTIFACT':
            label_origin = 'AI_QA_REJECTED'
            tremor_present = 0  # Artifacts are not tremor
        elif source_type == 'AI_GENERATED_VIDEO':
            label_origin = 'WEAK_PROMPT_LABEL'
        else:
            label_origin = 'OPERATOR_PROTOCOL'
        
        # Detect hands and extract windows
        windows = []
        window_duration_frames = int(fps * self.WINDOW_DURATION_SEC)
        stride = window_duration_frames  # Non-overlapping windows
        
        for start_frame in range(0, len(frames) - window_duration_frames, stride):
            end_frame = start_frame + window_duration_frames
            window_frames = frames[start_frame:end_frame]
            
            # Detect hand ROI in window
            hand_bbox = self._detect_hand_roi(window_frames)
            
            if hand_bbox is None:
                logger.debug(f"No hand detected in window {start_frame}-{end_frame}")
                continue
            
            x1, y1, x2, y2 = hand_bbox
            hand_area = (x2 - x1) * (y2 - y1)
            
            # Track dots using optical flow
            dot_tracks = self._track_dots_in_window(
                window_frames, hand_bbox, fps=fps
            )
            
            if len(dot_tracks) == 0:
                logger.debug(f"No dots tracked in window {start_frame}-{end_frame}")
                continue
            
            valid_dots = [d for d in dot_tracks if d.valid_for_training]
            valid_dot_count = len(valid_dots)
            valid_dot_ratio = valid_dot_count / len(dot_tracks) if dot_tracks else 0.0
            
            # Aggregate measurements
            global_dominant_freq = self._aggregate_dominant_frequency(valid_dots)
            global_amplitude = self._aggregate_amplitude(valid_dots)
            global_band_power = self._aggregate_band_power(valid_dots)
            global_coherence = self._aggregate_coherence(dot_tracks)
            tq_mean = np.mean([d.tracking_quality for d in dot_tracks]) if dot_tracks else 0.0
            
            # Create window
            window_id = f"{video_path.stem}_w{start_frame:06d}"
            window = WindowData(
                window_id=window_id,
                video_path=str(video_path),
                frame_range=(start_frame, end_frame),
                duration_sec=self.WINDOW_DURATION_SEC,
                fps=fps,
                hand_bbox=hand_bbox,
                hand_area_px=hand_area,
                dot_tracks=dot_tracks,
                valid_dot_count=valid_dot_count,
                valid_dot_ratio=valid_dot_ratio,
                global_dominant_frequency_hz=global_dominant_freq,
                global_amplitude_px=global_amplitude,
                global_band_power=global_band_power,
                global_coherence=global_coherence,
                tracking_quality_mean=tq_mean,
                source_type=source_type,
                tremor_present=tremor_present,
                label_origin=label_origin,
                label_confidence=label_confidence,
                prompt_category=prompt_category,
                qa_passed=qa_metrics.get('qa_passed', False),
                anatomy_valid=qa_metrics.get('anatomy_valid', False),
                temporal_valid=qa_metrics.get('temporal_valid', False),
                artifact_score=qa_metrics.get('generator_artifact_score', 0.0),
                validity_label=qa_metrics.get('validity_label', 'UNKNOWN')
            )
            
            windows.append(window)
        
        logger.info(f"Extracted {len(windows)} windows from {video_path}")
        
        # Write to JSON if requested
        if output_json:
            self._write_windows_json(windows, output_json)
        
        return windows
    
    def _detect_hand_roi(self, frames: List[np.ndarray]) -> Optional[Tuple[float, float, float, float]]:
        """Detect hand bounding box (aggregate over frames)."""
        bboxes = []
        
        for frame in frames:
            results = self.hands.process(frame)
            
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                h, w, _ = frame.shape
                
                points = np.array([[lm.x * w, lm.y * h] for lm in landmarks.landmark])
                x1, y1 = points.min(axis=0)
                x2, y2 = points.max(axis=0)
                
                bboxes.append((x1, y1, x2, y2))
        
        if not bboxes:
            return None
        
        # Return median bbox
        bboxes = np.array(bboxes)
        return tuple(np.median(bboxes, axis=0))
    
    def _track_dots_in_window(
        self,
        frames: List[np.ndarray],
        hand_bbox: Tuple[float, float, float, float],
        fps: float
    ) -> List[DotTrack]:
        """Track optical-flow dots in window with macro motion removal."""
        x1, y1, x2, y2 = hand_bbox
        hand_mask = None
        
        # Create hand mask
        h, w = frames[0].shape[:2]
        hand_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(hand_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)
        
        # Detect corners in first frame
        gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, mask=hand_mask, **self.FEATURE_PARAMS)
        
        if corners is None or len(corners) == 0:
            logger.debug("No corners detected in window")
            return []
        
        # Initialize dot IDs
        dot_tracks_dict = {i: DotTrack(
            dot_id=i,
            x_trajectory=[float(c[0, 0])],
            y_trajectory=[float(c[0, 1])],
            frame_ids=[0],
            macro_x=[],
            macro_y=[],
            residual_x=[],
            residual_y=[],
            residual_amplitude_px=0.0,
            dominant_frequency_hz=None,
            band_power_3_12hz=None,
            band_power_ratio=None,
            coherence_with_neighbors=None,
            tracking_quality=1.0,
            valid_for_training=True
        ) for i in range(len(corners))}
        
        # Track dots frame by frame
        prev_gray = gray
        
        for frame_idx in range(1, len(frames)):
            curr_frame = frames[frame_idx]
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
            
            # Get previous positions
            prev_points = np.array([
                [[dot_tracks_dict[i].x_trajectory[-1], dot_tracks_dict[i].y_trajectory[-1]]]
                for i in range(len(dot_tracks_dict))
            ], dtype=np.float32)
            
            # Optical flow (Lucas-Kanade)
            next_points, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_points, None, **self.LK_PARAMS
            )
            
            if next_points is not None and status is not None:
                for i, (pt, st, e) in enumerate(zip(next_points, status, err)):
                    if st == 1:  # Successfully tracked
                        x, y = float(pt[0, 0]), float(pt[0, 1])
                        
                        # Check if in hand ROI
                        if hand_mask[int(y) % h, int(x) % w] > 0:
                            dot_tracks_dict[i].x_trajectory.append(x)
                            dot_tracks_dict[i].y_trajectory.append(y)
                            dot_tracks_dict[i].frame_ids.append(frame_idx)
                            dot_tracks_dict[i].tracking_quality = 1.0 - float(e[0]) / 100.0
                        else:
                            dot_tracks_dict[i].valid_for_training = False
                            dot_tracks_dict[i].invalid_reason = "Out of hand ROI"
                    else:
                        dot_tracks_dict[i].valid_for_training = False
                        dot_tracks_dict[i].invalid_reason = "Tracking lost"
            
            prev_gray = curr_gray
        
        # Compute features for each dot
        for dot_id, dot in dot_tracks_dict.items():
            if len(dot.x_trajectory) < 3:
                dot.valid_for_training = False
                dot.invalid_reason = "Too few frames"
                continue
            
            # Compute macro motion (hand drift via median filter)
            traj_x = np.array(dot.x_trajectory)
            traj_y = np.array(dot.y_trajectory)
            
            # Macro motion: low-pass filter (smooth drift)
            macro_x = signal.medfilt(traj_x, kernel_size=min(5, len(traj_x) // 2 * 2 + 1))
            macro_y = signal.medfilt(traj_y, kernel_size=min(5, len(traj_y) // 2 * 2 + 1))
            
            dot.macro_x = list(macro_x)
            dot.macro_y = list(macro_y)
            
            # Residual micro-motion (high-frequency)
            residual_x = traj_x - macro_x
            residual_y = traj_y - macro_y
            
            dot.residual_x = list(residual_x)
            dot.residual_y = list(residual_y)
            
            # Amplitude
            rms_x = np.sqrt(np.mean(residual_x ** 2))
            rms_y = np.sqrt(np.mean(residual_y ** 2))
            dot.residual_amplitude_px = np.sqrt(rms_x ** 2 + rms_y ** 2)
            
            # Dominant frequency
            if len(residual_x) > 10:
                # Combine x and y residuals
                residual = np.sqrt(residual_x ** 2 + residual_y ** 2)
                
                # Welch PSD
                freqs, psd = signal.welch(
                    residual, fs=fps, nperseg=min(256, len(residual))
                )
                
                # Find dominant frequency in tremor band
                tremor_mask = (freqs >= self.TREMOR_FREQ_MIN) & (freqs <= self.TREMOR_FREQ_MAX)
                if tremor_mask.any():
                    tremor_psd = psd[tremor_mask]
                    tremor_freqs = freqs[tremor_mask]
                    dom_freq_idx = np.argmax(tremor_psd)
                    dot.dominant_frequency_hz = float(tremor_freqs[dom_freq_idx])
                    
                    # Band power
                    dot.band_power_3_12hz = float(np.sum(tremor_psd))
                    
                    # Band power ratio (tremor band / total)
                    total_psd = np.sum(psd[freqs <= 20])
                    if total_psd > 0:
                        dot.band_power_ratio = dot.band_power_3_12hz / total_psd
            
            # Check validity
            if dot.residual_amplitude_px < 0.5:
                dot.valid_for_training = False
                dot.invalid_reason = "Too small amplitude"
        
        return list(dot_tracks_dict.values())
    
    def _aggregate_dominant_frequency(self, dots: List[DotTrack]) -> Optional[float]:
        """Aggregate dominant frequency across valid dots."""
        freqs = [d.dominant_frequency_hz for d in dots if d.dominant_frequency_hz is not None]
        if freqs:
            return float(np.median(freqs))
        return None
    
    def _aggregate_amplitude(self, dots: List[DotTrack]) -> float:
        """Aggregate amplitude (RMS of residual amplitudes)."""
        if not dots:
            return 0.0
        amps = [d.residual_amplitude_px for d in dots]
        return float(np.sqrt(np.mean(np.array(amps) ** 2)))
    
    def _aggregate_band_power(self, dots: List[DotTrack]) -> Optional[float]:
        """Aggregate band power across dots."""
        powers = [d.band_power_3_12hz for d in dots if d.band_power_3_12hz is not None]
        if powers:
            return float(np.mean(powers))
        return None
    
    def _aggregate_coherence(self, dots: List[DotTrack]) -> Optional[float]:
        """
        Estimate multi-dot coherence (how synchronized the tremor is).
        
        Simple approach: measure correlation of residual motion across dots.
        """
        if len(dots) < 2:
            return None
        
        valid_dots = [d for d in dots if len(d.residual_x) > 10]
        if len(valid_dots) < 2:
            return None
        
        # Compute mean residual motion
        residuals = []
        for dot in valid_dots:
            res = np.sqrt(np.array(dot.residual_x) ** 2 + np.array(dot.residual_y) ** 2)
            residuals.append(res)
        
        # Normalize and compute pairwise correlations
        residuals = np.array(residuals)
        residuals = (residuals - residuals.mean(axis=1, keepdims=True)) / (residuals.std(axis=1, keepdims=True) + 1e-6)
        
        # Mean pairwise correlation
        correlations = []
        for i in range(len(residuals)):
            for j in range(i + 1, len(residuals)):
                corr = np.corrcoef(residuals[i], residuals[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        if correlations:
            return float(np.mean(correlations))
        
        return None
    
    def _write_windows_json(self, windows: List[WindowData], output_path: str) -> None:
        """Write windows to JSON (excluding trajectory data for size)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        windows_data = []
        for window in windows:
            w_dict = asdict(window)
            # Remove trajectory data (too large)
            w_dict['dot_tracks'] = [
                {
                    'dot_id': d.dot_id,
                    'residual_amplitude_px': d.residual_amplitude_px,
                    'dominant_frequency_hz': d.dominant_frequency_hz,
                    'band_power_3_12hz': d.band_power_3_12hz,
                    'band_power_ratio': d.band_power_ratio,
                    'coherence_with_neighbors': d.coherence_with_neighbors,
                    'tracking_quality': d.tracking_quality,
                    'valid_for_training': d.valid_for_training,
                    'invalid_reason': d.invalid_reason
                }
                for d in window.dot_tracks
            ]
            windows_data.append(w_dict)
        
        with open(output_path, 'w') as f:
            json.dump(windows_data, f, indent=2, default=str)
        
        logger.info(f"Wrote {len(windows_data)} windows to {output_path}")


def batch_ingest_videos(
    video_list_json: str,
    output_dir: str,
    ingestion_summary_json: str
) -> None:
    """
    Batch ingest multiple AI videos from manifest.
    
    video_list_json format:
    [
        {
            "video_path": "...",
            "qa_metrics": {...},
            "prompt_category": "...",
            "source_type": "AI_GENERATED_VIDEO",
            "tremor_present": 0,
            "label_confidence": 0.8
        },
        ...
    ]
    """
    ingestion = AIVideoIngestion()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load manifest
    with open(video_list_json, 'r') as f:
        videos = json.load(f)
    
    # Ingest each video
    all_windows = []
    ingestion_results = []
    
    for video_info in videos:
        try:
            windows = ingestion.ingest_video(
                video_path=video_info['video_path'],
                qa_metrics=video_info['qa_metrics'],
                prompt_category=video_info['prompt_category'],
                source_type=video_info.get('source_type', 'AI_GENERATED_VIDEO'),
                tremor_present=video_info.get('tremor_present', 0),
                label_confidence=video_info.get('label_confidence', 0.5),
                output_json=str(output_dir / f"{Path(video_info['video_path']).stem}_windows.json")
            )
            
            all_windows.extend(windows)
            ingestion_results.append({
                'video_path': video_info['video_path'],
                'windows_extracted': len(windows),
                'status': 'success'
            })
        
        except Exception as e:
            logger.error(f"Error ingesting {video_info['video_path']}: {e}")
            ingestion_results.append({
                'video_path': video_info['video_path'],
                'windows_extracted': 0,
                'status': 'error',
                'error': str(e)
            })
    
    # Write summary
    summary = {
        'total_videos': len(videos),
        'successful_ingestions': sum(1 for r in ingestion_results if r['status'] == 'success'),
        'total_windows': len(all_windows),
        'timestamp': datetime.utcnow().isoformat(),
        'video_results': ingestion_results,
        'window_distribution': {
            'by_source': _count_by_field(all_windows, 'source_type'),
            'by_tremor_label': {
                'tremor_0': sum(1 for w in all_windows if w.tremor_present == 0),
                'tremor_1': sum(1 for w in all_windows if w.tremor_present == 1)
            },
            'by_label_origin': _count_by_field(all_windows, 'label_origin'),
            'by_validity': _count_by_field(all_windows, 'validity_label'),
            'by_prompt_category': _count_by_field(all_windows, 'prompt_category')
        }
    }
    
    with open(ingestion_summary_json, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"\nIngestion Summary:")
    logger.info(f"  Total videos: {summary['total_videos']}")
    logger.info(f"  Successful: {summary['successful_ingestions']}")
    logger.info(f"  Total windows: {summary['total_windows']}")
    logger.info(f"  Summary written to {ingestion_summary_json}")


def _count_by_field(windows: List[WindowData], field_name: str) -> Dict[str, int]:
    """Count windows by field value."""
    counts = defaultdict(int)
    for w in windows:
        value = getattr(w, field_name, None)
        counts[str(value)] += 1
    return dict(counts)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ai_video_ingestion.py <video_path> <qa_metrics_json> [prompt_category]")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO)
    
    video_path = sys.argv[1]
    qa_metrics_json = sys.argv[2]
    prompt_category = sys.argv[3] if len(sys.argv) > 3 else "unknown"
    
    # Load QA metrics
    with open(qa_metrics_json, 'r') as f:
        qa_metrics = json.load(f)
    
    # Ingest
    ingestion = AIVideoIngestion()
    windows = ingestion.ingest_video(
        video_path=video_path,
        qa_metrics=qa_metrics,
        prompt_category=prompt_category,
        output_json=f"{Path(video_path).stem}_windows.json"
    )
    
    print(f"\nExtracted {len(windows)} windows")
    if windows:
        w = windows[0]
        print(f"\nFirst window:")
        print(f"  ID: {w.window_id}")
        print(f"  Frames: {w.frame_range[0]}-{w.frame_range[1]}")
        print(f"  Valid dots: {w.valid_dot_count}/{len(w.dot_tracks)}")
        print(f"  Dominant frequency: {w.global_dominant_frequency_hz:.1f} Hz" if w.global_dominant_frequency_hz else "  Dominant frequency: None")
        print(f"  Amplitude: {w.global_amplitude_px:.2f} px")
        print(f"  Tremor label: {w.tremor_present} (confidence {w.label_confidence:.2f})")
        print(f"  Validity: {w.validity_label}")
