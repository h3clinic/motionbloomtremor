"""
AI Video QA Module for TopoTremor

Comprehensive quality-assurance pipeline for AI-generated hand videos.
Detects anatomical anomalies, temporal inconsistencies, and camera/scene issues.

Rejects or flags videos as GENERATOR_ARTIFACT if they fail anatomy, temporal
consistency, or camera/motion gates.

Input: Video file path
Output: QA metrics dict with validity assessment and artifact scores
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Tuple, List, Dict, Optional
import mediapipe as mp
from scipy import signal
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class AIVideoQAMetrics:
    """QA metrics for a single AI-generated video."""
    
    # Basic metadata
    video_path: str
    frame_count: int
    fps: float
    duration_sec: float
    resolution: Tuple[int, int]
    
    # Hand presence and stability
    visible_hand_ratio: float  # [0, 1] frames with detectable hand
    hand_area_stability: float  # [0, 1] inverse of normalized std of hand bbox area
    hand_lost_ratio: float  # [0, 1] frames where hand disappears
    
    # Anatomy validation
    finger_count_estimate: Optional[int]  # Estimated 5 for valid, None if unreliable
    finger_count_variability: float  # [0, 1] how much finger count changes per frame
    multiple_hand_flag: bool  # True if >1 hand detected in same frame
    hand_area_cv: float  # Hand area coefficient of variation (std/mean)
    
    # Temporal consistency
    edge_consistency_score: float  # [0, 1] higher = more stable edges
    optical_flow_chaos_score: float  # [0, 1] normalized optical flow variance
    temporal_texture_drift_score: float  # [0, 1] how much texture changes frame-to-frame
    
    # Topology stability
    hand_identity_changes: int  # Count of frames where hand topology changes sharply
    finger_edge_crawl_detected: bool  # True if finger edges move unnaturally
    skin_texture_stability: float  # [0, 1] inverse of normalized texture variance
    
    # Camera and scene
    camera_motion_score: float  # [0, 1] estimated background motion
    background_optical_flow_variance: float  # Variance of flow in non-hand regions
    hand_frame_departure_ratio: float  # How often hand approaches frame edge
    
    # Overall QA decision
    anatomy_valid: bool
    temporal_valid: bool
    camera_scene_valid: bool
    generator_artifact_score: float  # [0, 1] confidence that this is a failure
    generator_artifact_reason: Optional[str]  # Why it failed (if artifact_score > 0.5)
    
    # Recommendation
    validity_label: str  # VALID, WEAK, INVALID, or GENERATOR_ARTIFACT
    qa_passed: bool


class AIVideoQA:
    """Quality assurance checker for AI-generated hand videos."""
    
    # QA thresholds
    MIN_VISIBLE_HAND_RATIO = 0.6  # At least 60% frames have hand
    MAX_HAND_LOST_RATIO = 0.4  # Allow up to 40% hand lost
    MAX_HAND_AREA_CV = 0.5  # Coefficient of variation in hand area
    MIN_EDGE_CONSISTENCY = 0.4  # Minimum edge stability score
    MAX_OPTICAL_FLOW_CHAOS = 0.7  # Maximum acceptable flow variance
    MAX_TEXTURE_DRIFT = 0.6  # Maximum acceptable texture change
    MAX_FINGER_COUNT_VARIABILITY = 0.3  # Finger count should be mostly stable
    MAX_IDENTITY_CHANGES = 3  # Allow a few topology shifts
    MAX_CAMERA_MOTION = 0.5  # Moderate camera motion acceptable
    MAX_HAND_DEPARTURE_RATIO = 0.2  # Hand shouldn't leave frame often
    
    def __init__(self):
        """Initialize MediaPipe Hands detector."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def analyze_video(self, video_path: str) -> AIVideoQAMetrics:
        """
        Run full QA pipeline on an AI-generated video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            AIVideoQAMetrics with all QA measurements and validity decision
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = frame_count / fps if fps > 0 else 0
        
        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames read from {video_path}")
        
        logger.info(f"Analyzing {len(frames)} frames from {video_path}")
        
        # Run detections
        hand_detections = self._detect_hands(frames)
        
        # Compute metrics
        visible_hand_ratio = len([d for d in hand_detections if d is not None]) / len(frames)
        hand_lost_ratio = 1.0 - visible_hand_ratio
        
        hand_areas = [d['area'] for d in hand_detections if d is not None]
        if len(hand_areas) > 1:
            hand_area_mean = np.mean(hand_areas)
            hand_area_std = np.std(hand_areas)
            hand_area_cv = hand_area_std / hand_area_mean if hand_area_mean > 0 else 1.0
            hand_area_stability = 1.0 / (1.0 + hand_area_cv)
        else:
            hand_area_cv = np.inf
            hand_area_stability = 0.0
        
        # Finger count and topology
        finger_counts = [d['finger_count'] for d in hand_detections if d is not None]
        if len(finger_counts) > 0:
            finger_count_estimate = max(set(finger_counts), key=finger_counts.count)  # mode
            finger_count_variability = 1.0 - (finger_counts.count(finger_count_estimate) / len(finger_counts))
        else:
            finger_count_estimate = None
            finger_count_variability = 1.0
        
        multiple_hands_frames = sum(1 for d in hand_detections if d and d['hand_count'] > 1)
        multiple_hand_flag = multiple_hands_frames > len(frames) * 0.1
        
        hand_identity_changes = self._count_topology_changes(hand_detections)
        finger_edge_crawl = self._detect_edge_crawl(hand_detections)
        
        # Temporal stability
        edge_consistency = self._compute_edge_consistency(hand_detections)
        texture_drift = self._compute_texture_drift(frames, hand_detections)
        skin_texture_stability = 1.0 - min(texture_drift, 1.0)
        
        # Optical flow
        optical_flow_chaos = self._compute_optical_flow_chaos(frames, hand_detections)
        
        # Camera and scene
        camera_motion, bg_flow_var = self._compute_camera_motion(frames, hand_detections)
        hand_departure = self._compute_hand_frame_departure(hand_detections, w, h)
        
        # Anatomy validation
        anatomy_valid = (
            visible_hand_ratio >= self.MIN_VISIBLE_HAND_RATIO and
            hand_lost_ratio <= self.MAX_HAND_LOST_RATIO and
            (finger_count_estimate == 5 or finger_count_estimate is None) and
            finger_count_variability <= self.MAX_FINGER_COUNT_VARIABILITY and
            not multiple_hand_flag and
            hand_area_cv <= self.MAX_HAND_AREA_CV and
            hand_identity_changes <= self.MAX_IDENTITY_CHANGES and
            not finger_edge_crawl
        )
        
        # Temporal validation
        temporal_valid = (
            edge_consistency >= self.MIN_EDGE_CONSISTENCY and
            optical_flow_chaos <= self.MAX_OPTICAL_FLOW_CHAOS and
            texture_drift <= self.MAX_TEXTURE_DRIFT
        )
        
        # Camera/scene validation
        camera_scene_valid = (
            camera_motion <= self.MAX_CAMERA_MOTION and
            hand_departure <= self.MAX_HAND_DEPARTURE_RATIO
        )
        
        # Compute artifact score (confidence that this is a generation failure)
        artifact_score = self._compute_artifact_score(
            anatomy_valid, temporal_valid, camera_scene_valid,
            visible_hand_ratio, finger_count_variability, hand_area_cv,
            edge_consistency, optical_flow_chaos, texture_drift
        )
        
        # Assign validity label and artifact reason
        artifact_reason = None
        if artifact_score >= 0.7:
            validity_label = "GENERATOR_ARTIFACT"
            if not anatomy_valid:
                artifact_reason = "Anatomy failure (finger count, merged digits, or topology instability)"
            elif not temporal_valid:
                artifact_reason = "Temporal instability (edge crawl, texture drift, or flow chaos)"
            elif not camera_scene_valid:
                artifact_reason = "Camera/scene issue (excessive motion or hand departure)"
        elif visible_hand_ratio < self.MIN_VISIBLE_HAND_RATIO:
            validity_label = "INVALID"
        elif not all([anatomy_valid, temporal_valid, camera_scene_valid]):
            validity_label = "WEAK"
        else:
            validity_label = "VALID"
        
        qa_passed = validity_label in ("VALID", "WEAK")
        
        metrics = AIVideoQAMetrics(
            video_path=str(video_path),
            frame_count=len(frames),
            fps=fps,
            duration_sec=duration_sec,
            resolution=(w, h),
            visible_hand_ratio=visible_hand_ratio,
            hand_area_stability=hand_area_stability,
            hand_lost_ratio=hand_lost_ratio,
            finger_count_estimate=finger_count_estimate,
            finger_count_variability=finger_count_variability,
            multiple_hand_flag=multiple_hand_flag,
            hand_area_cv=hand_area_cv,
            edge_consistency_score=edge_consistency,
            optical_flow_chaos_score=optical_flow_chaos,
            temporal_texture_drift_score=texture_drift,
            hand_identity_changes=hand_identity_changes,
            finger_edge_crawl_detected=finger_edge_crawl,
            skin_texture_stability=skin_texture_stability,
            camera_motion_score=camera_motion,
            background_optical_flow_variance=bg_flow_var,
            hand_frame_departure_ratio=hand_departure,
            anatomy_valid=anatomy_valid,
            temporal_valid=temporal_valid,
            camera_scene_valid=camera_scene_valid,
            generator_artifact_score=artifact_score,
            generator_artifact_reason=artifact_reason,
            validity_label=validity_label,
            qa_passed=qa_passed
        )
        
        return metrics
    
    def _detect_hands(self, frames: List[np.ndarray]) -> List[Optional[Dict]]:
        """Detect hand landmarks in all frames."""
        detections = []
        prev_landmarks = None
        
        for frame_idx, frame in enumerate(frames):
            results = self.hands.process(frame)
            
            if results.multi_hand_landmarks and results.multi_handedness:
                hand_count = len(results.multi_hand_landmarks)
                # Use first hand only
                landmarks = results.multi_hand_landmarks[0]
                
                # Extract landmarks as 2D points
                h, w, _ = frame.shape
                points = np.array([[lm.x * w, lm.y * h] for lm in landmarks.landmark])
                
                # Compute hand bounding box and area
                min_x, min_y = points.min(axis=0)
                max_x, max_y = points.max(axis=0)
                area = (max_x - min_x) * (max_y - min_y)
                
                # Estimate finger count (very rough: count peaks in convex hull)
                hull = cv2.convexHull(points.astype(np.float32))
                approx = cv2.approxPolyDP(hull, 0.01 * cv2.arcLength(hull, True), True)
                finger_count = min(len(approx), 10)  # Cap at 10 to avoid noise
                
                detections.append({
                    'frame_idx': frame_idx,
                    'landmarks': points,
                    'bbox': (min_x, min_y, max_x, max_y),
                    'area': area,
                    'finger_count': finger_count,
                    'hand_count': hand_count,
                    'convex_hull': hull
                })
                prev_landmarks = points
            else:
                detections.append(None)
        
        return detections
    
    def _count_topology_changes(self, detections: List[Optional[Dict]]) -> int:
        """Count sharp topology changes (convex hull shape changes)."""
        changes = 0
        prev_hull = None
        
        for det in detections:
            if det is not None and 'convex_hull' in det:
                curr_hull = det['convex_hull']
                
                if prev_hull is not None:
                    # Compare hull sizes (rough topology check)
                    prev_area = cv2.contourArea(prev_hull)
                    curr_area = cv2.contourArea(curr_hull)
                    
                    if prev_area > 0 and abs(curr_area - prev_area) / prev_area > 0.4:
                        changes += 1
                
                prev_hull = curr_hull
        
        return changes
    
    def _detect_edge_crawl(self, detections: List[Optional[Dict]]) -> bool:
        """Detect unnatural finger edge movement (edge crawl)."""
        # Edge crawl: fingertip positions move by large amounts frame-to-frame
        # without corresponding hand motion
        
        prev_landmarks = None
        crawl_events = 0
        
        for det in detections:
            if det is not None and 'landmarks' in det:
                landmarks = det['landmarks']
                
                if prev_landmarks is not None:
                    # Compute fingertip motion (points 4, 8, 12, 16, 20 are fingertips)
                    fingertip_indices = [4, 8, 12, 16, 20]
                    fingertip_distances = []
                    
                    for idx in fingertip_indices:
                        if idx < len(landmarks) and idx < len(prev_landmarks):
                            dist = np.linalg.norm(landmarks[idx] - prev_landmarks[idx])
                            fingertip_distances.append(dist)
                    
                    if fingertip_distances:
                        avg_fingertip_motion = np.mean(fingertip_distances)
                        
                        # Compute overall hand motion (wrist point 0)
                        wrist_motion = np.linalg.norm(landmarks[0] - prev_landmarks[0])
                        
                        # Edge crawl: fingertips move a lot but wrist doesn't
                        # (unnatural for actual hand motion)
                        if wrist_motion < 5 and avg_fingertip_motion > 15:
                            crawl_events += 1
                
                prev_landmarks = landmarks
        
        # Flag edge crawl if it happens in > 20% of frames
        total_frames = len([d for d in detections if d is not None])
        return crawl_events > 0.2 * total_frames if total_frames > 0 else False
    
    def _compute_edge_consistency(self, detections: List[Optional[Dict]]) -> float:
        """Compute edge stability across frames (inverse of edge crawl rate)."""
        if len(detections) < 2:
            return 1.0
        
        edge_movements = []
        prev_bbox = None
        
        for det in detections:
            if det is not None:
                bbox = det['bbox']
                
                if prev_bbox is not None:
                    # Compute change in bbox corners
                    corner_movement = (
                        abs(bbox[0] - prev_bbox[0]) +
                        abs(bbox[1] - prev_bbox[1]) +
                        abs(bbox[2] - prev_bbox[2]) +
                        abs(bbox[3] - prev_bbox[3])
                    )
                    edge_movements.append(corner_movement)
                
                prev_bbox = bbox
        
        if edge_movements:
            # Normalize by hand size
            avg_movement = np.mean(edge_movements)
            # Higher movement = lower consistency
            consistency = 1.0 / (1.0 + avg_movement / 100.0)
            return min(consistency, 1.0)
        
        return 1.0
    
    def _compute_texture_drift(self, frames: List[np.ndarray], detections: List[Optional[Dict]]) -> float:
        """Compute skin texture change frame-to-frame (temporal texture drift)."""
        if len(frames) < 2:
            return 0.0
        
        texture_changes = []
        prev_texture = None
        
        for frame_idx, det in enumerate(detections):
            if det is not None and frame_idx < len(frames):
                frame = frames[frame_idx]
                bbox = det['bbox']
                
                # Extract hand region
                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                
                if x2 > x1 and y2 > y1:
                    hand_region = frame[y1:y2, x1:x2]
                    
                    # Compute texture descriptor (simple: mean intensity per channel)
                    texture = np.mean(hand_region, axis=(0, 1))
                    
                    if prev_texture is not None:
                        change = np.linalg.norm(texture - prev_texture) / 255.0
                        texture_changes.append(change)
                    
                    prev_texture = texture
        
        if texture_changes:
            return min(np.mean(texture_changes), 1.0)
        
        return 0.0
    
    def _compute_optical_flow_chaos(self, frames: List[np.ndarray], detections: List[Optional[Dict]]) -> float:
        """Compute optical flow variance (chaos score)."""
        if len(frames) < 2:
            return 0.0
        
        flow_variances = []
        
        for i in range(len(frames) - 1):
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
            
            # Compute optical flow
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Compute flow magnitude variance
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_var = np.var(mag)
            flow_variances.append(flow_var)
        
        if flow_variances:
            # Normalize by frame size
            avg_var = np.mean(flow_variances)
            # Chaos: higher variance = higher score
            chaos = avg_var / 1000.0  # Empirical normalization
            return min(chaos, 1.0)
        
        return 0.0
    
    def _compute_camera_motion(self, frames: List[np.ndarray], detections: List[Optional[Dict]]) -> Tuple[float, float]:
        """Estimate camera motion via background optical flow."""
        if len(frames) < 2:
            return 0.0, 0.0
        
        bg_flow_vars = []
        
        for i in range(min(len(frames) - 1, 10)):  # Sample first 10 frame pairs
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
            
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Create hand mask from detection
            h, w = gray1.shape
            hand_mask = np.zeros((h, w), dtype=np.uint8)
            
            if i < len(detections) and detections[i] is not None:
                bbox = detections[i]['bbox']
                x1, y1, x2, y2 = [max(0, min(int(v), (w if j % 2 == 0 else h) - 1)) for j, v in enumerate(bbox)]
                cv2.rectangle(hand_mask, (x1, y1), (x2, y2), 255, -1)
            
            # Compute flow variance in background (inverse mask)
            bg_mask = cv2.bitwise_not(hand_mask)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            bg_flow = mag[bg_mask > 0]
            if len(bg_flow) > 0:
                bg_flow_vars.append(np.var(bg_flow))
        
        if bg_flow_vars:
            avg_bg_var = np.mean(bg_flow_vars)
            camera_motion = avg_bg_var / 500.0  # Empirical normalization
            return min(camera_motion, 1.0), avg_bg_var
        
        return 0.0, 0.0
    
    def _compute_hand_frame_departure(self, detections: List[Optional[Dict]], w: int, h: int) -> float:
        """Compute how often hand approaches or leaves frame edges."""
        if not detections:
            return 0.0
        
        departure_frames = 0
        margin = 0.1 * min(w, h)  # 10% of min dimension
        
        for det in detections:
            if det is not None:
                x1, y1, x2, y2 = det['bbox']
                
                # Check if close to edge
                if x1 < margin or x2 > w - margin or y1 < margin or y2 > h - margin:
                    departure_frames += 1
        
        total_frames = len([d for d in detections if d is not None])
        return departure_frames / total_frames if total_frames > 0 else 0.0
    
    def _compute_artifact_score(
        self,
        anatomy_valid: bool,
        temporal_valid: bool,
        camera_scene_valid: bool,
        visible_hand_ratio: float,
        finger_count_variability: float,
        hand_area_cv: float,
        edge_consistency: float,
        optical_flow_chaos: float,
        texture_drift: float
    ) -> float:
        """
        Compute artifact score (confidence that this is a generation failure).
        
        Score combines multiple failure signals:
        - Anatomy failures are strong signals (0.3–0.5 each)
        - Temporal failures are moderate signals (0.2–0.3 each)
        - Camera/scene issues are weak signals (0.1–0.2 each)
        """
        score = 0.0
        
        # Anatomy signals
        if not anatomy_valid:
            score += 0.4
        
        if visible_hand_ratio < 0.5:
            score += 0.3
        
        if finger_count_variability > 0.5:
            score += 0.25
        
        if hand_area_cv > 0.7:
            score += 0.2
        
        # Temporal signals
        if not temporal_valid:
            score += 0.3
        
        if edge_consistency < 0.3:
            score += 0.2
        
        if optical_flow_chaos > 0.8:
            score += 0.2
        
        if texture_drift > 0.7:
            score += 0.15
        
        # Camera/scene signals
        if not camera_scene_valid:
            score += 0.15
        
        return min(score, 1.0)


def batch_qa_videos(video_dir: str, output_csv: str) -> None:
    """
    Run QA on all videos in a directory, output results to CSV.
    
    Args:
        video_dir: Directory containing .mp4 files
        output_csv: Path to output CSV file
    """
    qa = AIVideoQA()
    video_dir = Path(video_dir)
    
    results = []
    for video_file in sorted(video_dir.glob("**/*.mp4")):
        logger.info(f"QA checking {video_file}")
        
        try:
            metrics = qa.analyze_video(str(video_file))
            results.append(asdict(metrics))
        except Exception as e:
            logger.error(f"Error analyzing {video_file}: {e}")
    
    # Write to CSV
    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        logger.info(f"Wrote {len(results)} QA results to {output_csv}")
        
        # Print summary
        qa_passed = df['qa_passed'].sum()
        artifacts = (df['validity_label'] == 'GENERATOR_ARTIFACT').sum()
        logger.info(f"\nQA Summary:")
        logger.info(f"  Total videos: {len(df)}")
        logger.info(f"  QA Passed: {qa_passed} ({100*qa_passed/len(df):.1f}%)")
        logger.info(f"  Generator Artifacts: {artifacts} ({100*artifacts/len(df):.1f}%)")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ai_video_qa.py <video_file_or_dir> [output_csv]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "ai_video_qa_results.csv"
    
    logging.basicConfig(level=logging.INFO)
    
    if Path(input_path).is_dir():
        batch_qa_videos(input_path, output_csv)
    else:
        qa = AIVideoQA()
        metrics = qa.analyze_video(input_path)
        
        print("\n=== AI Video QA Results ===")
        print(f"Video: {metrics.video_path}")
        print(f"Frames: {metrics.frame_count}, FPS: {metrics.fps:.1f}, Duration: {metrics.duration_sec:.1f}s")
        print(f"\nHand Presence:")
        print(f"  Visible: {100*metrics.visible_hand_ratio:.1f}%")
        print(f"  Lost: {100*metrics.hand_lost_ratio:.1f}%")
        print(f"\nAnatomy:")
        print(f"  Finger Count: {metrics.finger_count_estimate}")
        print(f"  Finger Count Variability: {metrics.finger_count_variability:.3f}")
        print(f"  Hand Area CV: {metrics.hand_area_cv:.3f}")
        print(f"  Multiple Hands: {metrics.multiple_hand_flag}")
        print(f"  Identity Changes: {metrics.hand_identity_changes}")
        print(f"  Edge Crawl: {metrics.finger_edge_crawl_detected}")
        print(f"\nTemporal Consistency:")
        print(f"  Edge Consistency: {metrics.edge_consistency_score:.3f}")
        print(f"  Optical Flow Chaos: {metrics.optical_flow_chaos_score:.3f}")
        print(f"  Texture Drift: {metrics.temporal_texture_drift_score:.3f}")
        print(f"\nCamera/Scene:")
        print(f"  Camera Motion: {metrics.camera_motion_score:.3f}")
        print(f"  Hand Departure: {metrics.hand_frame_departure_ratio:.3f}")
        print(f"\nQA Validation:")
        print(f"  Anatomy Valid: {metrics.anatomy_valid}")
        print(f"  Temporal Valid: {metrics.temporal_valid}")
        print(f"  Camera/Scene Valid: {metrics.camera_scene_valid}")
        print(f"  QA Passed: {metrics.qa_passed}")
        print(f"\nArtifact Assessment:")
        print(f"  Artifact Score: {metrics.generator_artifact_score:.3f}")
        print(f"  Validity Label: {metrics.validity_label}")
        if metrics.generator_artifact_reason:
            print(f"  Failure Reason: {metrics.generator_artifact_reason}")
