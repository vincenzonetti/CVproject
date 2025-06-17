"""
Multi-hypothesis tracker implementation.
Manages multiple possible tracks and selects the best one based on confidence.
"""

from collections import deque
import numpy as np
from config import TrackingConfig
from utils import euclidean_distance


class MultiHypothesisTracker:
    """Multi-hypothesis tracker for handling multiple possible tracks"""
    
    def __init__(self, max_hypotheses=TrackingConfig.MAX_HYPOTHESES):
        self.hypotheses = []
        self.max_hypotheses = max_hypotheses
        self.next_id = 0
        
    def add_hypothesis(self, detection, confidence):
        """
        Add new tracking hypothesis.
        
        Args:
            detection: Detection position (x, y)
            confidence: Initial confidence score
        """
        hypothesis = {
            'id': self.next_id,
            'positions': deque([detection], maxlen=10),
            'confidence': confidence,
            'age': 0,
            'last_update': 0
        }
        self.hypotheses.append(hypothesis)
        self.next_id += 1
        
    def update_hypotheses(self, frame_idx, detections=None):
        """
        Update all hypotheses.
        
        Args:
            frame_idx: Current frame index
            detections: List of current detections
        """
        for hyp in self.hypotheses:
            hyp['age'] += 1
            
            if detections:
                self._update_hypothesis_with_detections(hyp, detections, frame_idx)
            else:
                # No detections - decrease confidence
                hyp['confidence'] = max(0.0, hyp['confidence'] - TrackingConfig.CONFIDENCE_DECREMENT)
                
        # Remove low confidence and old hypotheses
        self._cleanup_hypotheses()
        
        # Limit number of hypotheses
        self._limit_hypotheses()
        
    def _update_hypothesis_with_detections(self, hypothesis, detections, frame_idx):
        """Update a single hypothesis with available detections."""
        best_match = None
        best_distance = float('inf')
        
        last_pos = hypothesis['positions'][-1]
        
        # Find best matching detection for this hypothesis
        for det in detections:
            distance = euclidean_distance(det, last_pos)
            if distance < best_distance and distance < TrackingConfig.DETECTION_DISTANCE_THRESHOLD:
                best_distance = distance
                best_match = det
        
        if best_match:
            # Update with matched detection
            hypothesis['positions'].append(best_match)
            hypothesis['confidence'] = min(1.0, hypothesis['confidence'] + TrackingConfig.CONFIDENCE_INCREMENT)
            hypothesis['last_update'] = frame_idx
        else:
            # No match found - decrease confidence
            hypothesis['confidence'] = max(0.0, hypothesis['confidence'] - TrackingConfig.CONFIDENCE_DECREMENT)
            
    def _cleanup_hypotheses(self):
        """Remove hypotheses that are too old or have low confidence."""
        self.hypotheses = [
            h for h in self.hypotheses 
            if h['confidence'] > TrackingConfig.MIN_HYPOTHESIS_CONFIDENCE 
            and h['age'] < TrackingConfig.HYPOTHESIS_MAX_AGE
        ]
        
    def _limit_hypotheses(self):
        """Limit the number of hypotheses to maximum allowed."""
        if len(self.hypotheses) > self.max_hypotheses:
            # Sort by confidence and keep the best ones
            self.hypotheses.sort(key=lambda x: x['confidence'], reverse=True)
            self.hypotheses = self.hypotheses[:self.max_hypotheses]
            
    def get_best_hypothesis(self):
        """
        Get the best hypothesis.
        
        Returns:
            tuple: Best position (x, y) or None if no good hypothesis
        """
        if not self.hypotheses:
            return None
            
        best = max(self.hypotheses, key=lambda x: x['confidence'])
        
        # Only return if confidence is above threshold
        if best['confidence'] > TrackingConfig.MIN_HYPOTHESIS_CONFIDENCE * 3:  # Higher threshold for output
            return best['positions'][-1]
            
        return None
        
    def get_all_hypotheses(self):
        """
        Get all current hypotheses.
        
        Returns:
            list: List of all hypotheses
        """
        return self.hypotheses.copy()
        
    def get_hypothesis_count(self):
        """Get the number of active hypotheses."""
        return len(self.hypotheses)
        
    def clear_hypotheses(self):
        """Clear all hypotheses."""
        self.hypotheses.clear()
        self.next_id = 0
        
    def get_trajectory(self, hypothesis_id):
        """
        Get trajectory for a specific hypothesis.
        
        Args:
            hypothesis_id: ID of the hypothesis
            
        Returns:
            list: List of positions or None if not found
        """
        for hyp in self.hypotheses:
            if hyp['id'] == hypothesis_id:
                return list(hyp['positions'])
        return None
        
    def prune_old_hypotheses(self, max_age):
        """
        Remove hypotheses older than specified age.
        
        Args:
            max_age: Maximum allowed age
        """
        self.hypotheses = [h for h in self.hypotheses if h['age'] <= max_age]