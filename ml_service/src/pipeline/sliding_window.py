# sliding_window.py - Circular queue buffer
from collections import deque
import numpy as np


class SlidingWindowBuffer:
    """Circular buffer holding the last N vectors of a fixed width.

    Used at inference time to smooth per-frame model output (average the
    last few softmax probability vectors before argmax) rather than to
    build model input - the model itself classifies a single frame.
    """

    def __init__(self, max_size=30, vector_size=None):
        self.max_size = max_size
        self.vector_size = vector_size
        self.buffer = deque(maxlen=max_size)

    def add(self, feature_vector):
        """Add a new vector to the buffer.

        Args:
            feature_vector: 1D numpy array; must match vector_size if one was set
        """
        if self.vector_size is not None and feature_vector.shape != (self.vector_size,):
            raise ValueError(f"Expected shape ({self.vector_size},), got {feature_vector.shape}")
        self.buffer.append(feature_vector.copy())
    
    def get_window(self):
        """Get the current window as a numpy array.
        
        Returns:
            numpy array of shape (window_len, 84) where window_len <= max_size
        """
        return np.array(list(self.buffer), dtype=np.float32)
    
    def is_full(self):
        """Check if the buffer is full."""
        return len(self.buffer) >= self.max_size
    
    def size(self):
        """Return current buffer size."""
        return len(self.buffer)