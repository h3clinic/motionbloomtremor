"""Process-wide lock to serialize OpenCV VideoCapture I/O.

macOS's AVFoundation backend used by ``cv2.VideoCapture`` is not safe for
concurrent calls from multiple Python threads. When the webcam tracker thread
calls ``cap.read()`` at the same time the main thread decodes an MP4 frame via
``HybridVideoPlayer.render_frame``, the GIL can get released improperly and the
interpreter aborts with::

    Fatal Python error: PyEval_RestoreThread: the function must be called with
    the GIL held, after Python initialization and before Python finalization

To avoid this, every ``cv2.VideoCapture.read/grab/retrieve/set`` call in the
codebase should be wrapped in ``with CV_LOCK:`` so only one capture is talking
to the platform layer at a time.
"""

from __future__ import annotations

import threading

# Re-entrant so the same thread can acquire it inside nested helpers.
CV_LOCK: "threading.RLock" = threading.RLock()
