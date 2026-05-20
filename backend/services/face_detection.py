"""
Face Detection Service (SIMPLIFIED)

NOTE:
We DO NOT use YOLO / MTCNN anymore.
InsightFace already performs detection + alignment internally.
"""

import logging
import numpy as np
import cv2
from typing import Optional

logger = logging.getLogger(__name__)


def decode_image(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decode raw image bytes to OpenCV BGR image.
    """

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            logger.error("Image decode failed (None)")
            return None

        return image

    except Exception as e:
        logger.error(f"Image decoding error: {e}")
        return None


def image_bytes_from_array(face_array: np.ndarray) -> bytes:
    """
    Convert numpy BGR array to JPEG bytes.
    """

    try:
        _, buffer = cv2.imencode(".jpg", face_array)
        return buffer.tobytes()
    except Exception as e:
        logger.error(f"Image encoding failed: {e}")
        return b""