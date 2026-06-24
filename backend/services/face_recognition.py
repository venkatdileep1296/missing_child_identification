"""
Face Recognition / Embedding Service
Uses InsightFace (ONNX Runtime) with the 'buffalo_l' model (SCRFD + ArcFace)
to generate high-accuracy 512-dim face embeddings.
"""

import logging
import numpy as np
import cv2
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)

# Lazy-loaded InsightFace model
face_app = None


def get_face_app():
    global face_app

    if face_app is None:
        logger.info("Loading InsightFace model...")
        face_app = FaceAnalysis(name="buffalo_l")
        face_app.prepare(ctx_id=-1)  # CPU
        logger.info("InsightFace loaded successfully.")

    return face_app


def preprocess_image(image: np.ndarray, max_size=640) -> np.ndarray:
    """Resize image while maintaining aspect ratio."""

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    h, w = image.shape[:2]

    if max(h, w) > max_size:
        scale = max_size / float(max(h, w))
        new_w = int(w * scale)
        new_h = int(h * scale)

        image = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )

        logger.info(
            f"Resized image from {w}x{h} to {new_w}x{new_h}"
        )

    return image


def extract_embeddings(image: np.ndarray) -> list:
    """
    Returns a list containing the embedding
    for the largest detected face.
    """

    try:
        app = get_face_app()

        image = preprocess_image(image, max_size=640)

        faces = app.get(image)

    except Exception as e:
        logger.error(f"InsightFace error: {e}")
        return []

    if len(faces) == 0:
        logger.warning("No faces detected.")
        return []

    valid_faces = []

    for face in faces:
        bbox = face.bbox

        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        logger.info(
            f"Detected face size: {width:.1f}x{height:.1f}"
        )

        if width >= 80 and height >= 80:
            area = width * height
            valid_faces.append((area, face))

    if not valid_faces:
        logger.warning(
            "No valid faces found after filtering."
        )
        return []

    valid_faces.sort(
        key=lambda x: x[0],
        reverse=True
    )

    largest_face = valid_faces[0][1]

    emb = largest_face.embedding

    if emb is None or len(emb) != 512:
        logger.error(
            "Failed to extract valid 512-d embedding."
        )
        return []

    norm = np.linalg.norm(emb)

    if norm == 0:
        logger.error(
            "Embedding normalization failed."
        )
        return []

    emb = emb / norm

    return [emb.tolist()]