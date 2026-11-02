"""Paquete modular para clasificación de imágenes hiperespectrales mediante distancia euclideana."""

from .data_loader import (
    CLASS_NAMES,
    CLASS_NAMES_ES,
    load_hsi_dataset,
    get_class_distribution,
    extract_coordinates_by_class,
    extract_spectra_by_class,
    train_test_split_by_class,
)
from .classifier import (
    compute_class_centroids,
    dist_euclideana,
    dist_clases,
    clasificar_pixel,
    classify_image_iterative,
    classify_image_vectorized,
    build_test_classification_map,
    evaluate_predictions,
)

__all__ = [
    "CLASS_NAMES",
    "CLASS_NAMES_ES",
    "load_hsi_dataset",
    "get_class_distribution",
    "extract_coordinates_by_class",
    "extract_spectra_by_class",
    "train_test_split_by_class",
    "compute_class_centroids",
    "dist_euclideana",
    "dist_clases",
    "clasificar_pixel",
    "classify_image_iterative",
    "classify_image_vectorized",
    "evaluate_predictions",
]
