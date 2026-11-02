"""Módulo del clasificador espectral de mínima distancia euclideana (iterativo y vectorizado)."""

import time
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def compute_class_centroids(
    train_spectra_by_class: Dict[float, np.ndarray]
) -> Dict[float, np.ndarray]:
    """Calcula el espectro promedio (centroide P_c de 120 bandas) para cada clase."""
    centroids: Dict[float, np.ndarray] = {}
    for cls, spectra in train_spectra_by_class.items():
        centroids[cls] = np.mean(spectra, axis=0)
    return centroids


def dist_euclideana(espectro1: np.ndarray, espectro2: np.ndarray) -> float:
    """Calcula la distancia euclideana (norma L2) entre dos vectores espectrales.

    Ecuación:
        d(x, y) = ||x - y||_2 = sqrt( sum_{k=1}^{B} (x_k - y_k)^2 )
    """
    return float(np.linalg.norm(espectro1 - espectro2))


def dist_clases(
    espectro: np.ndarray, espectros_promedio: Dict[float, np.ndarray]
) -> List[float]:
    """Calcula la distancia euclideana desde un espectro desconocido a los 7 espectros promedio."""
    return [
        dist_euclideana(espectro, espectro_prom)
        for espectro_prom in espectros_promedio.values()
    ]


def clasificar_pixel(
    espectro: np.ndarray, espectros_promedio: Dict[float, np.ndarray]
) -> int:
    """Clasifica un espectro de acuerdo con la regla de mínima distancia euclideana.

    Devuelve el índice 0-based (0 a 6) de la clase con menor distancia.
    """
    distancias = dist_clases(espectro, espectros_promedio)
    return int(np.argmin(distancias))


def classify_image_iterative(
    hsi: np.ndarray, espectros_promedio: Dict[float, np.ndarray]
) -> Tuple[np.ndarray, float]:
    """Clasifica todos los píxeles no nulos del cubo HSI de forma iterativa (píxel por píxel).

    Returns
    -------
    Tuple[np.ndarray, float]
        Mapa 2D clasificado (640, 640) con etiquetas 1..7 (0 para fondo nulo) y tiempo en segundos.
    """
    _, rows, cols = hsi.shape
    classified_img = np.zeros((rows, cols), dtype=np.float64)

    t0 = time.perf_counter()
    for i in range(rows):
        for j in range(cols):
            espectro = hsi[:, i, j]
            if np.any(espectro):
                classified_img[i, j] = clasificar_pixel(espectro, espectros_promedio) + 1
    elapsed = time.perf_counter() - t0
    return classified_img, elapsed


def classify_image_vectorized(
    hsi: np.ndarray, espectros_promedio: Dict[float, np.ndarray]
) -> Tuple[np.ndarray, float]:
    """Clasifica el cubo HSI completo mediante álgebra lineal vectorizada en NumPy.

    Aprovecha la identidad algebraica:
        ||x - c||_2^2 = ||x||_2^2 + ||c||_2^2 - 2 * (x · c^T)
    evitando bucles en Python sobre los 409,600 píxeles y reduciendo el tiempo de ~21 s a ~0.12 s.

    Returns
    -------
    Tuple[np.ndarray, float]
        Mapa 2D clasificado (640, 640) con etiquetas 1..7 (0 para fondo nulo) y tiempo en segundos.
    """
    bands, rows, cols = hsi.shape
    t0 = time.perf_counter()

    # Reorganizar cubo a matriz 2D de (N_pixeles, 120) en float64
    flat_pixels = hsi.reshape(bands, -1).T.astype(np.float64)
    centroids_matrix = np.vstack(
        [espectros_promedio[k] for k in sorted(espectros_promedio.keys())]
    ).astype(np.float64)  # (7, 120)

    # Máscara de píxeles con señal no nula
    valid_mask = np.any(flat_pixels != 0, axis=1)

    # Cálculo vectorizado de distancias cuadráticas hacia los 7 centroides
    # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x c^T
    c_norm_sq = np.sum(centroids_matrix ** 2, axis=1)[None, :]  # (1, 7)
    dots = flat_pixels @ centroids_matrix.T  # (409600, 7)
    dist_sq = c_norm_sq - 2.0 * dots  # ||x||^2 es constante respecto a la clase argmin

    pred_labels = np.argmin(dist_sq, axis=1) + 1
    pred_labels[~valid_mask] = 0

    classified_img = pred_labels.reshape(rows, cols).astype(np.float64)
    elapsed = time.perf_counter() - t0
    return classified_img, elapsed


def build_test_classification_map(
    classified_img: np.ndarray, test_coords_by_class: Dict[float, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Filtra el mapa clasificado conservando únicamente los píxeles del conjunto de prueba (80%)."""
    test_mask = np.zeros_like(classified_img, dtype=bool)
    for coords in test_coords_by_class.values():
        coords_arr = np.asarray(coords)
        test_mask[coords_arr[:, 0], coords_arr[:, 1]] = True

    test_classified_img = classified_img * test_mask
    return test_classified_img, test_mask


def evaluate_predictions(
    classified_img: np.ndarray, test_coords_by_class: Dict[float, np.ndarray]
) -> Dict[str, object]:
    """Evalúa el desempeño del clasificador sobre los 68,150 píxeles de prueba."""
    y_true_list: List[float] = []
    y_pred_list: List[float] = []

    for cls, coords in test_coords_by_class.items():
        coords_arr = np.asarray(coords)
        preds = classified_img[coords_arr[:, 0], coords_arr[:, 1]]
        y_true_list.extend([float(cls)] * len(coords_arr))
        y_pred_list.extend(preds.tolist())

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    labels = np.arange(1, 8)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = float(accuracy_score(y_true, y_pred))
    report_str = classification_report(y_true, y_pred, digits=2)
    report_dict = classification_report(y_true, y_pred, output_dict=True)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "confusion_matrix": cm,
        "accuracy": acc,
        "report_str": report_str,
        "report_dict": report_dict,
    }

