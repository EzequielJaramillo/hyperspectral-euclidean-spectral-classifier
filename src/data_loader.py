"""Módulo de ingesta, exploración y partición de datos hiperespectrales (HSI)."""

from pathlib import Path
from typing import Dict, List, Tuple, Union
import h5py
import numpy as np

# Diccionario oficial de clases del dataset de frutas (1 a 7) y fondo (0)
CLASS_NAMES: Dict[int, str] = {
    0: "Background",
    1: "Banana",
    2: "Green Apple",
    3: "Lemon",
    4: "Starfruit",
    5: "Mango",
    6: "Red Pepper",
    7: "Red Apple",
}

CLASS_NAMES_ES: Dict[int, str] = {
    0: "Fondo (Sin etiqueta)",
    1: "Banana (Plátano)",
    2: "Green Apple (Manzana Verde)",
    3: "Lemon (Limón)",
    4: "Starfruit (Carambola)",
    5: "Mango",
    6: "Red Pepper (Pimiento Rojo)",
    7: "Red Apple (Manzana Roja)",
}


def resolve_data_path(filename: str, base_dir: Union[str, Path] = ".") -> Path:
    """Busca un archivo .mat en la raíz del proyecto o dentro del subdirectorio data/."""
    base = Path(base_dir).resolve()
    candidates = [
        base / filename,
        base / "data" / filename,
        Path(__file__).resolve().parent.parent / filename,
        Path(__file__).resolve().parent.parent / "data" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No se encontró el archivo '{filename}'. Ubíquelo en la raíz del repositorio o en 'data/'."
    )


def load_hsi_dataset(
    hsi_filename: str = "fruit3D.mat",
    gt_filename: str = "groundTruth.mat",
    base_dir: Union[str, Path] = ".",
) -> Tuple[np.ndarray, np.ndarray]:
    """Carga el cubo hiperespectral (HSI) y la máscara Ground Truth (GT) desde archivos HDF5 (.mat v7.3).

    Parameters
    ----------
    hsi_filename : str
        Nombre del archivo .mat con el cubo hiperespectral (clave 'HSI').
    gt_filename : str
        Nombre del archivo .mat con las etiquetas 2D (clave 'groundTruth2D').
    base_dir : str | Path
        Directorio base de búsqueda.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (HSI, GT) con dimensiones (120, 640, 640) y (640, 640) respectivamente.
    """
    hsi_path = resolve_data_path(hsi_filename, base_dir)
    gt_path = resolve_data_path(gt_filename, base_dir)

    with h5py.File(hsi_path, "r") as f_hsi:
        hsi = np.array(f_hsi.get("HSI"))

    with h5py.File(gt_path, "r") as f_gt:
        gt = np.array(f_gt.get("groundTruth2D"))

    return hsi, gt


def get_class_distribution(gt: np.ndarray) -> Dict[int, int]:
    """Calcula los valores únicos en GT y el número exacto de ocurrencias por clase."""
    unique_vals, counts = np.unique(gt, return_counts=True)
    return {int(val): int(cnt) for val, cnt in zip(unique_vals, counts)}


def extract_coordinates_by_class(
    gt: np.ndarray, include_background: bool = False
) -> Dict[float, List[Tuple[int, int]]]:
    """Extrae la lista de coordenadas (fila, columna) para cada clase en el arreglo GT.

    Parameters
    ----------
    gt : np.ndarray
        Matriz 2D de etiquetas ground truth (640, 640).
    include_background : bool
        Si es True, incluye la clase 0 (fondo); si es False, solo incluye las clases 1 a 7.

    Returns
    -------
    Dict[float, List[Tuple[int, int]]]
        Diccionario mapeando cada etiqueta de clase a su lista de tuplas (i, j).
    """
    unique_classes = np.unique(gt)
    if not include_background:
        unique_classes = unique_classes[unique_classes != 0]

    coords_by_class: Dict[float, List[Tuple[int, int]]] = {}
    for cls in unique_classes:
        rows, cols = np.where(gt == cls)
        coords_by_class[float(cls)] = list(zip(rows.tolist(), cols.tolist()))
    return coords_by_class


def extract_spectra_by_class(
    hsi: np.ndarray, coords_by_class: Dict[float, List[Tuple[int, int]]]
) -> Dict[float, np.ndarray]:
    """Extrae y organiza los espectros completos (N_c x 120) para cada clase c."""
    spectra_by_class: Dict[float, np.ndarray] = {}
    for cls, coords in coords_by_class.items():
        coords_arr = np.array(coords)
        # Indexación vectorizada sobre las 120 bandas: resultado (N_c, 120)
        spectra_by_class[cls] = hsi[:, coords_arr[:, 0], coords_arr[:, 1]].T
    return spectra_by_class


def train_test_split_by_class(
    hsi: np.ndarray,
    coords_by_class: Dict[float, List[Tuple[int, int]]],
    train_ratio: float = 0.2,
    seed: int = 0,
) -> Tuple[
    Dict[float, np.ndarray],
    Dict[float, np.ndarray],
    Dict[float, np.ndarray],
    Dict[float, np.ndarray],
]:
    """Divide las muestras de cada clase en conjuntos de entrenamiento (20%) y prueba (80%).

    Mantiene la semilla aleatoria exacta (np.random.seed(0)) del flujo experimental original
    para garantizar 100% de reproducibilidad en las métricas y matrices de confusión.
    """
    np.random.seed(seed)
    train_spectra: Dict[float, np.ndarray] = {}
    test_spectra: Dict[float, np.ndarray] = {}
    train_coords: Dict[float, np.ndarray] = {}
    test_coords: Dict[float, np.ndarray] = {}

    for cls, coords in coords_by_class.items():
        coord_matrix = np.array(coords)
        indices = np.arange(len(coord_matrix))
        np.random.shuffle(indices)

        n_train = int(train_ratio * len(coord_matrix))
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        train_coords[cls] = coord_matrix[train_idx]
        test_coords[cls] = coord_matrix[test_idx]

        spectra_array = hsi[:, coord_matrix[:, 0], coord_matrix[:, 1]].T
        train_spectra[cls] = spectra_array[train_idx]
        test_spectra[cls] = spectra_array[test_idx]

    return train_spectra, test_spectra, train_coords, test_coords

