"""Punto de entrada principal (CLI) para el Clasificador Espectral de Distancias Euclideanas."""

import argparse
from pathlib import Path
import numpy as np

from src.data_loader import (
    CLASS_NAMES,
    load_hsi_dataset,
    get_class_distribution,
    extract_coordinates_by_class,
    extract_spectra_by_class,
    train_test_split_by_class,
)
from src.classifier import (
    compute_class_centroids,
    classify_image_iterative,
    classify_image_vectorized,
    build_test_classification_map,
    evaluate_predictions,
)
from src.visualization import (
    plot_dataset_overview,
    plot_train_test_coordinates,
    plot_mean_spectra,
    plot_classification_maps,
    plot_confusion_matrix_pair,
    plot_metrics_and_benchmark,
)


def run_pipeline(
    base_dir: Path,
    assets_dir: Path,
    run_iterative_benchmark: bool = True,
) -> None:
    """Ejecuta el pipeline completo de extremo a extremo y exporta las visualizaciones a assets/."""
    print("=" * 78)
    print("CLASIFICADOR ESPECTRAL DE DISTANCIAS EUCLIDEANAS — PIPELINE DE EJECUCIÓN")
    print("=" * 78)

    # 1. Ingesta de datos
    print("\n[1/7] Cargando cubo hiperespectral (fruit3D.mat) y Ground Truth (groundTruth.mat)...")
    hsi, gt = load_hsi_dataset(base_dir=base_dir)
    print(f"  -> Dimensiones HSI: {hsi.shape} (dtype={hsi.dtype})")
    print(f"  -> Dimensiones GT : {gt.shape} (dtype={gt.dtype})")

    # 2. Distribución de clases
    print("\n[2/7] Distribución de píxeles por clase en Ground Truth:")
    dist = get_class_distribution(gt)
    total_px = gt.size
    for cls_id, count in dist.items():
        pct = 100.0 * count / total_px
        print(f"  -> Clase {cls_id} ({CLASS_NAMES[cls_id]:<12}): {count:>7,} px ({pct:>5.2f}%)")

    # 3. Extracción de coordenadas y firmas espectrales (clases 1 a 7)
    print("\n[3/7] Extrayendo coordenadas espaciales y firmas espectrales (N_c x 120)...")
    coords_by_class = extract_coordinates_by_class(gt, include_background=False)
    spectra_by_class = extract_spectra_by_class(hsi, coords_by_class)
    for cls_id, spec in spectra_by_class.items():
        print(f"  -> Clase {int(cls_id)} ({CLASS_NAMES[int(cls_id)]:<12}): {spec.shape}")

    # 4. Partición entrenamiento (20%) y prueba (80%) con semilla reproducible (seed=0)
    print("\n[4/7] Dividiendo muestras en Entrenamiento (20%) y Prueba (80%) [seed=0]...")
    train_spec, test_spec, train_coords, test_coords = train_test_split_by_class(
        hsi, coords_by_class, train_ratio=0.2, seed=0
    )
    total_train = sum(len(v) for v in train_spec.values())
    total_test = sum(len(v) for v in test_spec.values())
    print(f"  -> Total entrenamiento: {total_train:,} píxeles")
    print(f"  -> Total prueba       : {total_test:,} píxeles")

    # 5. Cálculo de espectros promedio (centroides P_c)
    print("\n[5/7] Calculando firmas espectrales promedio (centroides P_c) por clase...")
    centroids = compute_class_centroids(train_spec)

    # 6. Clasificación de toda la imagen (Vectorizada vs Iterativa)
    print("\n[6/7] Clasificando los 409,600 píxeles del cubo hiperespectral...")
    classified_vec, vec_time = classify_image_vectorized(hsi, centroids)
    print(f"  -> Tiempo vectorizado (NumPy BLAS): {vec_time:.4f} s")

    if run_iterative_benchmark:
        print("  -> Ejecutando método iterativo original (píxel por píxel) para comparativa...")
        classified_iter, iter_time = classify_image_iterative(hsi, centroids)
        identical = np.array_equal(classified_vec, classified_iter)
        print(f"  -> Tiempo iterativo (bucles Python): {iter_time:.4f} s")
        print(f"  -> Equivalencia exacta de mapas    : {identical}")
    else:
        iter_time = 21.49  # Tiempo de referencia medido en el notebook original

    speedup = iter_time / max(vec_time, 1e-6)
    print(f"  -> Aceleración obtenida (Speedup)  : {speedup:.1f}x más rápido")

    # 7. Evaluación en el conjunto de prueba (80%) y generación de reportes visuales
    print("\n[7/7] Evaluando predicciones en el conjunto de prueba y exportando figuras a assets/...")
    test_classified_map, _ = build_test_classification_map(classified_vec, test_coords)
    eval_res = evaluate_predictions(test_classified_map, test_coords)

    print(f"\n  Exactitud Global (Accuracy): {eval_res['accuracy']*100:.2f}%")
    print("\n  Reporte de Clasificación por Clase:")
    print(eval_res["report_str"])

    assets_dir.mkdir(parents=True, exist_ok=True)
    plot_dataset_overview(hsi, gt, save_path=assets_dir / "01_hsi_gt_overview.png")
    plot_train_test_coordinates(
        train_coords, test_coords, save_path=assets_dir / "02_train_test_split.png"
    )
    plot_mean_spectra(centroids, save_path=assets_dir / "03_mean_spectral_signatures.png")
    plot_classification_maps(
        gt,
        classified_vec,
        test_classified_map,
        save_full_path=assets_dir / "04_full_image_classification.png",
        save_test_path=assets_dir / "05_test_pixels_classification.png",
    )
    plot_confusion_matrix_pair(
        eval_res["confusion_matrix"], save_path=assets_dir / "06_confusion_matrix.png"
    )
    plot_metrics_and_benchmark(
        eval_res["report_dict"],
        iter_time=iter_time,
        vec_time=vec_time,
        save_path=assets_dir / "07_performance_benchmark.png",
    )
    print(f"Figuras exportadas exitosamente en: {assets_dir.resolve()}")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ejecuta el pipeline completo del Clasificador Espectral de Distancias Euclideanas."
    )
    parser.add_argument(
        "--skip-iterative",
        action="store_true",
        help="Omite el bucle iterativo lento (~21 s) y usa directamente el motor vectorizado.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    run_pipeline(
        base_dir=project_root,
        assets_dir=project_root / "assets",
        run_iterative_benchmark=not args.skip_iterative,
    )

