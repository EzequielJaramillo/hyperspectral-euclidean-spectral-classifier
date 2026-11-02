"""Módulo de visualización científica para datos hiperespectrales y evaluación de modelos."""

from pathlib import Path
from typing import Dict, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from .data_loader import CLASS_NAMES

# Paleta de colores semántica para las 7 clases de frutas + fondo
FRUIT_PALETTE = [
    "#111827",  # 0: Background (gris muy oscuro)
    "#FACC15",  # 1: Banana (amarillo plátano)
    "#4ADE80",  # 2: Green Apple (verde manzana)
    "#FEF08A",  # 3: Lemon (amarillo limón claro)
    "#A3E635",  # 4: Starfruit (verde lima carambola)
    "#FB923C",  # 5: Mango (naranja mango)
    "#DC2626",  # 6: Red Pepper (rojo intenso pimiento)
    "#991B1B",  # 7: Red Apple (rojo carmesí manzana)
]


def get_fruit_cmap() -> tuple:
    """Devuelve el colormap discreto y su normalizador para las clases 0..7."""
    cmap = ListedColormap(FRUIT_PALETTE)
    bounds = np.arange(-0.5, 8.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def _save_fig(fig: plt.Figure, save_path: Optional[Union[str, Path]]) -> None:
    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")


def plot_dataset_overview(
    hsi: np.ndarray,
    gt: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """Genera la vista comparativa del Ground Truth 2D, primera banda espectral y falso color RGB."""
    cmap_gt, norm_gt = get_fruit_cmap()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    # Panel 1: Ground Truth con leyenda semántica
    im0 = axes[0].imshow(gt, cmap=cmap_gt, norm=norm_gt)
    axes[0].set_title("Ground Truth 2D (7 Clases + Fondo)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Columna (px)")
    axes[0].set_ylabel("Fila (px)")
    patches = [
        mpatches.Patch(color=FRUIT_PALETTE[k], label=f"{k}: {CLASS_NAMES[k]}")
        for k in range(8)
    ]
    axes[0].legend(
        handles=patches,
        loc="lower left",
        fontsize=7.5,
        framealpha=0.9,
        facecolor="white",
    )

    # Panel 2: Banda 0 original
    im1 = axes[1].imshow(hsi[0, :, :], cmap="viridis")
    axes[1].set_title("Primera Banda Espectral HSI[0, :, :]", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Columna (px)")
    axes[1].set_ylabel("Fila (px)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Intensidad (uint16)")

    # Panel 3: Compuesto pseudo-RGB (bandas 80, 50, 20 normalizadas)
    rgb = np.stack([hsi[80], hsi[50], hsi[20]], axis=-1).astype(np.float64)
    for c in range(3):
        p99 = np.percentile(rgb[:, :, c], 99.5)
        if p99 > 0:
            rgb[:, :, c] = np.clip(rgb[:, :, c] / p99, 0, 1)
    axes[2].imshow(rgb)
    axes[2].set_title("Compuesto Falso Color (Bandas 80, 50, 20)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Columna (px)")
    axes[2].set_ylabel("Fila (px)")

    plt.suptitle(
        "Exploración Inicial del Cubo Hiperespectral (120 × 640 × 640)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    _save_fig(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_train_test_coordinates(
    train_coords_by_class: Dict[float, np.ndarray],
    test_coords_by_class: Dict[float, np.ndarray],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """Grafica la distribución espacial de los píxeles de entrenamiento (20%) y prueba (80%)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(train_coords_by_class)))

    for (cls, coords), color in zip(train_coords_by_class.items(), colors):
        c_int = int(cls)
        axes[0].scatter(
            coords[:, 1],
            coords[:, 0],
            color=color,
            s=0.35,
            label=f"({c_int}) {CLASS_NAMES[c_int]} (n={len(coords):,})",
        )
    axes[0].set_title("Coordenadas de Entrenamiento (20% — 17,033 px)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Columna X (px)")
    axes[0].set_ylabel("Fila Y (px)")
    axes[0].set_xlim(0, 640)
    axes[0].set_ylim(640, 0)
    axes[0].legend(markerscale=12, fontsize=8.5, loc="lower left", framealpha=0.92)
    axes[0].grid(True, linestyle="--", alpha=0.3)

    for (cls, coords), color in zip(test_coords_by_class.items(), colors):
        c_int = int(cls)
        axes[1].scatter(
            coords[:, 1],
            coords[:, 0],
            color=color,
            s=0.35,
            label=f"({c_int}) {CLASS_NAMES[c_int]} (n={len(coords):,})",
        )
    axes[1].set_title("Coordenadas de Prueba (80% — 68,150 px)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Columna X (px)")
    axes[1].set_ylabel("Fila Y (px)")
    axes[1].set_xlim(0, 640)
    axes[1].set_ylim(640, 0)
    axes[1].legend(markerscale=12, fontsize=8.5, loc="lower left", framealpha=0.92)
    axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.suptitle(
        "Entregable 1: Partición Aleatoria Estratificada por Clase (Semilla = 0)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    _save_fig(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_mean_spectra(
    espectros_promedio: Dict[float, np.ndarray],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """Grafica las firmas espectrales promedio P_c de las 7 clases a lo largo de las 120 bandas."""
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(espectros_promedio)))
    bands = np.arange(1, 121)

    for (cls, spectrum), color in zip(espectros_promedio.items(), colors):
        c_int = int(cls)
        ax.plot(
            bands,
            spectrum,
            linewidth=2.3,
            color=color,
            label=f"Clase {c_int}: {CLASS_NAMES[c_int]}",
        )

    ax.set_title(
        "Entregable 2: Comportamiento Espectral de los Píxeles Representativos ($P_c$)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Banda Espectral (1 a 120)", fontsize=11)
    ax.set_ylabel("Intensidad Promedio de Reflectancia", fontsize=11)
    ax.set_xlim(1, 120)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=10, frameon=True, facecolor="white", framealpha=0.95)

    plt.tight_layout()
    _save_fig(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_classification_maps(
    gt: np.ndarray,
    full_classified: np.ndarray,
    test_classified: np.ndarray,
    save_full_path: Optional[Union[str, Path]] = None,
    save_test_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> None:
    """Genera las imágenes de los Entregables 3 (imagen completa) y 4 (píxeles de prueba)."""
    # Entregable 3: Imagen completa clasificada
    fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5.6))
    im_orig = axes1[0].imshow(full_classified, cmap="tab20")
    axes1[0].set_title("Clasificación Completa (Colormap tab20 Original)", fontsize=11.5, fontweight="bold")
    plt.colorbar(im_orig, ax=axes1[0], fraction=0.046, pad=0.04)

    cmap_sem, norm_sem = get_fruit_cmap()
    axes1[1].imshow(full_classified, cmap=cmap_sem, norm=norm_sem)
    axes1[1].set_title("Clasificación Completa (Leyenda por Fruta)", fontsize=11.5, fontweight="bold")
    patches = [
        mpatches.Patch(color=FRUIT_PALETTE[k], label=f"({k}) {CLASS_NAMES[k]}")
        for k in range(1, 8)
    ]
    axes1[1].legend(handles=patches, loc="lower left", fontsize=8, framealpha=0.92)
    plt.suptitle(
        "Entregable 3: Clasificación de Todos los Píxeles de la Imagen Hiperespectral",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    _save_fig(fig1, save_full_path)
    if show:
        plt.show()
    else:
        plt.close(fig1)

    # Entregable 4: Píxeles de prueba clasificados vs Ground Truth
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5.6))
    im_test = axes2[0].imshow(test_classified, cmap="tab20")
    axes2[0].set_title("Píxeles de Prueba Clasificados (80% — tab20)", fontsize=11.5, fontweight="bold")
    plt.colorbar(im_test, ax=axes2[0], fraction=0.046, pad=0.04)

    axes2[1].imshow(test_classified, cmap=cmap_sem, norm=norm_sem)
    axes2[1].set_title("Píxeles de Prueba Clasificados (Leyenda Semántica)", fontsize=11.5, fontweight="bold")
    patches_bg = [
        mpatches.Patch(color=FRUIT_PALETTE[k], label=f"({k}) {CLASS_NAMES[k]}")
        for k in range(0, 8)
    ]
    axes2[1].legend(handles=patches_bg, loc="lower left", fontsize=8, framealpha=0.92)
    plt.suptitle(
        "Entregable 4: Clasificación Exclusiva de Píxeles de Prueba (Excluyendo Entrenamiento y Fondo)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    _save_fig(fig2, save_test_path)
    if show:
        plt.show()
    else:
        plt.close(fig2)


def plot_confusion_matrix_pair(
    cm: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """Genera la matriz de confusión en conteos absolutos y normalizada por fila (recall)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    short_labels = [f"{k}: {CLASS_NAMES[k]}" for k in range(1, 8)]

    disp_abs = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=short_labels)
    disp_abs.plot(cmap="Blues", ax=axes[0], values_format="d", colorbar=False)
    axes[0].set_title("Matriz de Confusión (Conteo de Píxeles de Prueba)", fontsize=11.5, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=30)

    cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=short_labels)
    disp_norm.plot(cmap="Blues", ax=axes[1], values_format=".2f", colorbar=False)
    axes[1].set_title("Matriz de Confusión Normalizada (Recall por Clase)", fontsize=11.5, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=30)

    plt.suptitle(
        "Entregable 5: Evaluación en el Conjunto de Prueba (68,150 Píxeles — Exactitud Global: 63.90%)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    _save_fig(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_metrics_and_benchmark(
    report_dict: Dict[str, dict],
    iter_time: float,
    vec_time: float,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """Grafica las métricas por clase (Entregable 6) y la comparativa de tiempos (Entregable 7)."""
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5))

    classes = [str(float(k)) for k in range(1, 8)]
    names = [f"({k}) {CLASS_NAMES[k]}" for k in range(1, 8)]
    precisions = [report_dict[c]["precision"] * 100 for c in classes]
    recalls = [report_dict[c]["recall"] * 100 for c in classes]
    f1s = [report_dict[c]["f1-score"] * 100 for c in classes]

    x = np.arange(len(names))
    w = 0.26
    axes[0].bar(x - w, precisions, width=w, label="Precisión (%)", color="#2563EB")
    axes[0].bar(x, recalls, width=w, label="Recall / Sensibilidad (%)", color="#10B981")
    axes[0].bar(x + w, f1s, width=w, label="F1-Score (%)", color="#F59E0B")
    axes[0].axhline(
        report_dict["accuracy"] * 100,
        color="#DC2626",
        linestyle="--",
        linewidth=1.5,
        label=f"Exactitud Global ({report_dict['accuracy']*100:.2f}%)",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=28, ha="right", fontsize=9)
    axes[0].set_ylim(0, 108)
    axes[0].set_ylabel("Porcentaje (%)")
    axes[0].set_title("Entregable 6: Desempeño por Clase en Prueba", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=8.5, loc="upper left")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    # Panel derecho: Benchmark de tiempos (Iterativo vs Vectorizado)
    methods = ["Iterativo (Bucles Anidados)\n409,600 llamadas Python", "Vectorizado (Álgebra Lineal NumPy)\nProducto Matricial BLAS"]
    times = [iter_time, vec_time]
    speedup = iter_time / max(vec_time, 1e-6)

    bars = axes[1].bar(methods, times, color=["#EF4444", "#10B981"], width=0.48)
    axes[1].set_ylabel("Tiempo de Ejecución (segundos)")
    axes[1].set_title(
        f"Entregable 7: Optimización de Tiempo de Ejecución (Aceleración {speedup:.1f}x)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    for bar, t in zip(bars, times):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(times) * 0.03,
            f"{t:.3f} s",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    axes[1].set_ylim(0, max(times) * 1.18)

    plt.tight_layout()
    _save_fig(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
