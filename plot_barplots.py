import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FILES = [
    "HG004_chr20/summary_internal_test_long.csv",
    "HG005_chr20_chr21_5x/summary_hg005_trainmode_5x_long.csv",
    "HG005_chr20_chr21_10x/summary_hg005_trainmode_10x_long.csv",
    "HG005_chr20_chr21_20x/summary_hg005_trainmode_20x_long.csv",
    "HG005_chr20_chr21_40x/summary_hg005_trainmode_40x_long.csv",
    "HG003_ch20_initial_test/hg003_summary_internal_test_long.csv",
    "HG003_ch20_initial_test/hg003_two_stage_hybrid_ablation.csv",
]

COVERAGE_FILES = [
    "HG005_chr20_chr21_5x/summary_hg005_trainmode_5x_long.csv",
    "HG005_chr20_chr21_10x/summary_hg005_trainmode_10x_long.csv",
    "HG005_chr20_chr21_20x/summary_hg005_trainmode_20x_long.csv",
    "HG005_chr20_chr21_40x/summary_hg005_trainmode_40x_long.csv",
]

SPECIAL_ABLATION_FILE = "hg003_two_stage_hybrid_ablation.csv"

SCOPES = {
    "illumina_view": "Illumina",
    "ont_view": "ONT"
}

STANDARD_METRICS = {
    "acc": "Accuracy",
    "macro_f1": "F1 score",
    "macro_recall": "Recall",
    "macro_precision": "Precision"
}

ABLATION_PERFORMANCE_METRICS = {
    "macro_fq": "Macro FQ",
    "macro_recall": "Macro Recall"
}

ABLATION_ERROR_METRICS = {
    "variant_fp": "Variant FP",
    "variant_fn": "Variant FN"
}

MODEL_COLORS = [
    "#1B263B",  # azul 1
    "#415A77",  # azul medio grisáceo
    "#778DA9",  # azul grisáceo
    "#0F766E",  # azul verdoso
    "#14B8A6",  # turquesa
    "#A16207",  # dorado oscuro
    "#D97706",  # ámbar
    "#9F1239",  # granate
    "#BE123C",  # rosa vino
    "#581C87",  # morado
    "#7E22CE",  # violeta
    "#374151",  # gris
    "#2563EB",  # azul más llamativo
    "#059669",  # verde esmeralda
    "#C2410C",  # naranja oscuro
    "#9333EA",  # morado
]


def get_experiment_name(file_path):
    folder_name = os.path.basename(os.path.dirname(file_path))
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    if file_name == "hg003_two_stage_hybrid_ablation":
        return "HG003_two_stage_hybrid_ablation"

    if file_name == "hg003_summary_internal_test_long":
        return "HG003_initial_test"

    return folder_name


def extract_coverage(file_path):
    match = re.search(r"_(5x|10x|20x|40x)", file_path)

    if match:
        return match.group(1)

    folder_name = os.path.basename(os.path.dirname(file_path))
    match = re.search(r"(5x|10x|20x|40x)", folder_name)

    if match:
        return match.group(1)

    return "unknown"


def sort_coverages(coverage):
    order = {
        "5x": 5,
        "10x": 10,
        "20x": 20,
        "40x": 40
    }

    return order.get(coverage, 999)


def resolve_ablation_metric_names(df):
    metrics = ABLATION_PERFORMANCE_METRICS.copy()

    if "macro_f1" not in df.columns and "macro_f1" in df.columns:
        metrics.pop("macro_f1")
        metrics["macro_f1"] = "Macro F1"

    return metrics


def plot_grouped_metrics(
    grouped,
    metrics,
    title,
    output_name,
    model_column="name",
    output_dir="plots",
    y_min=0.8,
    y_max=1.02
):
    os.makedirs(output_dir, exist_ok=True)

    metric_keys = list(metrics.keys())
    metric_labels = [metrics[m] for m in metric_keys]

    models = grouped[model_column].tolist()
    x = np.arange(len(metric_keys))

    num_models = len(models)
    bar_width = min(0.8 / max(num_models, 1), 0.16)

    plt.figure(figsize=(12, 6.5))

    for i, model in enumerate(models):
        model_values = grouped.loc[grouped[model_column] == model, metric_keys].values.flatten()

        offset = (i - num_models / 2) * bar_width + bar_width / 2
        color = MODEL_COLORS[i % len(MODEL_COLORS)]

        bars = plt.bar(
            x + offset,
            model_values,
            width=bar_width,
            label=model,
            color=color,
            edgecolor="#111827",
            linewidth=0.45
        )

        for bar in bars:
            height = bar.get_height()

            if pd.notna(height):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + ((y_max - y_min) * 0.01),
                    f"{height:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90
                )

    plt.ylim(y_min, y_max)
    plt.ylabel("Valor de la métrica")
    plt.xlabel("Métrica")
    plt.title(title)

    plt.xticks(x, metric_labels)
    plt.grid(axis="y", linestyle="--", alpha=0.25)
    plt.legend(
        title="Modelo",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True
    )

    plt.tight_layout()

    output_path = os.path.join(output_dir, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Gráfica guardada en: {output_path}")


def plot_standard_file(df, file_path):
    experiment_name = get_experiment_name(file_path)

    if "scope" not in df.columns:
        print(f"[ERROR] El fichero no contiene la columna 'scope': {file_path}")
        return

    for scope, scope_label in SCOPES.items():
        filtered = df[df["scope"] == scope].copy()

        if filtered.empty:
            print(f"[AVISO] No se encontraron filas con scope='{scope}' en {file_path}")
            continue

        required_columns = ["name"] + list(STANDARD_METRICS.keys())
        missing_columns = [
            col for col in required_columns
            if col not in filtered.columns
        ]

        if missing_columns:
            print(f"[AVISO] Faltan columnas en {file_path}: {missing_columns}")
            continue

        grouped = (
            filtered
            .groupby("name", as_index=False)[list(STANDARD_METRICS.keys())]
            .mean()
        )

        title = f"Comparativa de modelos: {scope_label} - {experiment_name}"
        output_name = f"{experiment_name}_{scope}_metrics_grouped_by_model.png"

        plot_grouped_metrics(
            grouped=grouped,
            metrics=STANDARD_METRICS,
            title=title,
            output_name=output_name,
            model_column="name",
            y_min=0.8,
            y_max=1.02
        )


def plot_special_ablation_file(df, file_path):
    experiment_name = get_experiment_name(file_path)
    model_column = "architecture"

    if "variant_type" not in df.columns:
        print(f"[ERROR] El fichero especial no contiene la columna 'variant_type': {file_path}")
        return

    if model_column not in df.columns:
        print(f"[ERROR] El fichero especial no contiene la columna '{model_column}': {file_path}")
        return

    filtered = df[df["variant_type"] == "ALL"].copy()

    if filtered.empty:
        print(f"[AVISO] No se encontraron filas con variant_type='ALL' en {file_path}")
        return

    performance_metrics = resolve_ablation_metric_names(filtered)

    required_performance_columns = [model_column] + list(performance_metrics.keys())
    missing_performance_columns = [
        col for col in required_performance_columns
        if col not in filtered.columns
    ]

    if missing_performance_columns:
        print(
            f"[AVISO] Faltan columnas para la gráfica de rendimiento en "
            f"{file_path}: {missing_performance_columns}"
        )
    else:
        grouped_performance = (
            filtered
            .groupby(model_column, as_index=False)[list(performance_metrics.keys())]
            .mean()
        )

        plot_grouped_metrics(
            grouped=grouped_performance,
            metrics=performance_metrics,
            title=f"Comparativa de rendimiento por arquitectura - {experiment_name}",
            output_name=f"{experiment_name}_ALL_macro_fq_macro_recall.png",
            model_column=model_column,
            y_min=0.8,
            y_max=1.02
        )

    required_error_columns = [model_column] + list(ABLATION_ERROR_METRICS.keys())
    missing_error_columns = [
        col for col in required_error_columns
        if col not in filtered.columns
    ]

    if missing_error_columns:
        print(
            f"[AVISO] Faltan columnas para la gráfica de FP/FN en "
            f"{file_path}: {missing_error_columns}"
        )
    else:
        grouped_errors = (
            filtered
            .groupby(model_column, as_index=False)[list(ABLATION_ERROR_METRICS.keys())]
            .mean()
        )

        max_error_value = grouped_errors[list(ABLATION_ERROR_METRICS.keys())].max().max()
        y_max = max_error_value * 1.15 if max_error_value > 0 else 1

        plot_grouped_metrics(
            grouped=grouped_errors,
            metrics=ABLATION_ERROR_METRICS,
            title=f"Comparativa de FP y FN - {experiment_name}",
            output_name=f"{experiment_name}_ALL_variant_fp_variant_fn.png",
            model_column=model_column,
            y_min=0,
            y_max=y_max
        )


def collect_f1_by_coverage(scope):
    rows = []

    for file_path in COVERAGE_FILES:
        if not os.path.exists(file_path):
            print(f"[ERROR] No existe el fichero de cobertura: {file_path}")
            continue

        coverage = extract_coverage(file_path)
        df = pd.read_csv(file_path)

        required_columns = ["scope", "name", "macro_f1"]
        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            print(f"[AVISO] Faltan columnas en {file_path}: {missing_columns}")
            continue

        filtered = df[df["scope"] == scope].copy()

        if filtered.empty:
            print(f"[AVISO] No hay filas con scope='{scope}' en {file_path}")
            continue

        grouped = (
            filtered
            .groupby("name", as_index=False)["macro_f1"]
            .mean()
        )

        for _, row in grouped.iterrows():
            rows.append({
                "coverage": coverage,
                "name": row["name"],
                "macro_f1": row["macro_f1"]
            })

    result = pd.DataFrame(rows)

    if not result.empty:
        result["coverage_order"] = result["coverage"].apply(sort_coverages)
        result = result.sort_values(["coverage_order", "name"])

    return result


def plot_f1_by_coverage(scope, scope_label, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    df = collect_f1_by_coverage(scope)

    if df.empty:
        print(f"[AVISO] No se pudieron obtener datos de F1 para {scope}")
        return

    coverages = (
        df[["coverage", "coverage_order"]]
        .drop_duplicates()
        .sort_values("coverage_order")["coverage"]
        .tolist()
    )

    models = sorted(df["name"].unique().tolist())

    x = np.arange(len(coverages))
    num_models = len(models)
    bar_width = min(0.8 / max(num_models, 1), 0.16)

    plt.figure(figsize=(12, 6.5))

    for i, model in enumerate(models):
        values = []

        for coverage in coverages:
            value = df[
                (df["coverage"] == coverage) &
                (df["name"] == model)
            ]["macro_f1"]

            if value.empty:
                values.append(np.nan)
            else:
                values.append(value.iloc[0])

        offset = (i - num_models / 2) * bar_width + bar_width / 2
        color = MODEL_COLORS[i % len(MODEL_COLORS)]

        bars = plt.bar(
            x + offset,
            values,
            width=bar_width,
            label=model,
            color=color,
            edgecolor="#111827",
            linewidth=0.45
        )

        for bar in bars:
            height = bar.get_height()

            if pd.notna(height):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.002,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90
                )

    plt.ylim(0.8, 1.02)
    plt.ylabel("F1 score")
    plt.xlabel("Cobertura")
    plt.title(f"Evolución del F1 score por cobertura para {scope_label}")

    plt.xticks(x, coverages)
    plt.grid(axis="y", linestyle="--", alpha=0.25)
    plt.legend(
        title="Modelo",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True
    )

    plt.tight_layout()

    output_name = f"HG005_f1_score_by_coverage_{scope}.png"
    output_path = os.path.join(output_dir, output_name)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Gráfica guardada en: {output_path}")


def collect_f1_by_coverage_variant(scope_values, model_names):
    rows = []

    model_labels = {
        "dv_illumina": "DeepVariant",
        "dv_ont": "DeepVariant",
        "hybrid_groupwise": "Hybrid Groupwise"
    }

    for file_path in COVERAGE_FILES:
        if not os.path.exists(file_path):
            print(f"[ERROR] No existe el fichero de cobertura: {file_path}")
            continue

        coverage = extract_coverage(file_path)
        df = pd.read_csv(file_path)

        required_columns = ["scope", "name", "macro_f1"]
        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            print(f"[AVISO] Faltan columnas en {file_path}: {missing_columns}")
            continue

        filtered = df[
            (df["scope"].isin(scope_values)) &
            (df["name"].isin(model_names))
        ].copy()

        if filtered.empty:
            print(f"[AVISO] No hay filas válidas para variantes en {file_path}")
            continue

        grouped = (
            filtered
            .groupby(["name", "scope"], as_index=False)["macro_f1"]
            .mean()
        )

        for _, row in grouped.iterrows():
            if row["scope"].endswith("_SNP"):
                variant_type = "SNP"
            elif row["scope"].endswith("_INDEL"):
                variant_type = "INDEL"
            else:
                variant_type = row["scope"]

            model_label = model_labels.get(row["name"], row["name"])

            rows.append({
                "coverage": coverage,
                "coverage_order": sort_coverages(coverage),
                "name": row["name"],
                "model_label": model_label,
                "variant_type": variant_type,
                "series": f"{model_label} - {variant_type}",
                "macro_f1": row["macro_f1"]
            })

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(["coverage_order", "series"])

    return result


def plot_f1_by_coverage_variant_comparison(
    scope_values,
    model_names,
    title,
    output_name,
    output_dir="plots"
):
    os.makedirs(output_dir, exist_ok=True)

    df = collect_f1_by_coverage_variant(
        scope_values=scope_values,
        model_names=model_names
    )

    if df.empty:
        print(f"[AVISO] No se pudieron obtener datos para {title}")
        return

    coverages = (
        df[["coverage", "coverage_order"]]
        .drop_duplicates()
        .sort_values("coverage_order")["coverage"]
        .tolist()
    )

    series_names = sorted(df["series"].unique().tolist())

    x = np.arange(len(coverages))
    num_series = len(series_names)
    bar_width = min(0.8 / max(num_series, 1), 0.16)

    plt.figure(figsize=(12, 6.5))

    for i, series in enumerate(series_names):
        values = []

        for coverage in coverages:
            value = df[
                (df["coverage"] == coverage) &
                (df["series"] == series)
            ]["macro_f1"]

            if value.empty:
                values.append(np.nan)
            else:
                values.append(value.iloc[0])

        offset = (i - num_series / 2) * bar_width + bar_width / 2
        color = MODEL_COLORS[i % len(MODEL_COLORS)]

        bars = plt.bar(
            x + offset,
            values,
            width=bar_width,
            label=series,
            color=color,
            edgecolor="#111827",
            linewidth=0.45
        )

        for bar in bars:
            height = bar.get_height()

            if pd.notna(height):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.002,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90
                )

    plt.ylim(0.8, 1.02)
    plt.ylabel("F1 score")
    plt.xlabel("Cobertura")
    plt.title(title)

    plt.xticks(x, coverages)
    plt.grid(axis="y", linestyle="--", alpha=0.25)
    plt.legend(
        title="Modelo y tipo de variante",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True
    )

    plt.tight_layout()

    output_path = os.path.join(output_dir, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Gráfica guardada en: {output_path}")


def main():
    for file_path in FILES:
        if not os.path.exists(file_path):
            print(f"[ERROR] No existe el fichero: {file_path}")
            continue

        print(f"\nProcesando: {file_path}")

        df = pd.read_csv(file_path)
        file_name = os.path.basename(file_path)

        if file_name == SPECIAL_ABLATION_FILE:
            plot_special_ablation_file(df, file_path)
        else:
            plot_standard_file(df, file_path)

    print("\nGenerando gráficas de F1 score por cobertura...")

    plot_f1_by_coverage(
        scope="illumina_view",
        scope_label="Illumina"
    )

    plot_f1_by_coverage(
        scope="ont_view",
        scope_label="ONT"
    )

    print("\nGenerando gráficas de F1 score por cobertura y tipo de variante...")

    plot_f1_by_coverage_variant_comparison(
        scope_values=["illumina_view_SNP", "illumina_view_INDEL"],
        model_names=["dv_illumina", "hybrid_groupwise"],
        title="Comparativa de F1 score por cobertura y tipo de variante para Illumina",
        output_name="HG005_f1_score_by_coverage_variant_illumina.png"
    )

    plot_f1_by_coverage_variant_comparison(
        scope_values=["ont_view_SNP", "ont_view_INDEL"],
        model_names=["dv_ont", "hybrid_groupwise"],
        title="Comparativa de F1 score por cobertura y tipo de variante para ONT",
        output_name="HG005_f1_score_by_coverage_variant_ont.png"
    )


if __name__ == "__main__":
    main()