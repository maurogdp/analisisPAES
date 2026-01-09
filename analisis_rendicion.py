"""Herramientas de análisis para archivo de rendición.

Ejemplos de uso:
  python analisis_rendicion.py --list-columns
  python analisis_rendicion.py --situacion-egreso 1 --region 6 --count-by CODIGO_COMUNA
  python analisis_rendicion.py --min-score CLEC_REG_ACTUAL=500 --min-score MATE1_REG_ACTUAL=500 \
      --output-csv filtrados.csv --add-labels
  python analisis_rendicion.py --region 6 --stats CLEC_REG_ACTUAL --stats MATE1_REG_ACTUAL \
      --percentiles 25,50,75
  python analisis_rendicion.py --rbd 12345,67890 --stats CLEC_REG_ACTUAL --stats MATE1_REG_ACTUAL \
      --stats-by-rbd
  python analisis_rendicion.py --sort-by CODIGO_REGION:asc --sort-by PUNTAJE:desc --output-csv ordenados.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_DATA = (
    "PROCESO-DE-ADMISIÓN-2025-RENDICIÓN-19-01-2025T23-39-20/"
    "Rinden_Admisión2025/ArchivoC_Adm2025_corregido_final.csv"
)
DEFAULT_COD_ENS = (
    "PROCESO-DE-ADMISIÓN-2025-RENDICIÓN-19-01-2025T23-39-20/"
    "Rinden_Admisión2025/Libro_CódigosADM2025_ArchivoC_Anexo - COD_ENS.csv"
)
DEFAULT_COMUNAS = (
    "PROCESO-DE-ADMISIÓN-2025-RENDICIÓN-19-01-2025T23-39-20/"
    "Rinden_Admisión2025/Libro_CódigosADM2025_ArchivoC_Anexo - ComunasRegiones.csv"
)
DEFAULT_CODEBOOK = (
    "PROCESO-DE-ADMISIÓN-2025-RENDICIÓN-19-01-2025T23-39-20/"
    "Rinden_Admisión2025/Libro_CódigosADM2025_ArchivoC_Rinden.csv"
)
DEFAULT_RBD_COLEGIOS = "rbd_colegios_chile_2021_funcionando_geoportal.csv"


@dataclass(frozen=True)
class ScoreFilter:
    column: str
    threshold: float


@dataclass
class CodeMaps:
    cod_ens: Dict[str, str]
    regiones: Dict[str, str]
    comunas: Dict[str, str]
    value_labels: Dict[str, Dict[str, str]]
    rbd_info: Dict[str, "RbdInfo"]


@dataclass(frozen=True)
class SortKey:
    column: str
    descending: bool = False


@dataclass
class WeightingConfig:
    weights: Dict[str, float]
    history_science_mode: str
    enabled: bool = False


@dataclass
class RankingConfig:
    grouping_column: str
    value_columns: List[str]
    metric: str
    order_desc: bool
    limit: Optional[int]


@dataclass(frozen=True)
class RbdInfo:
    name: str
    comuna: str
    region: str


class AnalysisError(Exception):
    pass


MAX_SCORE_GROUPS: Dict[str, List[str]] = {
    "CLEC_MAX": [
        "CLEC_REG_ACTUAL",
        "CLEC_INV_ACTUAL",
        "CLEC_REG_ANTERIOR",
        "CLEC_INV_ANTERIOR",
    ],
    "MATE1_MAX": [
        "MATE1_REG_ACTUAL",
        "MATE1_INV_ACTUAL",
        "MATE1_REG_ANTERIOR",
        "MATE1_INV_ANTERIOR",
    ],
    "MATE2_MAX": [
        "MATE2_REG_ACTUAL",
        "MATE2_INV_ACTUAL",
        "MATE2_REG_ANTERIOR",
        "MATE2_INV_ANTERIOR",
    ],
    "HCSOC_MAX": [
        "HCSOC_REG_ACTUAL",
        "HCSOC_INV_ACTUAL",
        "HCSOC_REG_ANTERIOR",
        "HCSOC_INV_ANTERIOR",
    ],
    "CIEN_MAX": [
        "CIEN_REG_ACTUAL",
        "CIEN_INV_ACTUAL",
        "CIEN_REG_ANTERIOR",
        "CIEN_INV_ANTERIOR",
    ],
}
SCIENCE_MODULE_COLUMNS = [
    ("CIEN_REG_ACTUAL", "MODULO_REG_ACTUAL"),
    ("CIEN_INV_ACTUAL", "MODULO_INV_ACTUAL"),
    ("CIEN_REG_ANTERIOR", "MODULO_REG_ANTERIOR"),
    ("CIEN_INV_ANTERIOR", "MODULO_INV_ANTERIOR"),
]
SCIENCE_MAX_MODULE_COLUMN = "MODULO_CIEN_MAX"
WEIGHTED_SCORE_COLUMN = "PUNTAJE_PONDERADO"
WEIGHTED_SUBJECTS = [
    ("nem", "NEM"),
    ("ranking", "Ranking"),
    ("clec", "Competencia Lectora"),
    ("mate1", "Matemática 1"),
    ("mate2", "Matemática 2"),
    ("historia", "Historia/Ciencias Sociales"),
    ("ciencias", "Ciencias"),
]

def discover_rendicion_csvs(base_dir: Path) -> List[Path]:
    preferred: List[Path] = []
    fallback: List[Path] = []
    for path in base_dir.rglob("*.csv"):
        normalized = str(path).casefold()
        if "rendición" not in normalized and "rendicion" not in normalized:
            continue
        if path.name.startswith("ArchivoC_Adm"):
            preferred.append(path)
        else:
            fallback.append(path)
    candidates = preferred or fallback
    return sorted(candidates)


def sniff_dialect(path: Path) -> csv.Dialect:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        sample = handle.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
    except csv.Error:
        return csv.get_dialect("excel")


def read_csv_rows(path: Path) -> Tuple[List[str], Iterable[Dict[str, str]]]:
    dialect = sniff_dialect(path)
    handle = path.open("r", encoding="utf-8", errors="replace", newline="")
    reader = csv.DictReader(handle, dialect=dialect)
    if not reader.fieldnames:
        handle.close()
        raise AnalysisError(f"Archivo vacío o sin cabeceras: {path}")

    def row_iter() -> Iterable[Dict[str, str]]:
        try:
            for row in reader:
                yield row
        finally:
            handle.close()

    return list(reader.fieldnames), row_iter()


def normalize_code(value: Optional[str]) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_list_arg(value: Optional[str]) -> Optional[set[str]]:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_score_filters(values: Sequence[str]) -> List[ScoreFilter]:
    filters: List[ScoreFilter] = []
    for raw in values:
        if "=" not in raw:
            raise AnalysisError(f"Filtro inválido '{raw}'. Usa el formato COLUMNA=NUMERO.")
        column, threshold = raw.split("=", 1)
        column = column.strip()
        threshold = threshold.strip()
        if not column or not threshold:
            raise AnalysisError(f"Filtro inválido '{raw}'. Usa el formato COLUMNA=NUMERO.")
        try:
            value = float(threshold)
        except ValueError as exc:
            raise AnalysisError(
                f"No se pudo convertir '{threshold}' a número para '{column}'."
            ) from exc
        filters.append(ScoreFilter(column=column, threshold=value))
    return filters


def parse_sort_by_args(values: Sequence[str], fieldnames: Sequence[str]) -> List[SortKey]:
    sort_keys: List[SortKey] = []
    for raw in values:
        if not raw:
            continue
        column = raw
        direction = "asc"
        if ":" in raw:
            column, direction = raw.split(":", 1)
        column = column.strip()
        direction = direction.strip().lower()
        if not column:
            raise AnalysisError(f"Orden inválido '{raw}'. Usa COLUMNA o COLUMNA:asc/desc.")
        if column.isdigit():
            idx = int(column)
            if 1 <= idx <= len(fieldnames):
                column = fieldnames[idx - 1]
            else:
                raise AnalysisError(f"Columna inválida para ordenar: {column}")
        if column not in fieldnames:
            raise AnalysisError(f"Columna inválida para ordenar: {column}")
        if direction not in {"asc", "desc"}:
            raise AnalysisError(
                f"Dirección inválida '{direction}' en '{raw}'. Usa 'asc' o 'desc'."
            )
        sort_keys.append(SortKey(column=column, descending=direction == "desc"))
    return sort_keys


def parse_percentiles(raw: Optional[str]) -> List[float]:
    if not raw:
        return []
    values = []
    for item in raw.split(","):
        cleaned = item.strip()
        if not cleaned:
            continue
        try:
            value = float(cleaned)
        except ValueError as exc:
            raise AnalysisError(f"Percentil inválido '{cleaned}'.") from exc
        if not 0 <= value <= 100:
            raise AnalysisError(f"Percentil fuera de rango: {value}. Usa 0-100.")
        values.append(value)
    return sorted(set(values))


def to_float(value: str) -> Optional[float]:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def sort_value(raw: str) -> Tuple[int, object]:
    value = raw.strip()
    if not value:
        return (1, "")
    try:
        return (0, float(value))
    except ValueError:
        return (1, value.casefold())


def sort_rows(rows: List[Dict[str, str]], sort_keys: Sequence[SortKey]) -> None:
    for sort_key in reversed(sort_keys):
        def key(row: Dict[str, str]) -> Tuple[bool, Tuple[int, object]]:
            raw = row.get(sort_key.column, "")
            is_empty = not raw.strip()
            parsed = sort_value(raw)
            if sort_key.descending:
                return (not is_empty, parsed)
            return (is_empty, parsed)

        rows.sort(key=key, reverse=sort_key.descending)


def build_max_fieldnames(fieldnames: Sequence[str], enabled: bool) -> List[str]:
    if not enabled:
        return list(fieldnames)
    remove_columns = {
        column for columns in MAX_SCORE_GROUPS.values() for column in columns
    }
    remove_columns.update(module for _, module in SCIENCE_MODULE_COLUMNS)
    updated = [name for name in fieldnames if name not in remove_columns]
    updated.extend(MAX_SCORE_GROUPS.keys())
    updated.append(SCIENCE_MAX_MODULE_COLUMN)
    return updated


def apply_max_scores(row: Dict[str, str], enabled: bool) -> Dict[str, str]:
    if not enabled:
        return row
    remove_columns = {
        column for columns in MAX_SCORE_GROUPS.values() for column in columns
    }
    remove_columns.update(module for _, module in SCIENCE_MODULE_COLUMNS)
    transformed = {key: value for key, value in row.items() if key not in remove_columns}
    def format_score(value: Optional[float]) -> str:
        if value is None:
            return ""
        if value.is_integer():
            return str(int(value))
        return str(value)

    for max_name, columns in MAX_SCORE_GROUPS.items():
        best_value: Optional[float] = None
        for column in columns:
            value = to_float(row.get(column, "") or "")
            if value is None:
                continue
            if best_value is None or value > best_value:
                best_value = value
        transformed[max_name] = format_score(best_value)
    module_value = ""
    best_science: Optional[float] = None
    for score_column, module_column in SCIENCE_MODULE_COLUMNS:
        value = to_float(row.get(score_column, "") or "")
        if value is None:
            continue
        if best_science is None or value > best_science:
            best_science = value
            module_value = row.get(module_column, "") or ""
    transformed[SCIENCE_MAX_MODULE_COLUMN] = module_value
    return transformed


def build_active_fieldnames(
    fieldnames: Sequence[str],
    use_max_scores: bool,
    weighting: Optional[WeightingConfig],
) -> List[str]:
    updated = build_max_fieldnames(fieldnames, use_max_scores)
    if weighting and weighting.enabled and WEIGHTED_SCORE_COLUMN not in updated:
        updated.append(WEIGHTED_SCORE_COLUMN)
    return updated


def subject_columns_for_weighting(use_max_scores: bool) -> Dict[str, str]:
    return {
        "nem": "PTJE_NEM",
        "ranking": "PTJE_RANKING",
        "clec": "CLEC_MAX" if use_max_scores else "CLEC_REG_ACTUAL",
        "mate1": "MATE1_MAX" if use_max_scores else "MATE1_REG_ACTUAL",
        "mate2": "MATE2_MAX" if use_max_scores else "MATE2_REG_ACTUAL",
        "historia": "HCSOC_MAX" if use_max_scores else "HCSOC_REG_ACTUAL",
        "ciencias": "CIEN_MAX" if use_max_scores else "CIEN_REG_ACTUAL",
    }


def compute_weighted_score(
    row: Dict[str, str],
    weighting: WeightingConfig,
    subject_columns: Dict[str, str],
) -> Tuple[Optional[float], bool]:
    if not weighting.enabled:
        return None, True
    total = 0.0
    used = False
    for subject in ["nem", "ranking", "clec", "mate1", "mate2"]:
        weight = weighting.weights.get(subject, 0.0)
        if weight <= 0:
            continue
        value = to_float(row.get(subject_columns.get(subject, ""), "") or "")
        if value is None or value <= 0:
            return None, False
        total += value * (weight / 100)
        used = True

    history_weight = weighting.weights.get("historia", 0.0)
    science_weight = weighting.weights.get("ciencias", 0.0)
    if weighting.history_science_mode == "ciencias":
        history_weight = 0.0
    elif weighting.history_science_mode == "historia":
        science_weight = 0.0

    if weighting.history_science_mode == "mejor":
        history_value = None
        science_value = None
        if history_weight > 0:
            history_value = to_float(
                row.get(subject_columns.get("historia", ""), "") or ""
            )
            if history_value is None or history_value <= 0:
                history_value = None
        if science_weight > 0:
            science_value = to_float(
                row.get(subject_columns.get("ciencias", ""), "") or ""
            )
            if science_value is None or science_value <= 0:
                science_value = None
        candidates = []
        if history_weight > 0 and history_value is not None:
            candidates.append(history_value * (history_weight / 100))
        if science_weight > 0 and science_value is not None:
            candidates.append(science_value * (science_weight / 100))
        if history_weight > 0 or science_weight > 0:
            if not candidates:
                return None, False
            total += max(candidates)
            used = True
    else:
        for subject, weight in [("historia", history_weight), ("ciencias", science_weight)]:
            if weight <= 0:
                continue
            value = to_float(row.get(subject_columns.get(subject, ""), "") or "")
            if value is None or value <= 0:
                return None, False
            total += value * (weight / 100)
            used = True

    return (total if used else None), True


def apply_weighted_score(
    row: Dict[str, str],
    weighting: Optional[WeightingConfig],
    subject_columns: Dict[str, str],
) -> Tuple[Dict[str, str], bool]:
    if not weighting or not weighting.enabled:
        return row, True
    weighted_value, eligible = compute_weighted_score(row, weighting, subject_columns)
    transformed = dict(row)
    if weighted_value is None:
        transformed[WEIGHTED_SCORE_COLUMN] = ""
    elif weighted_value.is_integer():
        transformed[WEIGHTED_SCORE_COLUMN] = str(int(weighted_value))
    else:
        transformed[WEIGHTED_SCORE_COLUMN] = f"{weighted_value:.2f}"
    return transformed, eligible


def prune_filters_for_fieldnames(
    column_filters: Dict[str, set[str]],
    min_scores: Dict[str, float],
    max_scores: Dict[str, float],
    sort_by: List[str],
    fieldnames: Sequence[str],
) -> Tuple[Dict[str, set[str]], Dict[str, float], Dict[str, float], List[str]]:
    valid_fields = set(fieldnames)
    column_filters = {
        column: values for column, values in column_filters.items() if column in valid_fields
    }
    min_scores = {column: value for column, value in min_scores.items() if column in valid_fields}
    max_scores = {column: value for column, value in max_scores.items() if column in valid_fields}
    filtered_sort_by = []
    for entry in sort_by:
        column = entry.split(":", 1)[0].strip()
        if column in valid_fields:
            filtered_sort_by.append(entry)
    return column_filters, min_scores, max_scores, filtered_sort_by

def load_cod_ens(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    dialect = sniff_dialect(path)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, dialect=dialect)
        mapping: Dict[str, str] = {}
        for row in reader:
            codigo = normalize_code(row.get("Código"))
            descripcion = normalize_code(row.get("Descripción"))
            if codigo:
                mapping[codigo] = descripcion
        return mapping


def load_comunas_regiones(path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    if not path.exists():
        return {}, {}
    dialect = sniff_dialect(path)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, dialect=dialect)
        regiones: Dict[str, str] = {}
        comunas: Dict[str, str] = {}
        for row in reader:
            region = normalize_code(row.get("COD REG."))
            region_nombre = normalize_code(row.get("REGION NOMBRE"))
            comuna = normalize_code(row.get("COD.COMUNA"))
            comuna_nombre = normalize_code(row.get("COM NOMBRE"))
            if region and region_nombre:
                regiones[region] = region_nombre
            if comuna and comuna_nombre:
                comunas[comuna] = comuna_nombre
        return regiones, comunas


def load_codebook(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    dialect = sniff_dialect(path)
    mappings: Dict[str, Dict[str, str]] = {}
    current_variable = ""
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, dialect=dialect)
        for row in reader:
            variable = normalize_code(row.get("Variable"))
            detail = normalize_code(row.get("Detalle"))
            if variable:
                current_variable = variable
                continue
            if not current_variable or not detail:
                continue
            code = ""
            description = ""
            if "." in detail:
                code, description = detail.split(".", 1)
            elif ":" in detail:
                code, description = detail.split(":", 1)
            if not code or not description:
                continue
            code = code.strip()
            description = description.strip()
            if not code or not description:
                continue
            mappings.setdefault(current_variable, {})[code] = description
    return mappings


def load_rbd_colegios(path: Path) -> Dict[str, RbdInfo]:
    if not path.exists():
        return {}
    dialect = sniff_dialect(path)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, dialect=dialect)
        mapping: Dict[str, RbdInfo] = {}
        for row in reader:
            rbd = normalize_code(row.get("RBD"))
            name = normalize_code(row.get("NOM_RBD"))
            comuna = normalize_code(row.get("NOM_COM_RB"))
            region = normalize_code(row.get("NOM_REG_RB"))
            if not rbd:
                continue
            mapping[rbd] = RbdInfo(name=name, comuna=comuna, region=region)
        return mapping


def build_code_maps(
    cod_ens_path: Path,
    comunas_path: Path,
    codebook_path: Path,
    rbd_colegios_path: Path,
) -> CodeMaps:
    cod_ens = load_cod_ens(cod_ens_path)
    regiones, comunas = load_comunas_regiones(comunas_path)
    value_labels = load_codebook(codebook_path)
    rbd_info = load_rbd_colegios(rbd_colegios_path)
    return CodeMaps(
        cod_ens=cod_ens,
        regiones=regiones,
        comunas=comunas,
        value_labels=value_labels,
        rbd_info=rbd_info,
    )


def row_matches(
    row: Dict[str, str],
    column_filters: Dict[str, Optional[set[str]]],
    min_scores: Sequence[ScoreFilter],
    max_scores: Sequence[ScoreFilter],
) -> bool:
    for column, allowed in column_filters.items():
        if not allowed:
            continue
        if normalize_code(row.get(column)) not in allowed:
            return False
    for score in min_scores:
        value = to_float(row.get(score.column, ""))
        if value is None or value < score.threshold:
            return False
    for score in max_scores:
        value = to_float(row.get(score.column, ""))
        if value is None or value > score.threshold:
            return False
    return True


def value_description(column: str, value: str, maps: CodeMaps) -> str:
    if column in maps.value_labels and value in maps.value_labels[column]:
        return maps.value_labels[column][value]
    if column == "COD_ENS" and value in maps.cod_ens:
        return maps.cod_ens[value]
    if column == "CODIGO_REGION" and value in maps.regiones:
        return maps.regiones[value]
    if column == "CODIGO_COMUNA" and value in maps.comunas:
        return maps.comunas[value]
    if column == "RBD" and value in maps.rbd_info:
        info = maps.rbd_info[value]
        parts = [part for part in [info.name, info.comuna, info.region] if part]
        return " - ".join(parts)
    return ""


def label_value(column: str, value: str, maps: CodeMaps) -> str:
    description = value_description(column, value, maps)
    if description:
        return f"{value} - {description}"
    return value


def display_value(
    column: str,
    value: str,
    maps: CodeMaps,
    display_labels: bool,
) -> str:
    normalized = normalize_code(value)
    if display_labels and normalized:
        return label_value(column, normalized, maps)
    return normalized


def add_label_columns(row: Dict[str, str], maps: CodeMaps) -> Dict[str, str]:
    enriched = dict(row)
    cod_ens = normalize_code(row.get("COD_ENS"))
    region = normalize_code(row.get("CODIGO_REGION"))
    comuna = normalize_code(row.get("CODIGO_COMUNA"))
    if cod_ens:
        enriched["COD_ENS_DESC"] = maps.cod_ens.get(cod_ens, "")
    if region:
        enriched["REGION_NOMBRE"] = maps.regiones.get(region, "")
    if comuna:
        enriched["COMUNA_NOMBRE"] = maps.comunas.get(comuna, "")
    return enriched


def build_numbered_headers(headers: Sequence[str]) -> Tuple[List[str], List[Tuple[str, str]]]:
    numbered = [str(idx) for idx, _ in enumerate(headers, start=1)]
    mapping = [(str(idx), header) for idx, header in enumerate(headers, start=1)]
    return numbered, mapping


def print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [len(header) for header in headers]
    normalized_rows = []
    for row in rows:
        normalized = [str(cell) for cell in row]
        normalized_rows.append(normalized)
        for idx, cell in enumerate(normalized):
            widths[idx] = max(widths[idx], len(cell))

    header_line = " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))
    separator = "-+-".join("-" * width for width in widths)
    print(header_line)
    print(separator)
    for row in normalized_rows:
        print(" | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)))


def print_column_index(headers: Sequence[str]) -> None:
    print("\nÍndice de columnas:")
    for idx, header in enumerate(headers, start=1):
        print(f"- {idx}: {header}")


def print_counts(counts: Dict[str, Counter]) -> None:
    for column, counter in counts.items():
        print(f"\nRecuento por {column}:")
        rows = []
        for key, value in counter.most_common():
            label = key if key else "(vacío)"
            rows.append([label, str(value)])
        if not rows:
            print("(sin datos)")
            continue
        print_table(["Valor", "Cantidad"], rows)


def percentile(sorted_values: List[float], percent: float) -> Optional[float]:
    if not sorted_values:
        return None
    if percent <= 0:
        return sorted_values[0]
    if percent >= 100:
        return sorted_values[-1]
    position = (len(sorted_values) - 1) * (percent / 100)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return lower_value + (upper_value - lower_value) * fraction


def record_stat_value(
    column: str,
    value: Optional[float],
    stats_values: Dict[str, List[float]],
    totals: Dict[str, int],
    zero_counts: Dict[str, int],
) -> None:
    if value is None:
        return
    totals[column] += 1
    if value == 0:
        zero_counts[column] += 1
        return
    stats_values[column].append(value)


def record_stat_value_by_rbd(
    rbd: str,
    column: str,
    value: Optional[float],
    stats_by_rbd: Dict[str, Dict[str, List[float]]],
    totals_by_rbd: Dict[str, Dict[str, int]],
    zero_counts_by_rbd: Dict[str, Dict[str, int]],
) -> None:
    if value is None:
        return
    totals_by_rbd[rbd][column] += 1
    if value == 0:
        zero_counts_by_rbd[rbd][column] += 1
        return
    stats_by_rbd[rbd][column].append(value)


def print_zero_exclusions(
    totals: Dict[str, int],
    zero_counts: Dict[str, int],
) -> None:
    if not totals:
        return
    print("\nValores 0 excluidos de estadísticas:")
    headers = ["Columna", "Total", "Ceros excluidos", "Considerados"]
    rows = []
    for column in totals:
        total = totals[column]
        zeros = zero_counts.get(column, 0)
        considered = max(total - zeros, 0)
        rows.append([column, str(total), str(zeros), str(considered)])
    print_table(headers, rows)


def print_statistics(
    stats: Dict[str, List[float]],
    percentiles: Sequence[float],
    totals: Optional[Dict[str, int]] = None,
    zero_counts: Optional[Dict[str, int]] = None,
) -> None:
    if not stats:
        return
    if totals is not None and zero_counts is not None:
        print_zero_exclusions(totals, zero_counts)
    print("\nEstadísticas de columnas (filtradas):")
    base_headers = ["Columna", "N", "Promedio", "Mediana", "Desv. Est."]
    percentile_headers = [f"P{int(p)}" if p.is_integer() else f"P{p}" for p in percentiles]
    headers = base_headers + percentile_headers
    rows = []
    for column, values in stats.items():
        cleaned = [value for value in values if value is not None]
        if not cleaned:
            rows.append([column, "0", "-", "-", "-"] + ["-" for _ in percentiles])
            continue
        ordered = sorted(cleaned)
        avg = mean(ordered)
        med = median(ordered)
        deviation = pstdev(ordered) if len(ordered) > 1 else 0.0
        row = [
            column,
            str(len(ordered)),
            f"{avg:.2f}",
            f"{med:.2f}",
            f"{deviation:.2f}",
        ]
        for perc in percentiles:
            value = percentile(ordered, perc)
            row.append(f"{value:.2f}" if value is not None else "-")
        rows.append(row)
    print_table(headers, rows)


def print_zero_exclusions_by_rbd(
    totals_by_rbd: Dict[str, Dict[str, int]],
    zero_counts_by_rbd: Dict[str, Dict[str, int]],
) -> None:
    if not totals_by_rbd:
        return
    print("\nValores 0 excluidos por RBD:")
    headers = ["RBD", "Columna", "Total", "Ceros excluidos", "Considerados"]
    rows = []
    for rbd in sorted(totals_by_rbd.keys()):
        for column in sorted(totals_by_rbd[rbd].keys()):
            total = totals_by_rbd[rbd][column]
            zeros = zero_counts_by_rbd.get(rbd, {}).get(column, 0)
            considered = max(total - zeros, 0)
            rows.append([rbd, column, str(total), str(zeros), str(considered)])
    print_table(headers, rows)


def print_rbd_statistics(
    stats_by_rbd: Dict[str, Dict[str, List[float]]],
    percentiles: Sequence[float],
    totals_by_rbd: Optional[Dict[str, Dict[str, int]]] = None,
    zero_counts_by_rbd: Optional[Dict[str, Dict[str, int]]] = None,
) -> None:
    if not stats_by_rbd:
        return
    if totals_by_rbd is not None and zero_counts_by_rbd is not None:
        print_zero_exclusions_by_rbd(totals_by_rbd, zero_counts_by_rbd)
    print("\nEstadísticas por RBD (filtradas):")
    base_headers = ["RBD", "Columna", "N", "Promedio", "Mediana", "Desv. Est."]
    percentile_headers = [f"P{int(p)}" if p.is_integer() else f"P{p}" for p in percentiles]
    headers = base_headers + percentile_headers
    rows = []
    for rbd in sorted(stats_by_rbd.keys()):
        for column in sorted(stats_by_rbd[rbd].keys()):
            values = stats_by_rbd[rbd][column]
            cleaned = [value for value in values if value is not None]
            if not cleaned:
                rows.append([rbd, column, "0", "-", "-", "-"] + ["-" for _ in percentiles])
                continue
            ordered = sorted(cleaned)
            avg = mean(ordered)
            med = median(ordered)
            deviation = pstdev(ordered) if len(ordered) > 1 else 0.0
            row = [
                rbd,
                column,
                str(len(ordered)),
                f"{avg:.2f}",
                f"{med:.2f}",
                f"{deviation:.2f}",
            ]
            for perc in percentiles:
                value = percentile(ordered, perc)
                row.append(f"{value:.2f}" if value is not None else "-")
            rows.append(row)
    print_table(headers, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analiza archivos de rendición con filtros y recuentos.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Inicia un modo interactivo para elegir filtros y opciones.",
    )
    parser.add_argument("--data", default=DEFAULT_DATA, help="Ruta al CSV principal.")
    parser.add_argument(
        "--cod-ens-csv",
        default=DEFAULT_COD_ENS,
        help="CSV auxiliar con descripción de COD_ENS.",
    )
    parser.add_argument(
        "--comunas-csv",
        default=DEFAULT_COMUNAS,
        help="CSV auxiliar con regiones y comunas.",
    )
    parser.add_argument(
        "--codebook-csv",
        default=DEFAULT_CODEBOOK,
        help="CSV con códigos y descripciones de variables.",
    )
    parser.add_argument(
        "--rbd-colegios-csv",
        default=DEFAULT_RBD_COLEGIOS,
        help="CSV con equivalencias de RBD a colegio/comuna/región.",
    )
    parser.add_argument("--rbd", help="Lista de RBD separados por coma.")
    parser.add_argument("--situacion-egreso", help="Lista de situación de egreso.")
    parser.add_argument("--cod-ens", help="Lista de códigos de enseñanza.")
    parser.add_argument("--region", help="Lista de códigos de región.")
    parser.add_argument("--comuna", help="Lista de códigos de comuna.")
    parser.add_argument("--grupo-dependencia", help="Lista de grupos de dependencia.")
    parser.add_argument("--rama-educacional", help="Lista de rama educacional.")
    parser.add_argument(
        "--min-score",
        action="append",
        default=[],
        help="Filtro mínimo de puntaje. Formato: COLUMNA=NUMERO",
    )
    parser.add_argument(
        "--max-score",
        action="append",
        default=[],
        help="Filtro máximo de puntaje. Formato: COLUMNA=NUMERO",
    )
    parser.add_argument(
        "--count-by",
        action="append",
        default=[],
        help="Columna por la que se quiere agrupar (se puede repetir).",
    )
    parser.add_argument(
        "--stats",
        action="append",
        default=[],
        help="Columna numérica para calcular estadísticas (se puede repetir).",
    )
    parser.add_argument(
        "--percentiles",
        default="25,50,75",
        help="Percentiles a calcular para --stats (separados por coma).",
    )
    parser.add_argument(
        "--stats-by-rbd",
        action="store_true",
        help="Calcula estadísticas por cada RBD indicado en --rbd.",
    )
    parser.add_argument(
        "--sort-by",
        action="append",
        default=[],
        help="Ordena los datos por columna. Formato: COLUMNA o COLUMNA:asc/desc",
    )
    parser.add_argument("--output-csv", help="Ruta de salida para filas filtradas.")
    parser.add_argument(
        "--add-labels",
        action="store_true",
        help="Agrega columnas descriptivas usando los CSV auxiliares.",
    )
    parser.add_argument(
        "--list-columns",
        action="store_true",
        help="Imprime las columnas disponibles y termina.",
    )
    return parser.parse_args()


def prompt_value(label: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default else ""
    return input(f"{label}{suffix}: ").strip() or (default or "")


def prompt_choice(label: str, options: List[str], default: Optional[int] = None) -> int:
    print(label)
    for idx, option in enumerate(options, start=1):
        print(f"{idx}. {option}")
    while True:
        raw = input(
            "Selecciona una opción"
            + (f" [{default}]" if default else "")
            + ": "
        ).strip()
        if not raw and default is not None:
            return default
        if raw.isdigit():
            selection = int(raw)
            if 1 <= selection <= len(options):
                return selection
        print("Selección inválida. Intenta nuevamente.")


def prompt_data_file(default_path: str) -> str:
    candidates = discover_rendicion_csvs(Path.cwd())
    if not candidates:
        return prompt_value("Ruta del CSV principal", default_path)
    display = [str(path.relative_to(Path.cwd())) for path in candidates]
    default_index = None
    default_path_obj = Path(default_path)
    if default_path_obj in candidates:
        default_index = candidates.index(default_path_obj) + 1
    display.append("Ingresar otra ruta manualmente")
    selection = prompt_choice(
        "Archivos de rendición disponibles:",
        display,
        default=default_index,
    )
    if selection == len(display):
        return prompt_value("Ruta del CSV principal", default_path)
    return str(candidates[selection - 1])


def prompt_list_filter(label: str, current: Optional[str]) -> Optional[str]:
    if current:
        return current
    raw = input(f"{label} (valores separados por coma, Enter para omitir): ").strip()
    return raw or None


def prompt_yes_no(label: str, default: bool = False) -> bool:
    default_label = "s" if default else "n"
    while True:
        raw = input(f"{label} (s/n) [{default_label}]: ").strip().lower()
        if not raw:
            return default
        if raw in {"s", "si", "sí"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Respuesta inválida. Usa 's' o 'n'.")


def prompt_score_filters(kind: str, existing: List[str], fieldnames: List[str]) -> List[str]:
    if existing:
        return existing
    filters: List[str] = []
    print(f"\nIngresa filtros {kind} en formato COLUMNA=NUMERO. Enter para terminar.")
    print("Ejemplo: CLEC_REG_ACTUAL=500")
    while True:
        raw = input("> ").strip()
        if not raw:
            break
        if "=" not in raw:
            print("Formato inválido. Usa COLUMNA=NUMERO.")
            continue
        column, _ = raw.split("=", 1)
        if column.strip() not in fieldnames:
            print(f"Columna desconocida: {column.strip()}")
            continue
        filters.append(raw)
    return filters


def prompt_count_by(existing: List[str], fieldnames: List[str]) -> List[str]:
    if existing:
        return existing
    raw = input(
        "Columnas para agrupar (separadas por coma, Enter para omitir): "
    ).strip()
    if not raw:
        return []
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [item for item in requested if item not in fieldnames]
    if invalid:
        print(f"Columnas inválidas ignoradas: {', '.join(invalid)}")
    return [item for item in requested if item in fieldnames]


def prompt_stats_columns(existing: List[str], fieldnames: List[str]) -> List[str]:
    if existing:
        return existing
    print_column_index(fieldnames)
    raw = input(
        "Columnas numéricas para estadísticas (separadas por coma, "
        "puedes usar números, Enter para omitir): "
    ).strip()
    if not raw:
        return []
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    selected: List[str] = []
    invalid: List[str] = []
    for item in requested:
        if item.isdigit():
            idx = int(item)
            if 1 <= idx <= len(fieldnames):
                selected.append(fieldnames[idx - 1])
            else:
                invalid.append(item)
        elif item in fieldnames:
            selected.append(item)
        else:
            invalid.append(item)
    if invalid:
        print(f"Columnas inválidas ignoradas: {', '.join(invalid)}")
    unique_selected = []
    for column in selected:
        if column not in unique_selected:
            unique_selected.append(column)
    return unique_selected


def prompt_sort_by(existing: List[str], fieldnames: List[str]) -> List[str]:
    if existing:
        return existing
    raw = input(
        "Columnas para ordenar (separadas por coma, usa :asc o :desc, Enter para omitir): "
    ).strip()
    if not raw:
        return []
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    cleaned: List[str] = []
    invalid: List[str] = []
    for item in requested:
        column_part, *direction_part = item.split(":", 1)
        column_part = column_part.strip()
        direction = f":{direction_part[0].strip()}" if direction_part else ""
        if column_part.isdigit():
            idx = int(column_part)
            if 1 <= idx <= len(fieldnames):
                cleaned.append(f"{fieldnames[idx - 1]}{direction}")
            else:
                invalid.append(item)
        elif column_part in fieldnames:
            cleaned.append(item)
        else:
            invalid.append(item)
    if invalid:
        print(f"Orden inválido ignorado: {', '.join(invalid)}")
    return cleaned


def summarize_filters(
    column_filters: Dict[str, set[str]],
    min_scores: Dict[str, float],
    max_scores: Dict[str, float],
) -> None:
    print("\nFiltros actuales:")
    if not column_filters and not min_scores and not max_scores:
        print("- (sin filtros)")
        return
    for column, values in sorted(column_filters.items()):
        if values:
            joined = ", ".join(sorted(values))
            print(f"- {column} en [{joined}]")
    for column, threshold in sorted(min_scores.items()):
        print(f"- {column} >= {threshold}")
    for column, threshold in sorted(max_scores.items()):
        print(f"- {column} <= {threshold}")


def summarize_weighting(weighting: Optional[WeightingConfig]) -> None:
    if not weighting or not weighting.enabled:
        print("- Ponderación: (no configurada)")
        return
    parts = []
    for key, label in WEIGHTED_SUBJECTS:
        weight = weighting.weights.get(key, 0.0)
        if weight:
            parts.append(f"{label} {weight}%")
    mode_label = {
        "mejor": "Mejor entre historia y ciencias",
        "ciencias": "Solo ciencias",
        "historia": "Solo historia",
    }.get(weighting.history_science_mode, weighting.history_science_mode)
    print("- Ponderación: " + (", ".join(parts) if parts else "(sin pesos)"))
    print(f"- Hist/Ciencias: {mode_label}")


def prompt_weight_value(label: str, current: float) -> float:
    while True:
        raw = input(f"{label} [{current}%]: ").strip()
        if not raw:
            return current
        try:
            return float(raw)
        except ValueError:
            print("Número inválido. Usa un porcentaje (ej: 20).")


def prompt_weighting(
    weighting: Optional[WeightingConfig],
    use_max_scores: bool,
) -> WeightingConfig:
    current_weights = (
        dict(weighting.weights)
        if weighting
        else {key: 0.0 for key, _ in WEIGHTED_SUBJECTS}
    )
    print("\nConfigurar ponderación (porcentajes):")
    for key, label in WEIGHTED_SUBJECTS:
        current_weights[key] = prompt_weight_value(label, current_weights.get(key, 0.0))
    total_weight = sum(current_weights.values())
    if total_weight and total_weight != 100:
        print(f"Advertencia: la suma de ponderaciones es {total_weight}%.")

    mode_options = [
        "Mejor ponderado entre Historia/Ciencias",
        "Solo Ciencias",
        "Solo Historia",
    ]
    mode_default = 1
    if weighting:
        mode_default = {"mejor": 1, "ciencias": 2, "historia": 3}.get(
            weighting.history_science_mode, 1
        )
    selection = prompt_choice("Modo Historia/Ciencias:", mode_options, default=mode_default)
    history_science_mode = {1: "mejor", 2: "ciencias", 3: "historia"}[selection]
    subject_columns = subject_columns_for_weighting(use_max_scores)
    print("\nColumnas usadas para la ponderación:")
    for key, label in WEIGHTED_SUBJECTS:
        column = subject_columns.get(key, "(no definida)")
        print(f"- {label}: {column}")
    return WeightingConfig(
        weights=current_weights,
        history_science_mode=history_science_mode,
        enabled=True,
    )


def build_filter_index(
    column_filters: Dict[str, set[str]],
    min_scores: Dict[str, float],
    max_scores: Dict[str, float],
) -> List[Tuple[str, str]]:
    index: List[Tuple[str, str]] = []
    for column in sorted(column_filters):
        index.append(("igual", column))
    for column in sorted(min_scores):
        index.append(("min", column))
    for column in sorted(max_scores):
        index.append(("max", column))
    return index


def prompt_column(fieldnames: List[str]) -> str:
    selection = prompt_choice("Selecciona una columna:", fieldnames)
    return fieldnames[selection - 1]


def collect_column_values(
    column: str,
    data_path: Path,
    cache: Dict[str, Counter],
) -> Counter:
    if column in cache:
        return cache[column]
    _, rows = read_csv_rows(data_path)
    counts: Counter = Counter()
    for row in rows:
        value = normalize_code(row.get(column))
        counts[value] += 1
    cache[column] = counts
    return counts


def print_column_value_help(
    column: str,
    data_path: Path,
    maps: CodeMaps,
    cache: Dict[str, Counter],
) -> None:
    counts = collect_column_values(column, data_path, cache)
    if not counts:
        print(f"\nNo se encontraron valores para {column}.")
        return
    rows = []
    for value in sorted(counts.keys()):
        description = value_description(column, value, maps)
        rows.append(
            [
                value or "(vacío)",
                description or "(sin descripción)",
                str(counts[value]),
            ]
        )
    print(f"\nValores disponibles para {column}:")
    print_table(["Valor", "Descripción", "Cantidad"], rows)


def normalize_text(value: str) -> str:
    return value.casefold().strip()


def rbd_info_to_row(rbd: str, info: RbdInfo) -> List[str]:
    return [
        rbd,
        info.name or "(sin nombre)",
        info.comuna or "(sin comuna)",
        info.region or "(sin región)",
    ]


def search_rbd_info(maps: CodeMaps, query: str) -> List[Tuple[str, RbdInfo]]:
    if not query:
        return []
    needle = normalize_text(query)
    matches = []
    for rbd, info in maps.rbd_info.items():
        haystack = " ".join([info.name, info.comuna, info.region]).casefold()
        if needle in haystack:
            matches.append((rbd, info))
    return sorted(matches, key=lambda item: item[0])


def prompt_rbd_lookup(maps: CodeMaps) -> None:
    if not maps.rbd_info:
        print("No se encontró el archivo de RBD colegios o está vacío.")
        return
    options = [
        "Buscar por RBD",
        "Buscar por nombre/comuna/región",
        "Volver",
    ]
    selection = prompt_choice("Transformar RBD ↔ colegio:", options, default=1)
    if selection == 1:
        raw = input("Ingresa RBD (separados por coma): ").strip()
        if not raw:
            return
        rows = []
        for rbd in [item.strip() for item in raw.split(",") if item.strip()]:
            info = maps.rbd_info.get(rbd)
            if info:
                rows.append(rbd_info_to_row(rbd, info))
            else:
                rows.append([rbd, "(no encontrado)", "-", "-"])
        print_table(["RBD", "Colegio", "Comuna", "Región"], rows)
    elif selection == 2:
        query = input("Ingresa nombre/comuna/región: ").strip()
        if not query:
            return
        matches = search_rbd_info(maps, query)
        if not matches:
            print("No se encontraron colegios con ese criterio.")
            return
        rows = [rbd_info_to_row(rbd, info) for rbd, info in matches]
        print_table(["RBD", "Colegio", "Comuna", "Región"], rows)
def prompt_column_value(
    column: str,
    data_path: Path,
    maps: CodeMaps,
    cache: Dict[str, Counter],
) -> set[str]:
    if column in maps.value_labels or column in {"COD_ENS", "CODIGO_REGION", "CODIGO_COMUNA"}:
        print_column_value_help(column, data_path, maps, cache)
    raw = input(
        f"Valores para {column} (separados por coma, Enter para cancelar): "
    ).strip()
    if not raw:
        return set()
    return {value.strip() for value in raw.split(",") if value.strip()}


def prompt_threshold(column: str) -> Optional[float]:
    raw = input(f"Valor numérico para {column} (Enter para cancelar): ").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        print("Número inválido.")
        return None


def prompt_ranking_metric() -> str:
    options = [
        "Promedio",
        "Mediana",
        "Moda",
        "Desv. Est.",
    ]
    selection = prompt_choice(
        "Selecciona la métrica con la que se ordenará el ranking:", options, default=1
    )
    return {1: "mean", 2: "median", 3: "mode", 4: "stdev"}[selection]


def prompt_ranking_value_columns(fieldnames: List[str]) -> List[str]:
    options = [
        "Analizar una sola columna",
        "Promedio entre varias columnas",
    ]
    selection = prompt_choice(
        "¿Qué valores deseas comparar dentro de cada grupo?", options, default=1
    )
    if selection == 1:
        return [prompt_column(fieldnames)]
    columns = prompt_stats_columns([], fieldnames)
    if len(columns) < 2:
        print("Debes seleccionar al menos dos columnas para promediar.")
        return []
    return columns


def prompt_ranking_limit() -> Optional[int]:
    raw = input("¿Cuántos resultados mostrar? (0 para todos) [20]: ").strip()
    if not raw:
        return 20
    if raw == "0":
        return None
    if raw.isdigit():
        return int(raw)
    print("Número inválido, se mostrarán todos los resultados.")
    return None


def evaluate_row_value(
    row: Dict[str, str],
    columns: List[str],
) -> Tuple[Optional[float], bool]:
    values: List[float] = []
    for column in columns:
        value = to_float(row.get(column, ""))
        if value is None or value <= 0:
            return None, False
        values.append(value)
    if not values:
        return None, False
    return mean(values), True


def calculate_mode(values: List[float]) -> Optional[float]:
    if not values:
        return None
    counts = Counter(values)
    highest = max(counts.values())
    candidates = [value for value, count in counts.items() if count == highest]
    return min(candidates)


def metric_label(metric: str, columns: List[str]) -> str:
    column_label = columns[0] if len(columns) == 1 else "Promedio columnas"
    labels = {
        "mean": f"Promedio {column_label}",
        "median": f"Mediana {column_label}",
        "mode": f"Moda {column_label}",
        "stdev": f"Desv. Est. {column_label}",
    }
    return labels.get(metric, column_label)


def summarize_ranking(config: Optional[RankingConfig]) -> None:
    if not config:
        print("- Ranking: (no configurado)")
        return
    value_label = (
        config.value_columns[0]
        if len(config.value_columns) == 1
        else "Promedio columnas"
    )
    order_label = "desc" if config.order_desc else "asc"
    limit_label = "sin límite" if config.limit is None else str(config.limit)
    print(
        "- Ranking: "
        f"grupo {config.grouping_column}, "
        f"comparación {value_label}, "
        f"métrica {metric_label(config.metric, config.value_columns)}, "
        f"orden {order_label}, "
        f"tope {limit_label}"
    )


def prompt_ranking_config(fieldnames: List[str]) -> Optional[RankingConfig]:
    print("\nConfiguración de ranking")
    print(
        "Definirás un ranking que agrupa registros por una columna (por ejemplo RBD "
        "o CODIGO_REGION) y compara puntajes dentro de cada grupo."
    )
    print(
        "Luego elegirás qué puntaje(s) comparar y la métrica con la que se ordenará."
    )
    grouping_column = prompt_column(fieldnames)
    value_columns = prompt_ranking_value_columns(fieldnames)
    if not value_columns:
        return None
    metric = prompt_ranking_metric()
    order_desc = prompt_yes_no("¿Ordenar de mayor a menor?", default=True)
    limit = prompt_ranking_limit()
    return RankingConfig(
        grouping_column=grouping_column,
        value_columns=value_columns,
        metric=metric,
        order_desc=order_desc,
        limit=limit,
    )


def update_ranking_config(
    config: Optional[RankingConfig], fieldnames: List[str]
) -> Optional[RankingConfig]:
    if not config:
        return prompt_ranking_config(fieldnames)
    options = [
        "Cambiar grupo de comparación",
        "Cambiar valores de comparación",
        "Cambiar métrica",
        "Cambiar orden/límite",
        "Reconfigurar todo",
        "Volver",
    ]
    selection = prompt_choice("¿Qué deseas ajustar del ranking?", options, default=1)
    if selection == 1:
        config.grouping_column = prompt_column(fieldnames)
    elif selection == 2:
        value_columns = prompt_ranking_value_columns(fieldnames)
        if value_columns:
            config.value_columns = value_columns
    elif selection == 3:
        config.metric = prompt_ranking_metric()
    elif selection == 4:
        config.order_desc = prompt_yes_no("¿Ordenar de mayor a menor?", default=True)
        config.limit = prompt_ranking_limit()
    elif selection == 5:
        return prompt_ranking_config(fieldnames)
    return config


def validate_ranking_config(
    config: Optional[RankingConfig], fieldnames: List[str]
) -> Optional[RankingConfig]:
    if not config:
        return None
    missing = []
    if config.grouping_column not in fieldnames:
        missing.append(config.grouping_column)
    for column in config.value_columns:
        if column not in fieldnames:
            missing.append(column)
    if missing:
        print(
            "El ranking se eliminó porque ya no existen estas columnas: "
            + ", ".join(sorted(set(missing)))
        )
        return None
    return config


def count_filtered_rows(
    data_path: Path,
    fieldnames: List[str],
    column_filters: Dict[str, set[str]],
    min_scores: Dict[str, float],
    max_scores: Dict[str, float],
    sort_keys: Sequence[SortKey],
    use_max_scores: bool,
    weighting: Optional[WeightingConfig],
    display_labels: bool,
    maps: CodeMaps,
) -> None:
    _, rows = read_csv_rows(data_path)
    min_list = [ScoreFilter(column=col, threshold=value) for col, value in min_scores.items()]
    max_list = [ScoreFilter(column=col, threshold=value) for col, value in max_scores.items()]
    subject_columns = subject_columns_for_weighting(use_max_scores)
    total_rows = 0
    matched_rows = 0
    excluded_weighting = 0
    filtered_rows: List[Dict[str, str]] = []
    for row in rows:
        total_rows += 1
        transformed = apply_max_scores(row, use_max_scores)
        transformed, eligible = apply_weighted_score(
            transformed, weighting, subject_columns
        )
        if weighting and weighting.enabled and not eligible:
            excluded_weighting += 1
            continue
        if not row_matches(transformed, column_filters, min_list, max_list):
            continue
        matched_rows += 1
        filtered_rows.append(transformed)
    print("\nResumen con filtros actuales:")
    print(f"- Total de registros: {total_rows}")
    print(f"- Registros filtrados: {matched_rows}")
    if matched_rows == 0:
        print("\nNo hay filas filtradas para mostrar.")
        return

    if sort_keys:
        sort_rows(filtered_rows, sort_keys)

    print(
        "\nFilas filtradas (ordenadas por columna):"
        if sort_keys
        else "\nFilas filtradas (ordenadas por el archivo):"
    )
    numbered, mapping = build_numbered_headers(fieldnames)
    headers = ["#"] + numbered
    rows_table = []
    for index, row in enumerate(filtered_rows, start=1):
        row_values = [
            display_value(col, row.get(col, ""), maps, display_labels)
            for col in fieldnames
        ]
        rows_table.append([str(index)] + row_values)
    print_table(headers, rows_table)
    if mapping:
        print("\nEncabezados numéricos:")
        for short, original in mapping:
            print(f"- {short} = {original}")
    if weighting and weighting.enabled:
        print(
            "\nEstudiantes excluidos por falta de pruebas para ponderación: "
            f"{excluded_weighting}"
        )


def export_filtered_rows(
    data_path: Path,
    fieldnames: List[str],
    column_filters: Dict[str, set[str]],
    min_scores: Dict[str, float],
    max_scores: Dict[str, float],
    sort_keys: Sequence[SortKey],
    output_path: Path,
    use_max_scores: bool,
    weighting: Optional[WeightingConfig],
) -> None:
    min_list = [ScoreFilter(column=col, threshold=value) for col, value in min_scores.items()]
    max_list = [ScoreFilter(column=col, threshold=value) for col, value in max_scores.items()]
    _, rows = read_csv_rows(data_path)
    subject_columns = subject_columns_for_weighting(use_max_scores)
    total_rows = 0
    matched_rows = 0
    excluded_weighting = 0
    filtered_rows: List[Dict[str, str]] = []
    for row in rows:
        total_rows += 1
        transformed = apply_max_scores(row, use_max_scores)
        transformed, eligible = apply_weighted_score(
            transformed, weighting, subject_columns
        )
        if weighting and weighting.enabled and not eligible:
            excluded_weighting += 1
            continue
        if not row_matches(transformed, column_filters, min_list, max_list):
            continue
        matched_rows += 1
        filtered_rows.append(transformed)
    if sort_keys:
        sort_rows(filtered_rows, sort_keys)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in filtered_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print("\nExportación completada:")
    print(f"- Ruta: {output_path}")
    print(f"- Total de registros: {total_rows}")
    print(f"- Registros exportados: {matched_rows}")
    if weighting and weighting.enabled:
        print(
            "- Estudiantes excluidos por falta de pruebas para ponderación: "
            f"{excluded_weighting}"
        )


def show_filtered_statistics(
    data_path: Path,
    fieldnames: List[str],
    column_filters: Dict[str, set[str]],
    min_scores: Dict[str, float],
    max_scores: Dict[str, float],
    default_percentiles: str,
    use_max_scores: bool,
    weighting: Optional[WeightingConfig],
) -> None:
    stats_columns = prompt_stats_columns([], fieldnames)
    if not stats_columns:
        print("No se seleccionaron columnas para estadísticas.")
        return
    percentiles_raw = default_percentiles
    if prompt_yes_no("¿Deseas ajustar percentiles?", default=False):
        percentiles_raw = prompt_value(
            "Percentiles (separados por coma)", default_percentiles
        )
    percentiles = parse_percentiles(percentiles_raw)

    min_list = [ScoreFilter(column=col, threshold=value) for col, value in min_scores.items()]
    max_list = [ScoreFilter(column=col, threshold=value) for col, value in max_scores.items()]
    stats_values: Dict[str, List[float]] = {column: [] for column in stats_columns}
    stats_totals: Dict[str, int] = {column: 0 for column in stats_columns}
    stats_zero_counts: Dict[str, int] = {column: 0 for column in stats_columns}

    _, rows = read_csv_rows(data_path)
    subject_columns = subject_columns_for_weighting(use_max_scores)
    total_rows = 0
    matched_rows = 0
    excluded_weighting = 0
    for row in rows:
        total_rows += 1
        transformed = apply_max_scores(row, use_max_scores)
        transformed, eligible = apply_weighted_score(
            transformed, weighting, subject_columns
        )
        if weighting and weighting.enabled and not eligible:
            excluded_weighting += 1
            continue
        if not row_matches(transformed, column_filters, min_list, max_list):
            continue
        matched_rows += 1
        for column in stats_values:
            value = to_float(transformed.get(column, ""))
            record_stat_value(
                column,
                value,
                stats_values,
                stats_totals,
                stats_zero_counts,
            )

    print("\nResumen estadístico con filtros actuales:")
    print(f"- Total de registros: {total_rows}")
    print(f"- Registros filtrados: {matched_rows}")
    if weighting and weighting.enabled:
        print(
            "- Estudiantes excluidos por falta de pruebas para ponderación: "
            f"{excluded_weighting}"
        )
    print_statistics(stats_values, percentiles, stats_totals, stats_zero_counts)


def show_filtered_rbd_statistics(
    data_path: Path,
    fieldnames: List[str],
    column_filters: Dict[str, set[str]],
    min_scores: Dict[str, float],
    max_scores: Dict[str, float],
    default_percentiles: str,
    use_max_scores: bool,
    weighting: Optional[WeightingConfig],
) -> None:
    rbd_values = column_filters.get("RBD") or set()
    if not rbd_values:
        print("Debes indicar un filtro de RBD para ver estadísticas por RBD.")
        return
    stats_columns = prompt_stats_columns([], fieldnames)
    if not stats_columns:
        print("No se seleccionaron columnas para estadísticas.")
        return
    percentiles_raw = default_percentiles
    if prompt_yes_no("¿Deseas ajustar percentiles?", default=False):
        percentiles_raw = prompt_value(
            "Percentiles (separados por coma)", default_percentiles
        )
    percentiles = parse_percentiles(percentiles_raw)

    min_list = [ScoreFilter(column=col, threshold=value) for col, value in min_scores.items()]
    max_list = [ScoreFilter(column=col, threshold=value) for col, value in max_scores.items()]
    stats_by_rbd: Dict[str, Dict[str, List[float]]] = {
        rbd: {column: [] for column in stats_columns}
        for rbd in sorted(rbd_values)
    }
    stats_totals_by_rbd: Dict[str, Dict[str, int]] = {
        rbd: {column: 0 for column in stats_columns}
        for rbd in sorted(rbd_values)
    }
    stats_zero_counts_by_rbd: Dict[str, Dict[str, int]] = {
        rbd: {column: 0 for column in stats_columns}
        for rbd in sorted(rbd_values)
    }

    _, rows = read_csv_rows(data_path)
    subject_columns = subject_columns_for_weighting(use_max_scores)
    total_rows = 0
    matched_rows = 0
    excluded_weighting = 0
    for row in rows:
        total_rows += 1
        transformed = apply_max_scores(row, use_max_scores)
        transformed, eligible = apply_weighted_score(
            transformed, weighting, subject_columns
        )
        if weighting and weighting.enabled and not eligible:
            excluded_weighting += 1
            continue
        if not row_matches(transformed, column_filters, min_list, max_list):
            continue
        matched_rows += 1
        rbd = normalize_code(transformed.get("RBD"))
        if rbd not in stats_by_rbd:
            continue
        for column in stats_columns:
            value = to_float(transformed.get(column, ""))
            record_stat_value_by_rbd(
                rbd,
                column,
                value,
                stats_by_rbd,
                stats_totals_by_rbd,
                stats_zero_counts_by_rbd,
            )

    print("\nResumen estadístico por RBD con filtros actuales:")
    print(f"- Total de registros: {total_rows}")
    print(f"- Registros filtrados: {matched_rows}")
    if weighting and weighting.enabled:
        print(
            "- Estudiantes excluidos por falta de pruebas para ponderación: "
            f"{excluded_weighting}"
        )
    print_rbd_statistics(
        stats_by_rbd,
        percentiles,
        stats_totals_by_rbd,
        stats_zero_counts_by_rbd,
    )


def show_filtered_ranking(
    data_path: Path,
    fieldnames: List[str],
    column_filters: Dict[str, set[str]],
    min_scores: Dict[str, float],
    max_scores: Dict[str, float],
    use_max_scores: bool,
    weighting: Optional[WeightingConfig],
    maps: CodeMaps,
    display_labels: bool,
) -> Optional[RankingConfig]:
    return show_filtered_ranking_with_config(
        data_path,
        fieldnames,
        column_filters,
        min_scores,
        max_scores,
        use_max_scores,
        weighting,
        maps,
        display_labels,
        None,
    )


def show_filtered_ranking_with_config(
    data_path: Path,
    fieldnames: List[str],
    column_filters: Dict[str, set[str]],
    min_scores: Dict[str, float],
    max_scores: Dict[str, float],
    use_max_scores: bool,
    weighting: Optional[WeightingConfig],
    maps: CodeMaps,
    display_labels: bool,
    config: Optional[RankingConfig],
) -> Optional[RankingConfig]:
    if config is None:
        config = prompt_ranking_config(fieldnames)
        if not config:
            return None

    grouping_column = config.grouping_column
    value_columns = config.value_columns
    metric = config.metric
    order_desc = config.order_desc
    limit = config.limit

    min_list = [ScoreFilter(column=col, threshold=value) for col, value in min_scores.items()]
    max_list = [ScoreFilter(column=col, threshold=value) for col, value in max_scores.items()]
    subject_columns = subject_columns_for_weighting(use_max_scores)

    values_by_group: Dict[str, List[float]] = {}
    excluded_by_group: Dict[str, int] = {}
    _, rows = read_csv_rows(data_path)
    for row in rows:
        transformed = apply_max_scores(row, use_max_scores)
        transformed, eligible = apply_weighted_score(
            transformed, weighting, subject_columns
        )
        if weighting and weighting.enabled and not eligible:
            continue
        if not row_matches(transformed, column_filters, min_list, max_list):
            continue
        group_value = normalize_code(transformed.get(grouping_column))
        if not group_value:
            continue
        value, eligible_for_ranking = evaluate_row_value(transformed, value_columns)
        if not eligible_for_ranking:
            excluded_by_group[group_value] = excluded_by_group.get(group_value, 0) + 1
            continue
        values_by_group.setdefault(group_value, []).append(value)

    if not values_by_group:
        print("No hay datos suficientes para generar el ranking.")
        return

    ranking_rows = []
    for group, values in values_by_group.items():
        if not values:
            continue
        if metric == "mean":
            metric_value = mean(values)
        elif metric == "median":
            metric_value = median(values)
        elif metric == "mode":
            mode_value = calculate_mode(values)
            if mode_value is None:
                continue
            metric_value = mode_value
        else:
            metric_value = pstdev(values) if len(values) > 1 else 0.0
        description = label_value(grouping_column, group, maps)
        group_display = display_value(grouping_column, group, maps, display_labels)
        description_display = description or ""
        if display_labels and grouping_column in {"RBD", "CODIGO_COMUNA", "CODIGO_REGION"}:
            description_display = group or ""
        ranking_rows.append(
            [
                group_display or "(vacío)",
                description_display,
                len(values),
                excluded_by_group.get(group, 0),
                metric_value,
            ]
        )

    ranking_rows.sort(key=lambda row: row[4], reverse=order_desc)
    if limit:
        ranking_rows = ranking_rows[:limit]

    headers = [
        "Grupo",
        "Descripción",
        "Considerados",
        "Excluidos",
        metric_label(metric, value_columns),
    ]
    formatted_rows = []
    for index, row in enumerate(ranking_rows, start=1):
        formatted_rows.append(
            [
                f"{index}",
                row[0],
                row[1],
                str(row[2]),
                str(row[3]),
                f"{row[4]:.2f}",
            ]
        )
    print("\nRanking con filtros actuales:")
    print_table(["#", *headers], formatted_rows)
    return config


def manage_filters(
    fieldnames: List[str],
    data_path: Path,
    maps: CodeMaps,
    initial_column_filters: Dict[str, set[str]],
    initial_min_scores: Dict[str, float],
    initial_max_scores: Dict[str, float],
    initial_sort_by: List[str],
    sort_keys: List[SortKey],
    default_percentiles: str,
    initial_output_csv: Optional[str],
    use_max_scores: bool,
    weighting: Optional[WeightingConfig],
    display_labels: bool,
) -> Tuple[
    Dict[str, set[str]],
    Dict[str, float],
    Dict[str, float],
    List[str],
    List[SortKey],
    Optional[str],
    bool,
    Optional[WeightingConfig],
    bool,
]:
    base_fieldnames = list(fieldnames)
    column_filters = dict(initial_column_filters)
    min_scores = dict(initial_min_scores)
    max_scores = dict(initial_max_scores)
    sort_by = list(initial_sort_by)
    output_csv = initial_output_csv
    weighting_config = weighting
    value_cache: Dict[str, Counter] = {}
    ranking_config: Optional[RankingConfig] = None
    view_mode = "datos"

    def refresh_ranking_if_visible() -> None:
        nonlocal ranking_config
        if view_mode != "ranking":
            return
        if not ranking_config:
            print("No hay ranking configurado.")
            return
        ranking_config = show_filtered_ranking_with_config(
            data_path,
            fieldnames,
            column_filters,
            min_scores,
            max_scores,
            use_max_scores,
            weighting_config,
            maps,
            display_labels,
            ranking_config,
        )
    while True:
        summarize_filters(column_filters, min_scores, max_scores)
        summarize_weighting(weighting_config)
        summarize_ranking(ranking_config)
        if sort_by:
            print(f"- Orden actual: {', '.join(sort_by)}")
        print(
            "\n¿Qué deseas hacer ahora?\n"
            "1. Agregar filtro\n"
            "2. Modificar filtro\n"
            "3. Eliminar filtro\n"
            "4. Reiniciar filtros\n"
            "5. Ordenar datos\n"
            "6. Mostrar datos actuales\n"
            "7. Ver estadísticos\n"
            "8. Ver estadísticos por RBD\n"
            "9. Configurar/modificar ranking\n"
            "10. Ver ranking\n"
            "11. Eliminar ranking\n"
            "12. Exportar filtrados a CSV\n"
            "13. Alternar puntajes MAX\n"
            "14. Configurar ponderación\n"
            "15. Eliminar ponderación\n"
            "16. Transformar RBD/colegio\n"
            "17. Continuar\n"
            "18. Terminar programa"
        )
        choice = input("Selecciona una opción: ").strip()
        if choice == "1":
            print("\nTipos de filtro:")
            print("1. Valores exactos")
            print("2. Mínimo")
            print("3. Máximo")
            filter_type = input("Selecciona el tipo: ").strip()
            if filter_type == "1":
                column = prompt_column(fieldnames)
                values = prompt_column_value(column, data_path, maps, value_cache)
                if values:
                    column_filters[column] = values
            elif filter_type == "2":
                column = prompt_column(fieldnames)
                threshold = prompt_threshold(column)
                if threshold is not None:
                    min_scores[column] = threshold
            elif filter_type == "3":
                column = prompt_column(fieldnames)
                threshold = prompt_threshold(column)
                if threshold is not None:
                    max_scores[column] = threshold
            else:
                print("Tipo inválido.")
            refresh_ranking_if_visible()
        elif choice == "2":
            index = build_filter_index(column_filters, min_scores, max_scores)
            if not index:
                print("No hay filtros para modificar.")
                continue
            options = []
            for kind, column in index:
                label = (
                    f"{column} en [{', '.join(sorted(column_filters.get(column, [])))}]"
                    if kind == "igual"
                    else f"{column} {'>=' if kind == 'min' else '<='} "
                    f"{min_scores.get(column) if kind == 'min' else max_scores.get(column)}"
                )
                options.append(label)
            selection = prompt_choice("Selecciona el filtro a modificar:", options)
            kind, column = index[selection - 1]
            if kind == "igual":
                values = prompt_column_value(column, data_path, maps, value_cache)
                if values:
                    column_filters[column] = values
            elif kind == "min":
                threshold = prompt_threshold(column)
                if threshold is not None:
                    min_scores[column] = threshold
            elif kind == "max":
                threshold = prompt_threshold(column)
                if threshold is not None:
                    max_scores[column] = threshold
            refresh_ranking_if_visible()
        elif choice == "3":
            index = build_filter_index(column_filters, min_scores, max_scores)
            if not index:
                print("No hay filtros para eliminar.")
                continue
            options = []
            for kind, column in index:
                label = (
                    f"{column} en [{', '.join(sorted(column_filters.get(column, [])))}]"
                    if kind == "igual"
                    else f"{column} {'>=' if kind == 'min' else '<='} "
                    f"{min_scores.get(column) if kind == 'min' else max_scores.get(column)}"
                )
                options.append(label)
            selection = prompt_choice("Selecciona el filtro a eliminar:", options)
            kind, column = index[selection - 1]
            if kind == "igual":
                column_filters.pop(column, None)
            elif kind == "min":
                min_scores.pop(column, None)
            elif kind == "max":
                max_scores.pop(column, None)
            refresh_ranking_if_visible()
        elif choice == "4":
            if prompt_yes_no("¿Seguro que deseas reiniciar los filtros?", default=False):
                column_filters.clear()
                min_scores.clear()
                max_scores.clear()
                refresh_ranking_if_visible()
        elif choice == "5":
            sort_by = prompt_sort_by([], fieldnames)
            sort_keys = parse_sort_by_args(sort_by, fieldnames)
            if sort_by:
                print(f"Orden aplicado: {', '.join(sort_by)}")
            else:
                print("Orden limpiado; se usará el orden del archivo.")
        elif choice == "6":
            view_mode = "datos"
            count_filtered_rows(
                data_path,
                fieldnames,
                column_filters,
                min_scores,
                max_scores,
                sort_keys,
                use_max_scores,
                weighting_config,
                display_labels,
                maps,
            )
        elif choice == "7":
            show_filtered_statistics(
                data_path,
                fieldnames,
                column_filters,
                min_scores,
                max_scores,
                default_percentiles,
                use_max_scores,
                weighting_config,
            )
        elif choice == "8":
            show_filtered_rbd_statistics(
                data_path,
                fieldnames,
                column_filters,
                min_scores,
                max_scores,
                default_percentiles,
                use_max_scores,
                weighting_config,
            )
        elif choice == "9":
            ranking_config = update_ranking_config(ranking_config, fieldnames)
            if ranking_config:
                view_mode = "ranking"
                ranking_config = show_filtered_ranking_with_config(
                    data_path,
                    fieldnames,
                    column_filters,
                    min_scores,
                    max_scores,
                    use_max_scores,
                    weighting_config,
                    maps,
                    display_labels,
                    ranking_config,
                )
        elif choice == "10":
            view_mode = "ranking"
            ranking_config = show_filtered_ranking_with_config(
                data_path,
                fieldnames,
                column_filters,
                min_scores,
                max_scores,
                use_max_scores,
                weighting_config,
                maps,
                display_labels,
                ranking_config,
            )
        elif choice == "11":
            if ranking_config:
                ranking_config = None
                if view_mode == "ranking":
                    view_mode = "datos"
                print("Ranking eliminado.")
            else:
                print("No hay ranking configurado.")
        elif choice == "12":
            output_csv = prompt_value(
                "Ruta del CSV de salida", output_csv or "filtrados.csv"
            )
            if output_csv:
                export_filtered_rows(
                    data_path,
                    fieldnames,
                    column_filters,
                    min_scores,
                    max_scores,
                    sort_keys,
                    Path(output_csv),
                    use_max_scores,
                    weighting_config,
                )
        elif choice == "13":
            use_max_scores = not use_max_scores
            fieldnames = build_active_fieldnames(
                base_fieldnames, use_max_scores, weighting_config
            )
            (
                column_filters,
                min_scores,
                max_scores,
                sort_by,
            ) = prune_filters_for_fieldnames(
                column_filters,
                min_scores,
                max_scores,
                sort_by,
                fieldnames,
            )
            sort_keys = parse_sort_by_args(sort_by, fieldnames) if sort_by else []
            estado = "activado" if use_max_scores else "desactivado"
            print(f"Modo puntajes MAX {estado}.")
            ranking_config = validate_ranking_config(ranking_config, fieldnames)
            refresh_ranking_if_visible()
        elif choice == "14":
            weighting_config = prompt_weighting(weighting_config, use_max_scores)
            fieldnames = build_active_fieldnames(
                base_fieldnames, use_max_scores, weighting_config
            )
            (
                column_filters,
                min_scores,
                max_scores,
                sort_by,
            ) = prune_filters_for_fieldnames(
                column_filters,
                min_scores,
                max_scores,
                sort_by,
                fieldnames,
            )
            sort_keys = parse_sort_by_args(sort_by, fieldnames) if sort_by else []
            ranking_config = validate_ranking_config(ranking_config, fieldnames)
            refresh_ranking_if_visible()
        elif choice == "15":
            if weighting_config and weighting_config.enabled:
                weighting_config = WeightingConfig(
                    weights=weighting_config.weights,
                    history_science_mode=weighting_config.history_science_mode,
                    enabled=False,
                )
                fieldnames = build_active_fieldnames(
                    base_fieldnames, use_max_scores, weighting_config
                )
                (
                    column_filters,
                    min_scores,
                    max_scores,
                    sort_by,
                ) = prune_filters_for_fieldnames(
                    column_filters,
                    min_scores,
                    max_scores,
                    sort_by,
                    fieldnames,
                )
                sort_keys = parse_sort_by_args(sort_by, fieldnames) if sort_by else []
                print("Ponderación eliminada.")
            else:
                print("No hay ponderación activa para eliminar.")
            ranking_config = validate_ranking_config(ranking_config, fieldnames)
            refresh_ranking_if_visible()
        elif choice == "16":
            prompt_rbd_lookup(maps)
        elif choice == "17":
            return (
                column_filters,
                min_scores,
                max_scores,
                sort_by,
                sort_keys,
                output_csv,
                use_max_scores,
                weighting_config,
                display_labels,
            )
        elif choice == "18":
            print("Programa terminado por el usuario.")
            raise SystemExit(0)
        else:
            print("Opción inválida.")


def collect_interactive_filters(
    args: argparse.Namespace,
    fieldnames: List[str],
    data_path: Path,
    maps: CodeMaps,
) -> Tuple[
    Dict[str, set[str]],
    List[ScoreFilter],
    List[ScoreFilter],
    List[SortKey],
    bool,
    Optional[WeightingConfig],
    bool,
]:
    print("=== Modo interactivo: análisis de rendición ===")

    if prompt_yes_no("¿Deseas ver las columnas disponibles?", default=False):
        print_column_index(fieldnames)

    initial_filters = {}
    if args.rbd:
        initial_filters["RBD"] = parse_list_arg(args.rbd) or set()
    if args.situacion_egreso:
        initial_filters["SITUACION_EGRESO"] = parse_list_arg(args.situacion_egreso) or set()
    if args.cod_ens:
        initial_filters["COD_ENS"] = parse_list_arg(args.cod_ens) or set()
    if args.region:
        initial_filters["CODIGO_REGION"] = parse_list_arg(args.region) or set()
    if args.comuna:
        initial_filters["CODIGO_COMUNA"] = parse_list_arg(args.comuna) or set()
    if args.grupo_dependencia:
        initial_filters["GRUPO_DEPENDENCIA"] = (
            parse_list_arg(args.grupo_dependencia) or set()
        )
    if args.rama_educacional:
        initial_filters["RAMA_EDUCACIONAL"] = (
            parse_list_arg(args.rama_educacional) or set()
        )

    min_scores = {f.column: f.threshold for f in parse_score_filters(args.min_score)}
    max_scores = {f.column: f.threshold for f in parse_score_filters(args.max_score)}

    args.sort_by = prompt_sort_by(args.sort_by, fieldnames)
    sort_keys = parse_sort_by_args(args.sort_by, fieldnames)

    use_max_scores = False
    weighting_config = None
    (
        column_filters,
        min_scores,
        max_scores,
        args.sort_by,
        sort_keys,
        output_csv,
        use_max_scores,
        weighting_config,
        display_labels,
    ) = manage_filters(
        fieldnames,
        data_path,
        maps,
        initial_filters,
        min_scores,
        max_scores,
        args.sort_by,
        sort_keys,
        args.percentiles,
        args.output_csv,
        use_max_scores,
        weighting_config,
        False,
    )

    active_fieldnames = build_active_fieldnames(
        fieldnames, use_max_scores, weighting_config
    )
    args.count_by = prompt_count_by(args.count_by, active_fieldnames)
    args.stats = prompt_stats_columns(args.stats, active_fieldnames)
    if args.stats and prompt_yes_no("¿Deseas ajustar percentiles?", default=False):
        args.percentiles = prompt_value("Percentiles (separados por coma)", args.percentiles)

    if output_csv:
        args.output_csv = output_csv
    if not args.output_csv and prompt_yes_no("¿Deseas exportar los filtrados a CSV?"):
        args.output_csv = prompt_value("Ruta del CSV de salida", "filtrados.csv")

    if not args.add_labels:
        args.add_labels = prompt_yes_no("¿Agregar columnas descriptivas?", default=False)

    min_list = [ScoreFilter(column=col, threshold=value) for col, value in min_scores.items()]
    max_list = [ScoreFilter(column=col, threshold=value) for col, value in max_scores.items()]
    return (
        column_filters,
        min_list,
        max_list,
        sort_keys,
        use_max_scores,
        weighting_config,
        display_labels,
    )


def main() -> None:
    args = parse_args()
    if args.interactive or len(sys.argv) == 1:
        args.data = prompt_data_file(args.data)
    data_path = Path(args.data)
    if not data_path.exists():
        raise AnalysisError(f"No se encontró el archivo de datos: {data_path}")

    fieldnames, rows = read_csv_rows(data_path)
    if args.list_columns:
        print("Columnas disponibles:")
        print_column_index(fieldnames)
        return

    maps = build_code_maps(
        Path(args.cod_ens_csv),
        Path(args.comunas_csv),
        Path(args.codebook_csv),
        Path(args.rbd_colegios_csv),
    )

    use_max_scores = False
    weighting_config = None
    display_labels = False
    if args.interactive or len(sys.argv) == 1:
        (
            column_filters,
            min_scores,
            max_scores,
            sort_keys,
            use_max_scores,
            weighting_config,
            display_labels,
        ) = collect_interactive_filters(args, fieldnames, data_path, maps)
    else:
        column_filters = {
            "RBD": parse_list_arg(args.rbd),
            "SITUACION_EGRESO": parse_list_arg(args.situacion_egreso),
            "COD_ENS": parse_list_arg(args.cod_ens),
            "CODIGO_REGION": parse_list_arg(args.region),
            "CODIGO_COMUNA": parse_list_arg(args.comuna),
            "GRUPO_DEPENDENCIA": parse_list_arg(args.grupo_dependencia),
            "RAMA_EDUCACIONAL": parse_list_arg(args.rama_educacional),
        }
        min_scores = parse_score_filters(args.min_score)
        max_scores = parse_score_filters(args.max_score)
        sort_keys = parse_sort_by_args(args.sort_by, fieldnames)
    active_fieldnames = build_active_fieldnames(
        fieldnames, use_max_scores, weighting_config
    )
    percentiles = parse_percentiles(args.percentiles)
    stats_columns = list(dict.fromkeys(args.stats))
    if args.stats_by_rbd and not stats_columns:
        raise AnalysisError("Debes indicar al menos una columna en --stats para --stats-by-rbd.")
    invalid_stats = [column for column in stats_columns if column not in active_fieldnames]
    if invalid_stats:
        raise AnalysisError(
            "Columnas inválidas para estadísticas: " + ", ".join(invalid_stats)
        )
    counts: Dict[str, Counter] = {column: Counter() for column in args.count_by}
    stats_values: Dict[str, List[float]] = {column: [] for column in stats_columns}
    stats_totals: Dict[str, int] = {column: 0 for column in stats_columns}
    stats_zero_counts: Dict[str, int] = {column: 0 for column in stats_columns}
    stats_by_rbd: Optional[Dict[str, Dict[str, List[float]]]] = None
    stats_totals_by_rbd: Optional[Dict[str, Dict[str, int]]] = None
    stats_zero_counts_by_rbd: Optional[Dict[str, Dict[str, int]]] = None
    if args.stats_by_rbd:
        rbd_values = column_filters.get("RBD") or set()
        if not rbd_values:
            raise AnalysisError("Para --stats-by-rbd debes indicar --rbd con al menos un valor.")
        stats_by_rbd = {
            rbd: {column: [] for column in stats_columns}
            for rbd in sorted(rbd_values)
        }
        stats_totals_by_rbd = {
            rbd: {column: 0 for column in stats_columns}
            for rbd in sorted(rbd_values)
        }
        stats_zero_counts_by_rbd = {
            rbd: {column: 0 for column in stats_columns}
            for rbd in sorted(rbd_values)
        }

    output_writer = None
    output_handle = None
    output_fields = list(active_fieldnames)
    if args.add_labels:
        for label in ["COD_ENS_DESC", "REGION_NOMBRE", "COMUNA_NOMBRE"]:
            if label not in output_fields:
                output_fields.append(label)

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_handle = output_path.open("w", encoding="utf-8", newline="")
        output_writer = csv.DictWriter(output_handle, fieldnames=output_fields)
        output_writer.writeheader()

    total_rows = 0
    matched_rows = 0
    excluded_weighting = 0
    subject_columns = subject_columns_for_weighting(use_max_scores)

    try:
        if sort_keys:
            filtered_rows: List[Dict[str, str]] = []
            for row in rows:
                total_rows += 1
                transformed = apply_max_scores(row, use_max_scores)
                transformed, eligible = apply_weighted_score(
                    transformed, weighting_config, subject_columns
                )
                if weighting_config and weighting_config.enabled and not eligible:
                    excluded_weighting += 1
                    continue
                if not row_matches(transformed, column_filters, min_scores, max_scores):
                    continue
                matched_rows += 1
                enriched = (
                    add_label_columns(transformed, maps)
                    if args.add_labels
                    else transformed
                )
                filtered_rows.append(enriched)
                for column in stats_values:
                    value = to_float(transformed.get(column, ""))
                    record_stat_value(
                        column,
                        value,
                        stats_values,
                        stats_totals,
                        stats_zero_counts,
                    )
                if stats_by_rbd is not None:
                    rbd = normalize_code(transformed.get("RBD"))
                    if rbd in stats_by_rbd:
                        for column in stats_columns:
                            value = to_float(transformed.get(column, ""))
                            record_stat_value_by_rbd(
                                rbd,
                                column,
                                value,
                                stats_by_rbd,
                                stats_totals_by_rbd or {},
                                stats_zero_counts_by_rbd or {},
                            )
            if filtered_rows:
                sort_rows(filtered_rows, sort_keys)
            if output_writer:
                for row in filtered_rows:
                    output_writer.writerow({key: row.get(key, "") for key in output_fields})
            for row in filtered_rows:
                for column in counts:
                    value = normalize_code(row.get(column, ""))
                    if args.add_labels:
                        value = label_value(column, value, maps)
                    counts[column][value] += 1
        else:
            for row in rows:
                total_rows += 1
                transformed = apply_max_scores(row, use_max_scores)
                transformed, eligible = apply_weighted_score(
                    transformed, weighting_config, subject_columns
                )
                if weighting_config and weighting_config.enabled and not eligible:
                    excluded_weighting += 1
                    continue
                if not row_matches(transformed, column_filters, min_scores, max_scores):
                    continue
                matched_rows += 1
                enriched = (
                    add_label_columns(transformed, maps)
                    if args.add_labels
                    else transformed
                )
                if output_writer:
                    output_writer.writerow({key: enriched.get(key, "") for key in output_fields})
                for column in counts:
                    value = normalize_code(enriched.get(column, ""))
                    if args.add_labels:
                        value = label_value(column, value, maps)
                    counts[column][value] += 1
                for column in stats_values:
                    value = to_float(transformed.get(column, ""))
                    record_stat_value(
                        column,
                        value,
                        stats_values,
                        stats_totals,
                        stats_zero_counts,
                    )
                if stats_by_rbd is not None:
                    rbd = normalize_code(transformed.get("RBD"))
                    if rbd in stats_by_rbd:
                        for column in stats_columns:
                            value = to_float(transformed.get(column, ""))
                            record_stat_value_by_rbd(
                                rbd,
                                column,
                                value,
                                stats_by_rbd,
                                stats_totals_by_rbd or {},
                                stats_zero_counts_by_rbd or {},
                            )
    finally:
        if output_handle:
            output_handle.close()

    print("Resumen del análisis:")
    print(f"- Total de registros: {total_rows}")
    print(f"- Registros filtrados: {matched_rows}")
    if weighting_config and weighting_config.enabled:
        print(
            "- Estudiantes excluidos por falta de pruebas para ponderación: "
            f"{excluded_weighting}"
        )

    if counts:
        print_counts(counts)
    if stats_values:
        print_statistics(stats_values, percentiles, stats_totals, stats_zero_counts)
    if stats_by_rbd:
        print_rbd_statistics(
            stats_by_rbd,
            percentiles,
            stats_totals_by_rbd,
            stats_zero_counts_by_rbd,
        )


if __name__ == "__main__":
    main()
