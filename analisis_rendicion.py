"""Herramientas de análisis para archivo de rendición.

Ejemplos de uso:
  python analisis_rendicion.py --list-columns
  python analisis_rendicion.py --situacion-egreso 1 --region 6 --count-by CODIGO_COMUNA
  python analisis_rendicion.py --min-score CLEC_REG_ACTUAL=500 --min-score MATE1_REG_ACTUAL=500 \
      --output-csv filtrados.csv --add-labels
  python analisis_rendicion.py --sort-by CODIGO_REGION:asc --sort-by PUNTAJE:desc --output-csv ordenados.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
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


@dataclass(frozen=True)
class SortKey:
    column: str
    descending: bool = False


class AnalysisError(Exception):
    pass


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
        if column not in fieldnames:
            raise AnalysisError(f"Columna inválida para ordenar: {column}")
        if direction not in {"asc", "desc"}:
            raise AnalysisError(
                f"Dirección inválida '{direction}' en '{raw}'. Usa 'asc' o 'desc'."
            )
        sort_keys.append(SortKey(column=column, descending=direction == "desc"))
    return sort_keys


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


def build_code_maps(
    cod_ens_path: Path,
    comunas_path: Path,
    codebook_path: Path,
) -> CodeMaps:
    cod_ens = load_cod_ens(cod_ens_path)
    regiones, comunas = load_comunas_regiones(comunas_path)
    value_labels = load_codebook(codebook_path)
    return CodeMaps(
        cod_ens=cod_ens,
        regiones=regiones,
        comunas=comunas,
        value_labels=value_labels,
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
    return ""


def label_value(column: str, value: str, maps: CodeMaps) -> str:
    description = value_description(column, value, maps)
    if description:
        return f"{value} - {description}"
    return value


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
        column = item.split(":", 1)[0].strip()
        if column in fieldnames:
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


def count_filtered_rows(
    data_path: Path,
    column_filters: Dict[str, set[str]],
    min_scores: Dict[str, float],
    max_scores: Dict[str, float],
    sort_keys: Sequence[SortKey],
) -> None:
    fieldnames, rows = read_csv_rows(data_path)
    min_list = [ScoreFilter(column=col, threshold=value) for col, value in min_scores.items()]
    max_list = [ScoreFilter(column=col, threshold=value) for col, value in max_scores.items()]
    total_rows = 0
    matched_rows = 0
    filtered_rows: List[Dict[str, str]] = []
    for row in rows:
        total_rows += 1
        if not row_matches(row, column_filters, min_list, max_list):
            continue
        matched_rows += 1
        filtered_rows.append(row)
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
        rows_table.append([str(index)] + [row.get(col, "") for col in fieldnames])
    print_table(headers, rows_table)
    if mapping:
        print("\nEncabezados numéricos:")
        for short, original in mapping:
            print(f"- {short} = {original}")


def manage_filters(
    fieldnames: List[str],
    data_path: Path,
    maps: CodeMaps,
    initial_column_filters: Dict[str, set[str]],
    initial_min_scores: Dict[str, float],
    initial_max_scores: Dict[str, float],
    initial_sort_by: List[str],
    sort_keys: List[SortKey],
) -> Tuple[Dict[str, set[str]], Dict[str, float], Dict[str, float], List[str], List[SortKey]]:
    column_filters = dict(initial_column_filters)
    min_scores = dict(initial_min_scores)
    max_scores = dict(initial_max_scores)
    sort_by = list(initial_sort_by)
    value_cache: Dict[str, Counter] = {}
    while True:
        summarize_filters(column_filters, min_scores, max_scores)
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
            "7. Continuar\n"
            "8. Terminar programa"
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
        elif choice == "4":
            if prompt_yes_no("¿Seguro que deseas reiniciar los filtros?", default=False):
                column_filters.clear()
                min_scores.clear()
                max_scores.clear()
        elif choice == "5":
            sort_by = prompt_sort_by([], fieldnames)
            sort_keys = parse_sort_by_args(sort_by, fieldnames)
            if sort_by:
                print(f"Orden aplicado: {', '.join(sort_by)}")
            else:
                print("Orden limpiado; se usará el orden del archivo.")
        elif choice == "6":
            count_filtered_rows(
                data_path,
                column_filters,
                min_scores,
                max_scores,
                sort_keys,
            )
        elif choice == "7":
            return column_filters, min_scores, max_scores, sort_by, sort_keys
        elif choice == "8":
            print("Programa terminado por el usuario.")
            raise SystemExit(0)
        else:
            print("Opción inválida.")


def collect_interactive_filters(
    args: argparse.Namespace,
    fieldnames: List[str],
    data_path: Path,
    maps: CodeMaps,
) -> Tuple[Dict[str, set[str]], List[ScoreFilter], List[ScoreFilter], List[SortKey]]:
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

    column_filters, min_scores, max_scores, args.sort_by, sort_keys = manage_filters(
        fieldnames,
        data_path,
        maps,
        initial_filters,
        min_scores,
        max_scores,
        args.sort_by,
        sort_keys,
    )

    args.count_by = prompt_count_by(args.count_by, fieldnames)

    if not args.output_csv and prompt_yes_no("¿Deseas exportar los filtrados a CSV?"):
        args.output_csv = prompt_value("Ruta del CSV de salida", "filtrados.csv")

    if not args.add_labels:
        args.add_labels = prompt_yes_no("¿Agregar columnas descriptivas?", default=False)

    min_list = [ScoreFilter(column=col, threshold=value) for col, value in min_scores.items()]
    max_list = [ScoreFilter(column=col, threshold=value) for col, value in max_scores.items()]
    return column_filters, min_list, max_list, sort_keys


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
    )

    if args.interactive or len(sys.argv) == 1:
        column_filters, min_scores, max_scores, sort_keys = collect_interactive_filters(
            args, fieldnames, data_path, maps
        )
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
    counts: Dict[str, Counter] = {column: Counter() for column in args.count_by}

    output_writer = None
    output_handle = None
    output_fields = list(fieldnames)
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

    try:
        if sort_keys:
            filtered_rows: List[Dict[str, str]] = []
            for row in rows:
                total_rows += 1
                if not row_matches(row, column_filters, min_scores, max_scores):
                    continue
                matched_rows += 1
                enriched = add_label_columns(row, maps) if args.add_labels else row
                filtered_rows.append(enriched)
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
                if not row_matches(row, column_filters, min_scores, max_scores):
                    continue
                matched_rows += 1
                enriched = add_label_columns(row, maps) if args.add_labels else row
                if output_writer:
                    output_writer.writerow({key: enriched.get(key, "") for key in output_fields})
                for column in counts:
                    value = normalize_code(enriched.get(column, ""))
                    if args.add_labels:
                        value = label_value(column, value, maps)
                    counts[column][value] += 1
    finally:
        if output_handle:
            output_handle.close()

    print("Resumen del análisis:")
    print(f"- Total de registros: {total_rows}")
    print(f"- Registros filtrados: {matched_rows}")

    if counts:
        print_counts(counts)


if __name__ == "__main__":
    main()
