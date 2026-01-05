"""Herramientas de análisis para archivo de rendición.

Ejemplos de uso:
  python analisis_rendicion.py --list-columns
  python analisis_rendicion.py --situacion-egreso 1 --region 6 --count-by CODIGO_COMUNA
  python analisis_rendicion.py --min-score CLEC_REG_ACTUAL=500 --min-score MATE1_REG_ACTUAL=500 \
      --output-csv filtrados.csv --add-labels
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


@dataclass(frozen=True)
class ScoreFilter:
    column: str
    threshold: float


@dataclass
class CodeMaps:
    cod_ens: Dict[str, str]
    regiones: Dict[str, str]
    comunas: Dict[str, str]


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


def to_float(value: str) -> Optional[float]:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


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


def build_code_maps(cod_ens_path: Path, comunas_path: Path) -> CodeMaps:
    cod_ens = load_cod_ens(cod_ens_path)
    regiones, comunas = load_comunas_regiones(comunas_path)
    return CodeMaps(cod_ens=cod_ens, regiones=regiones, comunas=comunas)


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


def label_value(column: str, value: str, maps: CodeMaps) -> str:
    if column == "COD_ENS" and value in maps.cod_ens:
        return f"{value} - {maps.cod_ens[value]}"
    if column == "CODIGO_REGION" and value in maps.regiones:
        return f"{value} - {maps.regiones[value]}"
    if column == "CODIGO_COMUNA" and value in maps.comunas:
        return f"{value} - {maps.comunas[value]}"
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


def print_counts(counts: Dict[str, Counter]) -> None:
    for column, counter in counts.items():
        print(f"\nRecuento por {column}:")
        for key, value in counter.most_common():
            label = key if key else "(vacío)"
            print(f"  {label}: {value}")


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


def prompt_column_value(column: str) -> set[str]:
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
) -> None:
    fieldnames, rows = read_csv_rows(data_path)
    min_list = [ScoreFilter(column=col, threshold=value) for col, value in min_scores.items()]
    max_list = [ScoreFilter(column=col, threshold=value) for col, value in max_scores.items()]
    total_rows = 0
    matched_rows = 0
    for row in rows:
        total_rows += 1
        if not row_matches(row, column_filters, min_list, max_list):
            continue
        matched_rows += 1
    print("\nResumen con filtros actuales:")
    print(f"- Total de registros: {total_rows}")
    print(f"- Registros filtrados: {matched_rows}")
    if matched_rows == 0:
        print("\nNo hay filas filtradas para mostrar.")
        return

    print("\nFilas filtradas (ordenadas por el archivo):")
    _, rows = read_csv_rows(data_path)
    row_index = 0
    for row in rows:
        if not row_matches(row, column_filters, min_list, max_list):
            continue
        row_index += 1
        print(f"\nFila {row_index}:")
        for col in fieldnames:
            print(f"  - {col}: {row.get(col, '')}")


def manage_filters(
    fieldnames: List[str],
    data_path: Path,
    initial_column_filters: Dict[str, set[str]],
    initial_min_scores: Dict[str, float],
    initial_max_scores: Dict[str, float],
) -> Tuple[Dict[str, set[str]], Dict[str, float], Dict[str, float]]:
    column_filters = dict(initial_column_filters)
    min_scores = dict(initial_min_scores)
    max_scores = dict(initial_max_scores)
    while True:
        summarize_filters(column_filters, min_scores, max_scores)
        print(
            "\n¿Qué deseas hacer ahora?\n"
            "1. Agregar filtro\n"
            "2. Modificar filtro\n"
            "3. Eliminar filtro\n"
            "4. Reiniciar filtros\n"
            "5. Mostrar datos actuales\n"
            "6. Continuar"
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
                values = prompt_column_value(column)
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
                values = prompt_column_value(column)
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
            count_filtered_rows(data_path, column_filters, min_scores, max_scores)
        elif choice == "6":
            return column_filters, min_scores, max_scores
        else:
            print("Opción inválida.")


def collect_interactive_filters(
    args: argparse.Namespace,
    fieldnames: List[str],
    data_path: Path,
) -> Tuple[Dict[str, set[str]], List[ScoreFilter], List[ScoreFilter]]:
    print("=== Modo interactivo: análisis de rendición ===")

    if prompt_yes_no("¿Deseas ver las columnas disponibles?", default=False):
        print("\nColumnas disponibles:")
        for name in fieldnames:
            print(f"- {name}")

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

    column_filters, min_scores, max_scores = manage_filters(
        fieldnames,
        data_path,
        initial_filters,
        min_scores,
        max_scores,
    )

    args.count_by = prompt_count_by(args.count_by, fieldnames)

    if not args.output_csv and prompt_yes_no("¿Deseas exportar los filtrados a CSV?"):
        args.output_csv = prompt_value("Ruta del CSV de salida", "filtrados.csv")

    if not args.add_labels:
        args.add_labels = prompt_yes_no("¿Agregar columnas descriptivas?", default=False)

    min_list = [ScoreFilter(column=col, threshold=value) for col, value in min_scores.items()]
    max_list = [ScoreFilter(column=col, threshold=value) for col, value in max_scores.items()]
    return column_filters, min_list, max_list


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
        for name in fieldnames:
            print(f"- {name}")
        return

    if args.interactive or len(sys.argv) == 1:
        column_filters, min_scores, max_scores = collect_interactive_filters(
            args, fieldnames, data_path
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

    maps = build_code_maps(Path(args.cod_ens_csv), Path(args.comunas_csv))

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
