import fnmatch
import os

import pandas as pd
import tkinter as tk
from tkinter import filedialog

def buscar_archivos(base_dir, patrones):
    coincidencias = []
    for root, _, files in os.walk(base_dir):
        for pattern in patrones:
            for filename in fnmatch.filter(files, pattern):
                coincidencias.append(os.path.join(root, filename))
    return sorted(set(coincidencias))

def seleccionar_archivo_desde_lista(archivos, titulo):
    if not archivos:
        return None

    print(f"\nArchivos disponibles para {titulo}:")
    for idx, path in enumerate(archivos, start=1):
        print(f"{idx}. {os.path.relpath(path, os.getcwd())}")
    print("0. Abrir selector de archivos")

    while True:
        seleccion = input("Seleccione un número: ").strip()
        if seleccion == "0":
            return None
        if seleccion.isdigit():
            indice = int(seleccion) - 1
            if 0 <= indice < len(archivos):
                return archivos[indice]
        print("Selección no válida. Intente nuevamente.")

def convertir_columnas_a_enteros(ruta_csv, ruta_salida=None):
    df = pd.read_csv(ruta_csv)
    for col in df.columns:
        # Verifica si la columna es numérica
        if pd.api.types.is_numeric_dtype(df[col]):
            # Verifica si todos los valores son enteros (o floats con .0)
            if df[col].dropna().apply(lambda x: float(x).is_integer()).all():
                df[col] = df[col].astype('Int64')  # Usa Int64 para soportar NaN
    if ruta_salida:
        df.to_csv(ruta_salida, index=False)
    return df

def seleccionar_archivo():
    archivos_encontrados = buscar_archivos(os.getcwd(), ["*.csv"])
    archivo = seleccionar_archivo_desde_lista(
        archivos_encontrados,
        "CSV disponibles"
    )
    if archivo:
        return archivo

    root = tk.Tk()
    root.withdraw()
    archivo = filedialog.askopenfilename(
        title="Selecciona el archivo CSV",
        filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")]
    )
    return archivo

# Ejemplo de uso:
archivo = seleccionar_archivo()
if archivo:
    carpeta, nombre = os.path.split(archivo)
    nombre_salida = os.path.splitext(nombre)[0] + '_final.csv'
    ruta_salida = os.path.join(carpeta, nombre_salida)
    df = convertir_columnas_a_enteros(archivo, ruta_salida)
