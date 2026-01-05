import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os

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