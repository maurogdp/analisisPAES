import fnmatch
import os

import pandas as pd
from tkinter import Tk, filedialog

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

def select_and_convert_csv():
    """
    Permite navegar entre carpetas para seleccionar un CSV, lo convierte y guarda una versión corregida.
    """
    # Obtener rutas de carpetas padre
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    
    # Intentar seleccionar archivo en diferentes niveles
    archivos_encontrados = buscar_archivos(current_dir, ["*.csv"])
    file_path = seleccionar_archivo_desde_lista(
        archivos_encontrados,
        "CSV disponibles"
    )

    if not file_path:
        # Configurar la ventana de diálogo
        root = Tk()
        root.withdraw()

        for dir_path in [current_dir, parent_dir, grandparent_dir]:
            file_path = filedialog.askopenfilename(
                initialdir=dir_path,
                title=f"Seleccione archivo CSV (desde {os.path.basename(dir_path)})",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if file_path:
                break
    
    if not file_path:
        print("No se seleccionó ningún archivo.")
        return None
    
    # Generar nombre para el archivo corregido
    dir_name, file_name = os.path.split(file_path)
    name, ext = os.path.splitext(file_name)
    output_file = os.path.join(dir_name, f"{name}_corregido{ext}")
    
    # Convertir el archivo
    try:
        # Leer el archivo original
        df = pd.read_csv(file_path, sep=';', decimal=',', na_values='NA')
        
        # Guardar el archivo corregido
        df.to_csv(output_file, sep=',', decimal='.', index=False, na_rep='NA')
        
        print(f"\nArchivo convertido con éxito:")
        print(f"Original: {file_path}")
        print(f"Convertido: {output_file}")
        
        return output_file
    except Exception as e:
        print(f"\nError al convertir el archivo: {e}")
        return None

# Ejemplo de uso:
if __name__ == "__main__":
    select_and_convert_csv()
