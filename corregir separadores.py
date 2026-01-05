import os
import pandas as pd
from tkinter import Tk, filedialog

def select_and_convert_csv():
    """
    Permite navegar entre carpetas para seleccionar un CSV, lo convierte y guarda una versión corregida.
    """
    # Configurar la ventana de diálogo
    root = Tk()
    root.withdraw()
    
    # Obtener rutas de carpetas padre
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    
    # Intentar seleccionar archivo en diferentes niveles
    file_path = None
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