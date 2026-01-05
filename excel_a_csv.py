import os
import pandas as pd
from tkinter import Tk, filedialog

def select_excel_and_convert_to_csv():
    """
    Navega hasta 2 niveles hacia atrás desde la carpeta actual para seleccionar un archivo Excel,
    luego convierte cada hoja a CSV en la carpeta del archivo original.
    """
    # Configurar la ventana de diálogo (oculta)
    root = Tk()
    root.withdraw()
    
    # Obtener la carpeta actual y posibles carpetas padre
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    
    # Intentar seleccionar archivo empezando desde la carpeta actual
    file_path = filedialog.askopenfilename(
        initialdir=current_dir,
        title="Seleccione archivo Excel",
        filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
    )
    
    # Si no se seleccionó en la carpeta actual, intentar en carpetas padre
    if not file_path:
        file_path = filedialog.askopenfilename(
            initialdir=parent_dir,
            title="Seleccione archivo Excel (nivel 1 atrás)",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        
    if not file_path:
        file_path = filedialog.askopenfilename(
            initialdir=grandparent_dir,
            title="Seleccione archivo Excel (nivel 2 atrás)",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
    
    if not file_path:
        print("No se seleccionó ningún archivo.")
        return
    
    # Obtener la carpeta del archivo seleccionado para guardar los CSV
    output_folder = os.path.dirname(file_path)
    
    # Llamar a la función de conversión (usando la función anterior)
    try:
        created_files = excel_to_csv(file_path, output_folder)
        print(f"\nConversión completada. Archivos creados:")
        for file in created_files:
            print(f"- {file}")
    except Exception as e:
        print(f"\nError al convertir el archivo: {e}")

# Función de conversión (la misma que antes)
def excel_to_csv(input_file, output_folder=''):
    """
    Convierte cada hoja de un archivo Excel en un archivo CSV separado.
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"No se encontró el archivo: {input_file}")
    
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    excel_file = pd.ExcelFile(input_file)
    created_files = []
    
    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        csv_file = f"{base_name}_{sheet_name}.csv"
        
        if output_folder:
            csv_file = os.path.join(output_folder, csv_file)
        
        df.to_csv(csv_file, index=False, encoding='utf-8')
        created_files.append(csv_file)
    
    return created_files

# Ejemplo de uso:
select_excel_and_convert_to_csv()