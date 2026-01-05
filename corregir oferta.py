import os
import pandas as pd
from tkinter import Tk, filedialog
import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def fix_encoding(text):
    # Mapeo de caracteres mal codificados a sus correcciones
    # replacements = {
    #     'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó', 'Ãº': 'ú',
    #     'Ã±': 'ñ', 'Ã‘': 'Ñ', 'Ã¼': 'ü', 'Ã‰': 'É', 'Ã“': 'Ó',
    #     'Ã': 'í', 'Ã': 'Í', 'Ã“': 'Ó', 'Ãš': 'Ú', 'Ã‘': 'Ñ',
    #     'Ã‰': 'É', 'Ã': 'Á', 'Ã“': 'Ó', 'Ã“': 'Ó', 'Ã‘': 'Ñ',
    #     'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó', 'Ãº': 'ú',
    #     'Ã±': 'ñ', 'Ã¼': 'ü', 'Â°': '°', 'Ã': 'Â', 'Ã£': 'ã',
    #     'Ã¢': 'â', 'Ã§': 'ç', 'Ãª': 'ê', 'Ãµ': 'õ', 'Ã¨': 'è',
    #     'Ã«': 'ë', 'Ã¯': 'ï', 'Ã´': 'ô', 'Ã¶': 'ö', 'Ã¹': 'ù',
    #     'Ã»': 'û', 'Ã¼': 'ü', 'Ã½': 'ý', 'Ã¿': 'ÿ', 'Ã€': 'À',
    #     'Ã‚': 'Â', 'Ãƒ': 'Ã', 'Ã„': 'Ä', 'Ã…': 'Å', 'Ã‡': 'Ç',
    #     'Ãˆ': 'È', 'Ã‰': 'É', 'ÃŠ': 'Ê', 'Ã‹': 'Ë', 'ÃŒ': 'Ì',
    #     'ÃŽ': 'Î', 'Ã‘': 'Ñ', 'Ã’': 'Ò', 'Ã“': 'Ó', 'Ã”': 'Ô',
    #     'Ã•': 'Õ', 'Ã–': 'Ö', 'Ã—': '×', 'Ã˜': 'Ø', 'Ã™': 'Ù',
    #     'Ãš': 'Ú', 'Ã›': 'Û', 'Ãœ': 'Ü', 'Ãž': 'Þ', 'ÃŸ': 'ß',
    #     'Ã¡': 'á', 'Ã¢': 'â', 'Ã£': 'ã', 'Ã¤': 'ä', 'Ã¥': 'å',
    #     'Ã¦': 'æ', 'Ã§': 'ç', 'Ã¨': 'è', 'Ã©': 'é', 'Ãª': 'ê',
    #     'Ã«': 'ë', 'Ã¬': 'ì', 'Ã­': 'í', 'Ã®': 'î', 'Ã¯': 'ï',
    #     'Ã°': 'ð', 'Ã±': 'ñ', 'Ã²': 'ò', 'Ã³': 'ó', 'Ã´': 'ô',
    #     'Ãµ': 'õ', 'Ã¶': 'ö', 'Ã·': '÷', 'Ã¸': 'ø', 'Ã¹': 'ù',
    #     'Ãº': 'ú', 'Ã»': 'û', 'Ã½': 'ý', 'Ã¾': 'þ', 'Ã¿': 'ÿ'
    # }

    replacements = {
        'Ã‘': 'Ñ', 'Ã‰': 'É', 'Ã“': 'Ó', 'Ã': 'Í', 'Ãš': 'Ú', 'Ã': 'Á',
        'Ã€': 'À', 'Ã‚': 'Â', 'Ãƒ': 'Ã', 'Ã„': 'Ä', 'Ã…': 'Å',
        'Ã‡': 'Ç', 'Ãˆ': 'È', 'ÃŠ': 'Ê', 'Ã‹': 'Ë', 'ÃŒ': 'Ì',
        'ÃŽ': 'Î', 'Ã’': 'Ò', 'Ã”': 'Ô', 'Ã•': 'Õ', 'Ã–': 'Ö',
        'Ã—': '×', 'Ã˜': 'Ø', 'Ã™': 'Ù', 'Ã›': 'Û', 'Ãœ': 'Ü',
        'Ãž': 'Þ', 'ÃŸ': 'ß'
    }
    
    if isinstance(text, str):
        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)
    return text

def select_and_fix_csv():
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
    
    try:
        # Detectar codificación del archivo
        encoding = detect_encoding(file_path)
        print(f"Codificación detectada: {encoding}")
        
        # Leer el archivo CSV
        df = pd.read_csv(file_path, encoding=encoding)
        
        # Aplicar corrección a todas las columnas de texto
        for col in df.columns:
            if df[col].dtype == object:  # Si es texto
                df[col] = df[col].apply(fix_encoding)
        
        # Generar nombre para el archivo corregido
        dir_name, file_name = os.path.split(file_path)
        name, ext = os.path.splitext(file_name)
        output_file = os.path.join(dir_name, f"{name}_corregido{ext}")
        
        # Guardar el archivo corregido
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"\nArchivo corregido guardado en:")
        print(output_file)
        
        return output_file
    
    except Exception as e:
        print(f"\nError al procesar el archivo: {str(e)}")
        return None

if __name__ == "__main__":
    select_and_fix_csv()