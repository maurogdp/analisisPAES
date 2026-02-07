import fnmatch
import os

import pandas as pd
from tabulate import tabulate

PATRONES_UNIVERSIDADES = [
    "Libro_CódigosADM*_ArchivoD_Anexo -  Oferta académica_corregido.csv",
    "Libro_CódigosADM*_ArchivoD_Anexo - Oferta académica_corregido.csv",
]

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
    print("0. Ingresar ruta manual")

    while True:
        seleccion = input("Seleccione un número: ").strip()
        if seleccion == "0":
            return None
        if seleccion.isdigit():
            indice = int(seleccion) - 1
            if 0 <= indice < len(archivos):
                return archivos[indice]
        print("Selección no válida. Intente nuevamente.")

def seleccionar_archivo_csv(titulo, patrones):
    archivos = buscar_archivos(os.getcwd(), patrones)
    archivo = seleccionar_archivo_desde_lista(archivos, titulo)
    if archivo:
        print(f"Usando archivo: {archivo}")
        return archivo

    return input(f"\nIngrese la ruta del archivo para {titulo}: ").strip()

def cargar_datos_universidades():
    """Carga el archivo CSV con información de universidades y carreras"""
    archivo_universidades = seleccionar_archivo_csv(
        "universidades y carreras",
        PATRONES_UNIVERSIDADES
    )
    if not archivo_universidades:
        print("No se seleccionó un archivo de universidades.")
        return None

    try:
        return pd.read_csv(archivo_universidades)
    except Exception as e:
        print(f"Error al cargar el archivo de universidades: {e}")
        return None

def seleccionar_universidad(df_universidades):
    """Permite al usuario seleccionar una universidad"""
    universidades = df_universidades[['UNI_CODIGO', 'NOMBRE_UNIVERSIDAD']].drop_duplicates()
    print("\nUniversidades disponibles:")
    print(tabulate(universidades, headers=['Código', 'Nombre'], tablefmt='psql', showindex=False))
    
    while True:
        try:
            codigo = int(input("\nIngrese el código de la universidad: "))
            nombre_uni = universidades[universidades['UNI_CODIGO'] == codigo]['NOMBRE_UNIVERSIDAD'].values[0]
            print(f"\nHas seleccionado: {nombre_uni}")
            return codigo
        except (ValueError, IndexError):
            print("Código no válido. Intente nuevamente.")

def seleccionar_carrera(df_universidades, codigo_uni):
    """Permite al usuario seleccionar una carrera de la universidad elegida"""
    carreras = df_universidades[df_universidades['UNI_CODIGO'] == codigo_uni][['CODIGO_CARRERA', 'NOMBRE_CARRERA']]
    print(f"\nCarreras disponibles en esta universidad:")
    print(tabulate(carreras, headers=['Código', 'Nombre de Carrera'], tablefmt='psql', showindex=False))
    
    while True:
        try:
            codigo_carrera = int(input("\nIngrese el código de la carrera: "))
            nombre_carrera = carreras[carreras['CODIGO_CARRERA'] == codigo_carrera]['NOMBRE_CARRERA'].values[0]
            print(f"\nHas seleccionado: {nombre_carrera}")
            return codigo_carrera
        except (ValueError, IndexError):
            print("Código no válido. Intente nuevamente.")

def cargar_y_analizar_csv(archivo, codigo_carrera=None):
    """Carga y analiza el archivo CSV de postulaciones, opcionalmente filtrando por carrera"""
    try:
        df = pd.read_csv(archivo)
        
        print("\nInformación básica del archivo:")
        print(f"Total de registros: {len(df)}")
        
        if codigo_carrera:
            df = df[df['COD_CARRERA_PREF'] == codigo_carrera]
            print(f"Registros filtrados por carrera {codigo_carrera}: {len(df)}")
        
        return df
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return None

def mostrar_menu_analisis():
    print("\nOpciones de análisis:")
    print("1. Mostrar primeros registros")
    print("2. Filtrar por columna específica")
    print("3. Estadísticas básicas")
    print("4. Buscar por ID específico")
    print("5. Filtrar por múltiples condiciones")
    print("6. Mostrar valores únicos en columna")
    print("7. Cambiar universidad/carrera")
    print("8. Salir")

def filtrar_por_columna(df):
    print("\nColumnas disponibles:", ', '.join(df.columns))
    columna = input("Ingrese el nombre de la columna para filtrar: ")
    
    if columna not in df.columns:
        print("Columna no válida")
        return
    
    valores_unicos = df[columna].unique()
    print(f"\nValores únicos en {columna}:")
    print(valores_unicos)
    
    valor = input(f"Ingrese el valor para filtrar (o 'todos' para mostrar todo): ")
    
    if valor.lower() != 'todos':
        try:
            if df[columna].dtype in ['int64', 'float64']:
                valor = float(valor)
            filtrado = df[df[columna] == valor]
        except ValueError:
            filtrado = df[df[columna].astype(str).str.contains(valor, case=False)]
        
        print(f"\nResultados filtrados por {columna} = {valor}:")
        print(tabulate(filtrado, headers='keys', tablefmt='psql', showindex=False))
    else:
        print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))

def filtrar_por_multiples_condiciones(df):
    print("\nCrear filtro múltiple")
    print("Columnas disponibles:", ', '.join(df.columns))
    
    while True:
        print("\nEjemplos válidos:")
        print("1. ESTADO_PREF == 24")
        print("2. (ESTADO_PREF == 24) & (TIPO_PREF == 'REGULAR')")
        print("3. (PTJE_PREF > 600) | (COD_CARRERA_PREF == 11020)")
        
        consulta = input("\nIngrese la condición (o 'salir' para volver): ")
        if consulta.lower() == 'salir':
            return
        
        try:
            filtrado = df.query(consulta)
            if len(filtrado) > 0:
                print(f"\nResultados ({len(filtrado)} registros):")
                print(tabulate(filtrado, headers='keys', tablefmt='psql', showindex=False))
                return
            else:
                print("No se encontraron resultados. ¿Desea intentar con otra condición?")
        except Exception as e:
            print(f"Error en la consulta: {e}")
            print("Posibles causas:")
            print("- Valores de texto deben ir entre comillas")
            print("- Use & para AND y | para OR")
            print("- Agrupe condiciones con paréntesis")

def main():
    print("Sistema de Análisis de Postulaciones Universitarias")
    
    # Cargar datos de universidades y carreras
    df_universidades = cargar_datos_universidades()
    if df_universidades is None:
        return
    
    # Selección inicial de universidad y carrera
    codigo_uni = seleccionar_universidad(df_universidades)
    codigo_carrera = seleccionar_carrera(df_universidades, codigo_uni)
    
    # Cargar archivo de postulaciones
    archivo_postulaciones = input("\nIngrese la ruta del archivo CSV de postulaciones: ")
    df = cargar_y_analizar_csv(archivo_postulaciones, codigo_carrera)
    if df is None:
        return
    
    # Menú principal
    while True:
        mostrar_menu_analisis()
        opcion = input("Seleccione una opción: ")
        
        if opcion == '1':
            print(tabulate(df.head(), headers='keys', tablefmt='psql', showindex=False))
        elif opcion == '2':
            filtrar_por_columna(df)
        elif opcion == '3':
            print("\nEstadísticas descriptivas:")
            print(df.describe(include='all'))
        elif opcion == '4':
            id_buscar = input("Ingrese el ID a buscar: ")
            resultados = df[df['ID_aux'].str.contains(id_buscar, case=False)]
            if len(resultados) > 0:
                print(f"\nResultados para ID que contiene '{id_buscar}':")
                print(tabulate(resultados, headers='keys', tablefmt='psql', showindex=False))
            else:
                print("No se encontraron resultados para ese ID")
        elif opcion == '5':
            filtrar_por_multiples_condiciones(df)
        elif opcion == '6':
            columna = input("Ingrese el nombre de la columna para ver valores únicos: ")
            if columna in df.columns:
                print(f"\nValores únicos en {columna}:")
                print(df[columna].unique())
            else:
                print("Columna no válida")
        elif opcion == '7':
            # Volver a seleccionar universidad y carrera
            codigo_uni = seleccionar_universidad(df_universidades)
            codigo_carrera = seleccionar_carrera(df_universidades, codigo_uni)
            df = cargar_y_analizar_csv(archivo_postulaciones, codigo_carrera)
        elif opcion == '8':
            print("Saliendo del programa...")
            break
        else:
            print("Opción no válida")

if __name__ == "__main__":
    main()
