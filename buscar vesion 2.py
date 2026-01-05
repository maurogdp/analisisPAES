import pandas as pd
from tabulate import tabulate
import pyperclip  # Para copiar al portapapeles

def cargar_datos_universidades():
    """Carga el archivo CSV con información de universidades y carreras"""
    try:
        return pd.read_csv("analisis PAES\\PROCESO-DE-ADMISIÓN-2025-POSTULACIÓN-19-01-2025T23-38-41 (2)\\PostulaciónySelección_Admisión2025\\Libro_CódigosADM2025_ArchivoD_Anexo -  Oferta académica_corregido.csv")
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
    print("7. Ordenar datos")
    print("8. Buscar estudiante en archivo específico")
    print("9. Copiar IDs de resultados actuales")
    print("10. Cambiar universidad/carrera")
    print("11. Salir")

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
    return df

def ordenar_datos(df):
    """Permite ordenar los datos por una columna específica con control de visualización"""
    print("\nColumnas disponibles para ordenar:", ', '.join(df.columns))
    columna = input("Ingrese la columna para ordenar: ")
    
    if columna not in df.columns:
        print("Columna no válida")
        return df
    
    orden = input("Orden (asc/desc): ").lower()
    try:
        if orden == 'asc':
            df_ordenado = df.sort_values(by=columna)
        elif orden == 'desc':
            df_ordenado = df.sort_values(by=columna, ascending=False)
        else:
            print("Orden no reconocido. Use 'asc' o 'desc'")
            return df
        
        print(f"\nDatos ordenados por {columna} ({orden}):")
        
        while True:
            try:
                num_registros = int(input("\n¿Cuántos registros deseas ver? (10-100, 0=todos, -1=exportar): "))
                
                if num_registros == 0:
                    print(tabulate(df_ordenado, headers='keys', tablefmt='psql', showindex=False))
                elif num_registros > 0:
                    print(tabulate(df_ordenado.head(num_registros), headers='keys', tablefmt='psql', showindex=False))
                elif num_registros == -1:
                    archivo_salida = input("Ingrese nombre del archivo para exportar (ej: resultados_ordenados.csv): ")
                    df_ordenado.to_csv(archivo_salida, index=False)
                    print(f"Datos exportados a {archivo_salida}")
                else:
                    print("Número no válido")
                    continue
                
                break  # Salir del bucle si todo salió bien
                
            except ValueError:
                print("Por favor ingrese un número válido")
        
        return df_ordenado
    
    except Exception as e:
        print(f"Error al ordenar: {e}")
        return df

def buscar_estudiante_archivo():
    """Busca un estudiante en un archivo específico por su ID"""
    archivo = input("\nIngrese la ruta del archivo donde buscar: ")
    try:
        df_estudiantes = pd.read_csv(archivo)
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return
    
    id_buscar = input("Ingrese el ID del estudiante a buscar: ")
    resultados = df_estudiantes[df_estudiantes['ID_aux'].str.contains(id_buscar, case=False)]
    
    if len(resultados) > 0:
        print(f"\nResultados para ID que contiene '{id_buscar}':")
        print(tabulate(resultados, headers='keys', tablefmt='psql', showindex=False))
    else:
        print("No se encontraron resultados para ese ID")

def copiar_ids_resultados(df):
    """Copia los IDs de los estudiantes filtrados al portapapeles"""
    if df is None or len(df) == 0:
        print("No hay resultados actuales para copiar")
        return
    
    ids = df['ID_aux'].unique()
    ids_texto = "\n".join(ids)
    
    try:
        pyperclip.copy(ids_texto)
        print(f"\nSe copiaron {len(ids)} IDs al portapapeles:")
        print(ids_texto[:200] + ("..." if len(ids_texto) > 200 else ""))
        print("\nPuedes pegarlos directamente en tu próximo análisis")
    except Exception as e:
        print(f"\nNo se pudo copiar al portapapeles. Instala pyperclip con: pip install pyperclip")
        print("IDs para copiar manualmente:")
        print(ids_texto)

def main():
    print("Sistema de Análisis de Postulaciones Universitarias")
    
    # Cargar datos de universidades y carreras
    df_universidades = cargar_datos_universidades()
    if df_universidades is None:
        return
    
    # Selección inicial de universidad y carrera
    codigo_uni = seleccionar_universidad(df_universidades)
    codigo_carrera = seleccionar_carrera(df_universidades, codigo_uni)
    
    # Preguntar si se desea usar el archivo de postulaciones por defecto
    usar_por_defecto = input("\n¿Desea utilizar el archivo de postulaciones por defecto? (s/n): ").strip().lower()
    if usar_por_defecto == 's':
        archivo_postulaciones = "analisis PAES\\PROCESO-DE-ADMISIÓN-2025-POSTULACIÓN-19-01-2025T23-38-41 (2)\\PostulaciónySelección_Admisión2025\\ArchivoD_Adm2025_corregido_final.csv"
        print(f"Usando archivo por defecto: {archivo_postulaciones}")
    else:
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
            df = filtrar_por_columna(df)
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
            print("\nCrear filtro múltiple (ejemplo: (ESTADO_PREF == 24) & (TIPO_PREF == 'REGULAR')")
            print("Columnas disponibles:", ', '.join(df.columns))
            consulta = input("Ingrese la condición de filtro (usar sintaxis de pandas): ")
            try:
                filtrado = df.query(consulta)
                if len(filtrado) > 0:
                    print(f"\nResultados ({len(filtrado)} registros):")
                    print(tabulate(filtrado, headers='keys', tablefmt='psql', showindex=False))
                    df = filtrado
                else:
                    print("No se encontraron resultados con esos criterios")
            except Exception as e:
                print(f"Error en la consulta: {e}")
        elif opcion == '6':
            columna = input("Ingrese el nombre de la columna para ver valores únicos: ")
            if columna in df.columns:
                print(f"\nValores únicos en {columna}:")
                print(df[columna].unique())
            else:
                print("Columna no válida")
        elif opcion == '7':
            df = ordenar_datos(df)
        elif opcion == '8':
            buscar_estudiante_archivo()
        elif opcion == '9':
            copiar_ids_resultados(df)
        elif opcion == '10':
            # Volver a seleccionar universidad y carrera
            codigo_uni = seleccionar_universidad(df_universidades)
            codigo_carrera = seleccionar_carrera(df_universidades, codigo_uni)
            df = cargar_y_analizar_csv(archivo_postulaciones, codigo_carrera)
        elif opcion == '11':
            print("Saliendo del programa...")
            break
        else:
            print("Opción no válida")

if __name__ == "__main__":
    main()
