import fnmatch
import os

import pandas as pd
import pyperclip
from tabulate import tabulate

PATRONES_PONDERACIONES = [
    "Libro_CódigosADM*_ArchivoD_Anexo -  Oferta académica_corregido.csv",
    "Libro_CódigosADM*_ArchivoD_Anexo - Oferta académica_corregido.csv",
]
PATRONES_RENDICION = [
    "ArchivoC_Adm*_corregido_final.csv",
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

# --------------------------
# Funciones de ponderación (MODIFICADAS para nueva estructura)
# --------------------------
def cargar_ponderaciones(archivo_ponderaciones=None):
    """Carga el archivo con las ponderaciones de carreras"""
    if not archivo_ponderaciones:
        archivo_ponderaciones = seleccionar_archivo_csv(
            "ponderaciones",
            PATRONES_PONDERACIONES
        )

    if not archivo_ponderaciones:
        print("No se seleccionó un archivo de ponderaciones.")
        return None

    try:
        return pd.read_csv(archivo_ponderaciones, encoding='utf-8')
    except Exception as e:
        print(f"Error al cargar ponderaciones: {e}")
        return None

def seleccionar_universidad(df_ponderaciones):
    """Permite al usuario seleccionar una universidad"""
    universidades = df_ponderaciones[['UNI_CODIGO', 'NOMBRE_UNIVERSIDAD']].drop_duplicates()
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

def seleccionar_carrera(df_ponderaciones, codigo_uni):
    """Permite al usuario seleccionar una carrera"""
    carreras = df_ponderaciones[df_ponderaciones['UNI_CODIGO'] == codigo_uni][['CODIGO_CARRERA', 'NOMBRE_CARRERA']]
    print("\nCarreras disponibles:")
    print(tabulate(carreras, headers=['Código', 'Nombre'], tablefmt='psql', showindex=False))
    
    while True:
        try:
            codigo = int(input("\nIngrese el código de la carrera: "))
            nombre_carrera = carreras[carreras['CODIGO_CARRERA'] == codigo]['NOMBRE_CARRERA'].values[0]
            print(f"\nHas seleccionado: {nombre_carrera}")
            return codigo, nombre_carrera
        except (ValueError, IndexError):
            print("Código no válido. Intente nuevamente.")

def calcular_ponderacion(df, df_ponderaciones, codigo_uni, codigo_carrera):
    """Calcula el puntaje ponderado para cada estudiante según nueva estructura"""
    # Obtener ponderaciones de la carrera seleccionada
    ponderacion = df_ponderaciones[
        (df_ponderaciones['UNI_CODIGO'] == codigo_uni) & 
        (df_ponderaciones['CODIGO_CARRERA'] == codigo_carrera)
    ].iloc[0]
    
    # Verificar y limpiar datos antes del cálculo
    columnas_requeridas = ['PTJE_NEM', 'PTJE_RANKING', 'MAX_CLEC', 'MAX_MATE1']
    for col in columnas_requeridas:
        if col not in df.columns:
            print(f"Error: Columna requerida {col} no encontrada")
            return df
        
        # Convertir a numérico y reemplazar valores no numéricos
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Columnas opcionales
    columnas_opcionales = {
        'MAX_MATE2': '%_MATE2',
        'MAX_HCSOC': '%_HYCS', 
        'MAX_CIEN': '%_CIEN'
    }
    
    for col, ponder in columnas_opcionales.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0  # Si la columna no existe, usar 0
    # #-------------------------------------------------
    # # Calcular componentes por separado
    # df['NEM_POND'] = df['PTJE_NEM'] * ponderacion['%_NOTAS'] / 100
    # df['RANK_POND'] = df['PTJE_RANKING'] * ponderacion['%_Ranking'] / 100
    # df['LENG_POND'] = df['MAX_CLEC'] * ponderacion['%_LENG'] / 100
    # df['MATE1_POND'] = df['MAX_MATE1'] * ponderacion['%_MATE1'] / 100
    # df['MATE2_POND'] = df['MAX_MATE2'].fillna(0) * ponderacion.get('%_MATE2', 0) / 100
    # df['HCSOC_POND'] = df['MAX_HCSOC'].fillna(0) * ponderacion.get('%_HYCS', 0) / 100
    # df['CIEN_POND'] = df['MAX_CIEN'].fillna(0) * ponderacion.get('%_CIEN', 0) / 100

    # # Suma final
    # df['PUNTAJE_PONDERADO'] = (df['NEM_POND'] + df['RANK_POND'] + df['LENG_POND'] + 
    #                           df['MATE1_POND'] + df['MATE2_POND'].fillna(0) + df['HCSOC_POND'].fillna(0) + 
    #                           df['CIEN_POND'].fillna(0))
    
    # # Mostrar verificación para los primeros 5 registros
    # print("\nVerificación de cálculo (primeros 5 registros):")
    # columnas_verificar = ['PTJE_NEM', 'NEM_POND', 'PTJE_RANKING', 'RANK_POND', 
    #                      'MAX_CLEC', 'LENG_POND', 'MAX_MATE1', 'MATE1_POND', 'MAX_CIEN', 'CIEN_POND', 'MAX_HCSOC', 'HCSOC_POND', 'MAX_MATE2', 'MATE2_POND',
    #                      'PUNTAJE_PONDERADO']
    
    # print(tabulate(df[columnas_verificar].head(), headers='keys', tablefmt='psql'))

    ##--------------------------------------------------


    #Calcular puntaje ponderado
    df['PUNTAJE_PONDERADO'] = (
        (df['PTJE_NEM'] * ponderacion['%_NOTAS'] / 100) +
        (df['PTJE_RANKING'] * ponderacion['%_Ranking'] / 100) +
        (df['MAX_CLEC'] * ponderacion['%_LENG'] / 100) +
        (df['MAX_MATE1'] * ponderacion['%_MATE1'] / 100) +
        (df['MAX_MATE2'] * ponderacion.get('%_MATE2', 0) / 100).fillna(0) +
        (df['MAX_HCSOC'] * ponderacion.get('%_HYCS', 0) / 100).fillna(0) +
        (df['MAX_CIEN'] * ponderacion.get('%_CIEN', 0) / 100).fillna(0)
    )
    
    # Agregar información de la carrera seleccionada
    df['UNIVERSIDAD'] = ponderacion['NOMBRE_UNIVERSIDAD']
    df['CARRERA'] = ponderacion['NOMBRE_CARRERA']
    df['COD_CARRERA'] = ponderacion['CODIGO_CARRERA']
    
    return df
# --------------------------
# Funciones originales (se mantienen igual)
# --------------------------
def procesar_dataframe(df):
    """Filtra columnas y calcula los máximos según los requisitos"""
    # Columnas a mantener directamente
    columnas_base = [
        'ID_aux',
        'PROMEDIO_NOTAS',
        'PORC_SUP_NOTAS',
        'PTJE_NEM',
        'PTJE_RANKING'
    ]
    
    # Grupos de columnas para calcular el máximo
    grupos_columnas = {
        'MAX_CLEC': ['CLEC_REG_ACTUAL', 'CLEC_INV_ACTUAL', 'CLEC_REG_ANTERIOR', 'CLEC_INV_ANTERIOR'],
        'MAX_MATE1': ['MATE1_REG_ACTUAL', 'MATE1_INV_ACTUAL', 'MATE1_REG_ANTERIOR', 'MATE1_INV_ANTERIOR'],
        'MAX_MATE2': ['MATE2_REG_ACTUAL', 'MATE2_INV_ACTUAL', 'MATE2_REG_ANTERIOR', 'MATE2_INV_ANTERIOR'],
        'MAX_HCSOC': ['HCSOC_REG_ACTUAL', 'HCSOC_INV_ACTUAL', 'HCSOC_REG_ANTERIOR', 'HCSOC_INV_ANTERIOR'],
        'MAX_CIEN': ['CIEN_REG_ACTUAL', 'CIEN_INV_ACTUAL', 'CIEN_REG_ANTERIOR', 'CIEN_INV_ANTERIOR']
    }
    
    # Verificar que las columnas base existan
    for col in columnas_base:
        if col not in df.columns:
            print(f"Advertencia: Columna {col} no encontrada en el archivo")
            df[col] = None
    
    # Crear nuevo DataFrame con las columnas base
    df_procesado = df[columnas_base].copy()
    
    # Calcular los máximos para cada grupo
    for nuevo_nombre, columnas_grupo in grupos_columnas.items():
        columnas_existentes = [col for col in columnas_grupo if col in df.columns]
        if columnas_existentes:
            df_procesado[nuevo_nombre] = df[columnas_existentes].max(axis=1)
        else:
            df_procesado[nuevo_nombre] = None
    
    # Obtener el módulo de la prueba de ciencias
    modulos = ['MODULO_REG_ACTUAL', 'MODULO_INV_ACTUAL', 'MODULO_REG_ANTERIOR', 'MODULO_INV_ANTERIOR']
    for mod in modulos:
        if mod in df.columns:
            df_procesado['MODULO_CIENCIAS'] = df[mod]
            break
    else:
        df_procesado['MODULO_CIENCIAS'] = None
    
    return df_procesado

def filtrar_por_columnas(df):
    """Permite filtrar por una o múltiples columnas"""
    print("\nColumnas disponibles para filtrar:")
    print(", ".join(df.columns))
    
    condiciones = []
    while True:
        columna = input("\nIngrese columna para filtrar (o 'fin' para terminar): ")
        if columna.lower() == 'fin':
            break
        
        if columna not in df.columns:
            print("Columna no válida")
            continue
        
        valor = input(f"Ingrese valor para filtrar en {columna} (o 'mostrar' para ver valores únicos): ")
        
        if valor.lower() == 'mostrar':
            print(f"\nValores únicos en {columna}:")
            print(df[columna].unique())
            continue
        
        try:
            # Intentar convertir a número si es posible
            if pd.api.types.is_numeric_dtype(df[columna]):
                valor = float(valor)
        except ValueError:
            pass
        
        operador = input("Ingrese operador (==, !=, >, <, >=, <=, contiene): ")
        
        if operador == '==':
            condicion = (df[columna] == valor)
        elif operador == '!=':
            condicion = (df[columna] != valor)
        elif operador == '>':
            condicion = (df[columna] > valor)
        elif operador == '<':
            condicion = (df[columna] < valor)
        elif operador == '>=':
            condicion = (df[columna] >= valor)
        elif operador == '<=':
            condicion = (df[columna] <= valor)
        elif operador == 'contiene':
            condicion = df[columna].astype(str).str.contains(str(valor), case=False)
        else:
            print("Operador no válido")
            continue
        
        condiciones.append(condicion)
    
    if condiciones:
        # Combinar todas las condiciones con AND
        condicion_final = condiciones[0]
        for cond in condiciones[1:]:
            condicion_final &= cond
        
        df_filtrado = df[condicion_final]
        print(f"\nResultados después de filtrar ({len(df_filtrado)} registros):")
        return df_filtrado
    else:
        print("No se aplicaron filtros")
        return df

def ordenar_dataframe(df):
    """Permite ordenar el DataFrame por una o múltiples columnas"""
    print("\nColumnas disponibles para ordenar:")
    print(", ".join(df.columns))
    
    columnas_orden = []
    while True:
        columna = input("\nIngrese columna para ordenar (o 'fin' para terminar): ")
        if columna.lower() == 'fin':
            break
        
        if columna not in df.columns:
            print("Columna no válida")
            continue
        
        orden = input(f"Orden para {columna} (asc/desc): ").lower()
        if orden not in ['asc', 'desc']:
            print("Orden no válido, usando 'asc' por defecto")
            orden = 'asc'
        
        columnas_orden.append((columna, orden == 'asc'))
    
    if columnas_orden:
        # Aplicar ordenamiento múltiple
        columnas = [col for col, _ in columnas_orden]
        ascending = [asc for _, asc in columnas_orden]
        
        try:
            df_ordenado = df.sort_values(by=columnas, ascending=ascending)
            print("\nDatos ordenados:")
            return df_ordenado
        except Exception as e:
            print(f"Error al ordenar: {e}")
            return df
    else:
        print("No se aplicó ordenamiento")
        return df

def ingresar_puntaje_simulado(df, codigo_uni, codigo_carrera):
    """Permite ingresar puntajes simulados para cada prueba y agregarlos como un nuevo registro con ID 'SIMULADO'"""
    if df.empty:
        print("No hay datos para simular. Primero cargue un archivo con resultados.")
        return df
    
    # Determinar qué columnas de puntaje existen
    columnas_puntaje = [
        'PTJE_NEM', 'PTJE_RANKING', 'MAX_CLEC', 'MAX_MATE1', 'MAX_MATE2', 'MAX_HCSOC', 'MAX_CIEN'
    ]
    columnas_disponibles = [col for col in columnas_puntaje if col in df.columns]
    
    if not columnas_disponibles:
        print("No se encontraron columnas de puntaje para simular")
        return df
    
    puntajes_simulados = {}
    print("\nIngrese los puntajes simulados para cada prueba (deje vacío para 0):")
    for col in columnas_disponibles:
        while True:
            try:
                valor = input(f"{col} (rango típico {df[col].min()}-{df[col].max()}): ")
                if valor.strip() == "":
                    puntajes_simulados[col] = 0
                    break
                valor_num = float(valor)
                if valor_num < 0:
                    print("El puntaje no puede ser negativo")
                    continue
                puntajes_simulados[col] = valor_num
                break
            except ValueError:
                print("Valor no válido, intente nuevamente")
    
    # Crear un nuevo registro simulado
    nuevo_registro = {col: 0 for col in df.columns}  # Inicializar todas las columnas con 0
    nuevo_registro.update(puntajes_simulados)
    nuevo_registro['ID_aux'] = 'SIMULADO'
    
    # Copiar valores de contexto de la primera fila
    for col in ['UNIVERSIDAD', 'CARRERA', 'COD_CARRERA']:
        if col in df.columns:
            nuevo_registro[col] = df.iloc[0][col]
    
    # Calcular el puntaje ponderado si hay contexto
    if 'UNIVERSIDAD' in nuevo_registro and 'COD_CARRERA' in nuevo_registro:
        df_ponderaciones = cargar_ponderaciones()
        if df_ponderaciones is not None:
            try:
                df_simulado = pd.DataFrame([nuevo_registro])
                df_simulado = calcular_ponderacion(df_simulado, df_ponderaciones, codigo_uni, codigo_carrera)
                nuevo_registro = df_simulado.iloc[0].to_dict()
            except Exception as e:
                print(f"Error al calcular ponderación para simulación: {e}")
    
    # Concatenar el registro simulado al DataFrame original
    df_simulado = pd.DataFrame([nuevo_registro])
    df = pd.concat([df, df_simulado], ignore_index=True)
    
    print("\nPuntaje simulado agregado correctamente:")
    print(tabulate(df_simulado, headers='keys', tablefmt='psql', showindex=False))
    return df


def buscar_ids_en_csv():
    """Busca IDs copiados en el portapapeles dentro de un archivo CSV"""
    try:
        # Obtener IDs del portapapeles
        ids_texto = pyperclip.paste()
        ids = [id_.strip() for id_ in ids_texto.split('\n') if id_.strip()]
        
        if not ids:
            print("No se encontraron IDs en el portapapeles")
            return
        
        print(f"\nIDs encontrados en portapapeles ({len(ids)}):")
        print('\n'.join(ids[:5]) + ('\n...' if len(ids) > 5 else ''))
        
        archivo_csv = seleccionar_archivo_csv(
            "postulaciones/rendición",
            PATRONES_RENDICION
        )


        # # Solicitar archivo CSV
        # archivo_csv = input("\nIngrese la ruta del archivo CSV donde buscar: ")
        
        # Cargar el archivo CSV
        try:
            df = pd.read_csv(archivo_csv)
            print(f"\nArchivo cargado. Total de registros: {len(df)}")
            
            if 'ID_aux' not in df.columns:
                print("El archivo no contiene la columna 'ID_aux'")
                return
            
            # Filtrar por los IDs
            resultados = df[df['ID_aux'].isin(ids)].copy()
            
            if len(resultados) > 0:
                print(f"\nResultados encontrados ({len(resultados)}):")
                resultados_procesados = procesar_dataframe(resultados)
                
                # --- Cálculo de ponderación ---
                archivo_ponderaciones = seleccionar_archivo_csv(
                    "ponderaciones",
                    PATRONES_PONDERACIONES
                )
                df_ponderaciones = cargar_ponderaciones(archivo_ponderaciones)

                if df_ponderaciones is not None:
                    print("\nAhora seleccione universidad y carrera para calcular ponderación:")
                    codigo_uni = seleccionar_universidad(df_ponderaciones)
                    codigo_carrera, nombre_carrera = seleccionar_carrera(df_ponderaciones, codigo_uni)
                    
                    # Obtener ponderaciones antes de calcular
                    ponderacion = df_ponderaciones[
                        (df_ponderaciones['UNI_CODIGO'] == codigo_uni) & 
                        (df_ponderaciones['CODIGO_CARRERA'] == codigo_carrera)
                    ].iloc[0]
                    
                    resultados_procesados = calcular_ponderacion(
                        resultados_procesados, 
                        df_ponderaciones, 
                        codigo_uni, 
                        codigo_carrera
                    )
                    
                    print(f"\nPuntajes ponderados calculados para {nombre_carrera}")
                    
                    # Mostrar ponderaciones aplicadas
                    print("\nPonderaciones aplicadas:")
                    print(f"NEM: {ponderacion['%_NOTAS']}%")
                    print(f"Ranking: {ponderacion['%_Ranking']}%")
                    print(f"Lenguaje: {ponderacion['%_LENG']}%")
                    print(f"Matemática 1: {ponderacion['%_MATE1']}%")
                    if ponderacion.get('%_MATE2', 0) > 0:
                        print(f"Matemática 2: {ponderacion['%_MATE2']}%")
                    if ponderacion.get('%_HYCS', 0) > 0:
                        print(f"Historia/Ciencias Sociales: {ponderacion['%_HYCS']}%")
                    if ponderacion.get('%_CIEN', 0) > 0:
                        print(f"Ciencias: {ponderacion['%_CIEN']}%")
                # -------------------------------------
                
                # Menú de opciones para los resultados
                while True:
                    print("\nOpciones para los resultados:")
                    print("1. Mostrar resultados actuales")
                    print("2. Filtrar por columnas")
                    print("3. Ordenar resultados")
                    print("4. Exportar a CSV")
                    print("5. Agregar puntaje simulado")
                    print("6. Volver al menú principal")
                    
                    opcion = input("Seleccione una opción: ")
                    
                    if opcion == '1':
                        columnas_mostrar = [
                            'ID_aux', 'UNIVERSIDAD', 'CARRERA', 'COD_CARRERA',
                            'PROMEDIO_NOTAS', 'PTJE_NEM', 'PTJE_RANKING',
                            'MAX_CLEC', 'MAX_MATE1', 'MAX_MATE2', 'MAX_HCSOC', 'MAX_CIEN',
                            'PUNTAJE_PONDERADO'
                        ]
                        # Mostrar solo columnas existentes
                        columnas_existentes = [col for col in columnas_mostrar if col in resultados_procesados.columns]
                        print(tabulate(
                            resultados_procesados[columnas_existentes], 
                            headers='keys', 
                            tablefmt='psql', 
                            showindex=False
                        ))
                    elif opcion == '2':
                        resultados_procesados = filtrar_por_columnas(resultados_procesados)
                    elif opcion == '3':
                        resultados_procesados = ordenar_dataframe(resultados_procesados)
                    elif opcion == '4':
                        nombre_archivo = input("Ingrese nombre del archivo (ej: resultados.csv): ")
                        resultados_procesados.to_csv(nombre_archivo, index=False)
                        print(f"Resultados exportados a {nombre_archivo}")
                    elif opcion == '5':
                        resultados_procesados = ingresar_puntaje_simulado(resultados_procesados, codigo_uni, codigo_carrera)
                    elif opcion == '6':
                        break
                    else:
                        print("Opción no válida")
            else:
                print("\nNo se encontraron coincidencias para los IDs proporcionados")
                
        except Exception as e:
            print(f"\nError al procesar el archivo CSV: {e}")
            
    except Exception as e:
        print(f"\nError al acceder al portapapeles: {e}")
def main():
    print("Sistema Avanzado de Análisis de Datos Universitarios")
    print("---------------------------------------------------")
    print("Funcionalidades:")
    print("- Buscar IDs específicos en archivos CSV")
    print("- Procesar datos según requisitos específicos")
    print("- Filtrar por múltiples columnas con diversos operadores")
    print("- Ordenar por múltiples columnas")
    print("- Exportar resultados a CSV\n")
    
    while True:
        print("\nMenú Principal:")
        print("1. Buscar IDs en archivo CSV")
        print("2. Salir")
        
        opcion = input("Seleccione una opción: ")
        
        if opcion == '1':
            buscar_ids_en_csv()
        elif opcion == '2':
            print("\nPrograma terminado")
            break
        else:
            print("Opción no válida")

if __name__ == "__main__":
    # Verificar si pyperclip está instalado
    try:
        import pyperclip
    except ImportError:
        print("\nAdvertencia: pyperclip no está instalado.")
        print("Puede instalarlo con: pip install pyperclip")
        print("El programa funcionará pero no podrá leer del portapapeles automáticamente.")
        pyperclip = None
    
    main()
