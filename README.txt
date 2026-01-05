excela_csv: Primero, ejecuta este archivo para convertir los datos de un archivo Excel a formato CSV.
corregir_separadores: Después, ejecuta este archivo para asegurarte de que los separadores en el archivo CSV sean correctos (por ejemplo, cambiar comas por punto y coma si es necesario).
convertir_a_entero: Finalmente, ejecuta este archivo para convertir las columnas necesarias del CSV a valores enteros.
Este orden garantiza que los datos estén correctamente formateados y listos para su procesamiento posterior.

--------------------------------------------------------------------------------------------------------
DESCRIPCIÓN "buscar version 2.py"

Este código implementa un sistema interactivo en consola para analizar postulaciones universitarias usando archivos CSV. Aquí tienes un resumen de su funcionamiento:

Carga de datos:

Lee un archivo CSV con información de universidades y carreras.
Permite al usuario seleccionar una universidad y luego una carrera específica.
Carga y filtrado de postulaciones:

Solicita la ruta de un archivo CSV con postulaciones.
Filtra los registros por la carrera seleccionada.
Menú de análisis:

Ofrece opciones como mostrar registros, filtrar por columna, ver estadísticas, buscar por ID, aplicar filtros múltiples, ver valores únicos, ordenar datos, buscar estudiantes en otros archivos, copiar IDs al portapapeles, y cambiar universidad/carrera.
Interactividad:

Utiliza input() para recibir opciones y parámetros del usuario.
Muestra resultados en formato tabla usando tabulate.
Exportación y copia:

Permite exportar resultados filtrados/ordenados a un nuevo CSV.
Copia IDs al portapapeles usando pyperclip.
Puntos clave:

El flujo es completamente interactivo y guiado por menú.
Utiliza pandas para manipulación de datos.
El usuario puede realizar análisis exploratorio y filtrado avanzado sin salir del programa.

------------------------------------------------------------------------------------------------------------
El "buscar puntajes version 4.py" funciona corrrectamente, los otros se eliminan
