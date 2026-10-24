from clases import *


def cargar_estudio(sis, nombre, carpeta):
    if sis.existe(nombre):
        print("Ya existe un estudio con ese nombre")
        return
    try:
        estudio = EstudioDicom(nombre, carpeta)
    except FileNotFoundError:
        print("No se encontró la carpeta o no tiene archivos .dcm, revise la ruta")
        return
    sis.ingresarEstudio(estudio)
    print(estudio)


def elegir_estudio(sis):
    print("Estudios cargados:", sis.verNombres())
    estudio = sis.verEstudio(input("Nombre del estudio: "))
    if estudio == False:
        print("Ese estudio no está cargado")
    return estudio


def main():
    # Trabajar desde la carpeta de este archivo (ahí deben estar los datos)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sis = Sistema()
    while True:
        print("\n>>>>> MENU PRINCIPAL <<<<<")
        print("1- Cargar carpeta DICOM")
        print("2- Cargar los estudios de clase (PPMI, Sarcoma y T2)")
        print("3- Ver información de un estudio")
        print("4- Corregir ángulo (rotación)")
        print("5- Normalizar y ecualizar un corte")
        print("6- Segmentar elemento de interés (umbral, apertura y área)")
        print("7- Buscar estudio")
        print("8- Salir")
        menu = validar_entero("--> ", 1, 8)

        if menu == 1:
            nombre = input("Nombre para identificar el estudio (ej. t2): ")
            carpeta = input("Ruta de la carpeta (ej. datos/T2): ")
            cargar_estudio(sis, nombre, carpeta)

        elif menu == 2:
            cargar_estudio(sis, "ppmi", "datos/PPMI/3105/MPRAGE_GRAPPA")
            cargar_estudio(sis, "sarcoma", "datos/Sarcoma/img1")
            cargar_estudio(sis, "t2", "datos/T2")

        elif menu == 3:
            estudio = elegir_estudio(sis)
            if estudio != False:
                print(estudio)

        elif menu == 8:
            print("Hasta luego")
            break


if __name__ == "__main__":
    main()
