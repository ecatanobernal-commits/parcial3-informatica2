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

        elif menu == 4:
            estudio = elegir_estudio(sis)
            if estudio != False:
                grados = validar_decimal("Ángulo de inclinación en grados: ")
                MR, imagen = estudio.corregirAngulo(grados)
                print("Matriz de rotación:\n", MR)
                print("Gráfica guardada como:", imagen)

        elif menu == 5:
            estudio = elegir_estudio(sis)
            if estudio != False:
                n = estudio.verVolumen().shape[2]
                indice = validar_entero(f"Corte a ecualizar (0 a {n - 1}): ", 0, n - 1)
                grafico, imagen = estudio.ecualizar(indice)
                print("Gráfica guardada como:", grafico)
                print("Imagen guardada como:", imagen)

        elif menu == 6:
            estudio = elegir_estudio(sis)
            if estudio != False:
                print("1- Transversal\n2- Coronal\n3- Sagital")
                plano = validar_entero("Plano: ", 1, 3)
                n = estudio.tamanoPlano(plano)
                indice = validar_entero(f"Corte (0 a {n - 1}): ", 0, n - 1)
                umbral = validar_entero("Umbral (1 a 254, o 0 para Otsu): ", 0, 254)
                tam_kernel = validar_entero("Tamaño del kernel (ej. 3, 5, 7): ", 1, 51)
                iteraciones = validar_entero("Número de iteraciones: ", 1, 20)
                candidatos = estudio.segmentar(plano, indice, umbral, tam_kernel, iteraciones)
                if len(candidatos) == 0:
                    print("No se encontraron huecos internos, pruebe otro umbral o kernel")
                else:
                    numeros = []
                    print("Elemento\tÁrea (mm²)")
                    for num, area in candidatos:
                        print(f"{num}\t\t{area:.1f}")
                        numeros.append(num)
                    E = validar_entero("Número del elemento de interés: ", 1, max(numeros))
                    while E not in numeros:
                        print("Ese número no está en la lista")
                        E = validar_entero("Número del elemento de interés: ", 1, max(numeros))
                    area, grafico, imagen = estudio.marcarElemento(E)
                    print("Gráfica guardada como:", grafico)
                    print("Imagen guardada como:", imagen)

        elif menu == 7:
            texto = input("Texto a buscar: ")
            print("Encontrados:", sis.buscar(texto))

        elif menu == 8:
            print("Hasta luego")
            break


if __name__ == "__main__":
    main()
