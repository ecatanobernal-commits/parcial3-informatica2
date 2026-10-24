import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import cv2


def validar_entero(mensaje, minimo, maximo):
    while True:
        try:
            valor = int(input(mensaje))
            if valor >= minimo and valor <= maximo:
                return valor
            print(f"El número debe estar entre {minimo} y {maximo}")
        except ValueError:
            print("Debe ingresar un número entero")


def validar_decimal(mensaje):
    while True:
        try:
            return float(input(mensaje))
        except ValueError:
            print("Debe ingresar un número")


def normalizar(corte):
    # Lleva los valores del corte al rango 0-255 y los pasa a uint8
    img = np.array(corte, np.float32)
    img = img - np.min(img)
    if np.max(img) > 0:
        img = img * 255 / np.max(img)
    return img.astype(np.uint8)


class EstudioDicom:
    def __init__(self, nombre, carpeta):
        self.__nombre = nombre
        self.__carpeta = carpeta

        # Leer todos los .dcm de la carpeta
        archivos = []
        for f in os.listdir(carpeta):
            if f.lower().endswith(".dcm"):
                archivos.append(f)
        if len(archivos) == 0:
            raise FileNotFoundError("La carpeta no tiene archivos .dcm")

        datasets = []
        for f in archivos:
            datasets.append(pydicom.dcmread(os.path.join(carpeta, f)))

        # Ordenar los cortes por ImagePositionPatient en el eje que cambia
        posiciones = []
        for ds in datasets:
            posiciones.append([float(v) for v in ds.ImagePositionPatient])
        posiciones = np.array(posiciones)
        cambio = np.max(posiciones, axis=0) - np.min(posiciones, axis=0)
        self.__eje = int(np.argmax(cambio))      # 0 = x, 1 = y, 2 = z
        orden = np.argsort(posiciones[:, self.__eje])
        self.__datasets = []
        for i in orden:
            self.__datasets.append(datasets[i])

        # Información extraída del dataset
        ds = self.__datasets[0]
        self.__filas = int(ds.Rows)
        self.__columnas = int(ds.Columns)
        self.__paciente = str(ds.PatientName)
        self.__modalidad = str(ds.Modality)
        self.__voxel = (float(ds.PixelSpacing[0]),     # alto (mm)
                        float(ds.PixelSpacing[1]),     # largo (mm)
                        float(ds.SliceThickness))      # ancho (mm)

        # Matriz de píxeles original (filas, columnas, cortes)
        cortes = []
        for ds in self.__datasets:
            cortes.append(ds.pixel_array)
        self.__pixeles = np.stack(cortes, axis=-1)

        # Escalar cada corte al doble y reconstruir el volumen 3D
        cortes_dobles = []
        for ds in self.__datasets:
            cortes_dobles.append(cv2.resize(ds.pixel_array, None, fx=2, fy=2,
                                            interpolation=cv2.INTER_LINEAR))
        self.__volumen = np.stack(cortes_dobles, axis=-1)
        # Al duplicar el tamaño, cada píxel mide la mitad en mm
        self.__voxel_volumen = (self.__voxel[0] / 2, self.__voxel[1] / 2, self.__voxel[2])

        self.__volumen_corregido = None
        self.__angulo = 0

    def verNombre(self):
        return self.__nombre

    def verCarpeta(self):
        return self.__carpeta

    def verFilas(self):
        return self.__filas

    def verColumnas(self):
        return self.__columnas

    def verPaciente(self):
        return self.__paciente

    def verModalidad(self):
        return self.__modalidad

    def verVoxel(self):
        return self.__voxel

    def verPixeles(self):
        return self.__pixeles

    def verVolumen(self):
        return self.__volumen

    def verVolumenCorregido(self):
        return self.__volumen_corregido

    def verPlanoAdquisicion(self):
        if self.__eje == 0:
            return "sagital"
        elif self.__eje == 1:
            return "coronal"
        return "transversal"

    def __str__(self):
        return f"""
========== ESTUDIO DICOM: {self.__nombre} ==========
Carpeta:            {self.__carpeta}
Paciente:           {self.__paciente}
Modalidad:          {self.__modalidad}
Filas x columnas:   {self.__filas} x {self.__columnas}
Número de cortes:   {self.__pixeles.shape[2]}
Plano de los cortes: {self.verPlanoAdquisicion()}
Tamaño del vóxel (alto, largo, ancho): {self.__voxel} mm
Matriz de píxeles original: {self.__pixeles.shape} {self.__pixeles.dtype}
Volumen 3D (cortes al doble): {self.__volumen.shape}
Vóxel del volumen escalado:   {self.__voxel_volumen} mm
"""

    def tamanoPlano(self, plano):
        # Cantidad de cortes que hay en un plano: 1 transversal, 2 coronal, 3 sagital
        eje = self.__ejeDelPlano(plano)
        if eje == 2:
            return self.__volumen.shape[2]
        elif eje == 0:
            return self.__volumen.shape[0]
        return self.__volumen.shape[1]

    def __ejeDelPlano(self, plano):
        # Devuelve en qué eje del volumen se mueve cada plano.
        # El eje 2 del volumen es siempre el de los cortes originales.
        planos = {1: "transversal", 2: "coronal", 3: "sagital"}
        nombre = planos[plano]
        if nombre == self.verPlanoAdquisicion():
            return 2
        if self.__eje == 2:                  # cortes transversales
            if nombre == "coronal":
                return 0
            return 1
        if self.__eje == 0:                  # cortes sagitales
            if nombre == "coronal":
                return 1
            return 0
        if nombre == "transversal":          # cortes coronales
            return 0
        return 1

    def obtenerCorte(self, plano, indice):
        # Devuelve el corte, los mm de cada fila, los mm de cada columna y el nombre del plano
        planos = {1: "transversal", 2: "coronal", 3: "sagital"}
        alto, largo, ancho = self.__voxel_volumen
        eje = self.__ejeDelPlano(plano)
        if eje == 2:
            return self.__volumen[:, :, indice], alto, largo, planos[plano]
        if eje == 0:
            corte = self.__volumen[indice, :, :]
            mm_fila, mm_col = largo, ancho
        else:
            corte = self.__volumen[:, indice, :]
            mm_fila, mm_col = alto, ancho
        if self.__eje == 2:
            # los cortes quedan como columnas: se gira para dejar la cabeza arriba
            corte = np.rot90(corte)
            mm_fila, mm_col = mm_col, mm_fila
        return corte, mm_fila, mm_col, planos[plano]

    def corregirAngulo(self, grados):
        # Rota cada corte del volumen alrededor del centro de la imagen
        filas, columnas, n = self.__volumen.shape
        MR = cv2.getRotationMatrix2D((columnas / 2, filas / 2), grados, 1)
        rotados = []
        for k in range(n):
            rotados.append(cv2.warpAffine(self.__volumen[:, :, k], MR, (columnas, filas)))
        self.__volumen_corregido = np.stack(rotados, axis=-1)
        self.__angulo = grados

        k = n // 2
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(self.__volumen[:, :, k], cmap="gray")
        ax1.set_title(f"Corte {k} original")
        ax1.axis("off")
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(self.__volumen_corregido[:, :, k], cmap="gray")
        ax2.set_title(f"Corte {k} rotado {grados}°")
        ax2.axis("off")
        fig.tight_layout()
        nombre_imagen = f"{self.__nombre}_rotado_{grados}.png"
        fig.savefig(nombre_imagen)
        plt.show()
        return MR, nombre_imagen
