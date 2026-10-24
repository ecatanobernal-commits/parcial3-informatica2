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
