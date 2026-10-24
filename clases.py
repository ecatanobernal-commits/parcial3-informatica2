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
