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

    def ecualizar(self, indice):
        # Trabaja sobre una copia para no modificar el volumen original
        copia = self.__volumen.copy()
        corte8 = normalizar(copia[:, :, indice])
        ecualizado = cv2.equalizeHist(corte8)

        fig = plt.figure(figsize=(14, 8))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(corte8, cmap="gray", vmin=0, vmax=255)
        ax1.set_title(f"Corte {indice} normalizado (uint8)")
        ax1.axis("off")
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(ecualizado, cmap="gray", vmin=0, vmax=255)
        ax2.set_title(f"Corte {indice} ecualizado")
        ax2.axis("off")
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.hist(corte8.ravel(), bins=256, range=(0, 256))
        ax3.set_title("Histograma normalizado")
        ax3.set_xlabel("Nivel de gris")
        ax3.set_ylabel("Número de píxeles")
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.hist(ecualizado.ravel(), bins=256, range=(0, 256), color="orange")
        ax4.set_title("Histograma ecualizado")
        ax4.set_xlabel("Nivel de gris")
        ax4.set_ylabel("Número de píxeles")
        fig.tight_layout()

        nombre_grafico = f"{self.__nombre}_ecualizacion_corte{indice}.png"
        nombre_imagen = f"{self.__nombre}_ecualizado_corte{indice}.png"
        fig.savefig(nombre_grafico)
        cv2.imwrite(nombre_imagen, ecualizado)
        plt.show()
        return nombre_grafico, nombre_imagen

    def segmentar(self, plano, indice, umbral, tam_kernel, iteraciones):
        corte, mm_fila, mm_col, nombre_plano = self.obtenerCorte(plano, indice)

        # aspect: cuántos mm mide una fila comparado con una columna.
        # Se corrige estirando el corte para que cada píxel quede cuadrado,
        # así la imagen, el círculo y el área quedan con la proporción real.
        aspecto = mm_fila / mm_col
        if mm_fila > mm_col:
            corte = cv2.resize(corte, None, fx=1, fy=aspecto)
            mm_fila = mm_col
        elif mm_col > mm_fila:
            corte = cv2.resize(corte, None, fx=1 / aspecto, fy=1)
            mm_col = mm_fila
        corte8 = normalizar(corte)

        # Binarizar: los huecos (zonas oscuras) quedan en blanco
        if umbral == 0:
            umb, imgB = cv2.threshold(corte8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            umb, imgB = cv2.threshold(corte8, umbral, 255, cv2.THRESH_BINARY_INV)

        # Apertura: quita los puntos blancos pequeños
        kernel = np.ones((tam_kernel, tam_kernel), np.uint8)
        imgA = cv2.morphologyEx(imgB, cv2.MORPH_OPEN, kernel, iterations=iteraciones)

        # Etiquetar los objetos blancos
        elem, mask = cv2.connectedComponents(imgA)

        # Los objetos que tocan el borde son el aire de afuera, no huecos internos
        borde = np.concatenate((mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]))
        candidatos = []
        for i in range(1, elem):
            if i not in borde:
                area_mm2 = np.sum(mask == i) * mm_fila * mm_col
                candidatos.append((i, area_mm2))

        # Se guarda todo para poder marcar después el elemento que elija el usuario
        self.__seg_corte = corte8
        self.__seg_binaria = imgA
        self.__seg_mask = mask
        self.__seg_mm = (mm_fila, mm_col)
        self.__seg_nombre = f"{nombre_plano}{indice}"
        self.__seg_parametros = (tam_kernel, iteraciones)

        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(corte8, cmap="gray")
        ax1.set_title(f"Corte {nombre_plano} {indice} (aspect {aspecto:.2f} corregido)")
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(imgA, cmap="gray")
        ax2.set_title(f"Umbral {umb:.0f} + apertura (kernel {tam_kernel}, {iteraciones} it.)")
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.imshow(mask, cmap="nipy_spectral")
        for num, area in candidatos:
            filas_obj, cols_obj = np.where(mask == num)
            ax3.text(np.mean(cols_obj), np.mean(filas_obj), str(num), color="white", fontsize=9)
        ax3.set_title(f"Elementos encontrados: {len(candidatos)}")
        fig.tight_layout()
        plt.show()

        print(f"Umbral usado: {umb}")
        print(f"Objetos (contando el fondo): {elem}")
        return candidatos

    def marcarElemento(self, E):
        # Dibuja un círculo sobre el elemento E elegido y calcula su área
        mask = self.__seg_mask
        mm_fila, mm_col = self.__seg_mm
        tam_kernel, iteraciones = self.__seg_parametros

        area_px = np.sum(mask == E)
        area_mm2 = area_px * mm_fila * mm_col
        filas_obj, cols_obj = np.where(mask == E)
        y1, y2 = np.min(filas_obj), np.max(filas_obj)
        x1, x2 = np.min(cols_obj), np.max(cols_obj)
        alto_mm = (y2 - y1 + 1) * mm_fila
        ancho_mm = (x2 - x1 + 1) * mm_col
        centro = (int(np.mean(cols_obj)), int(np.mean(filas_obj)))      # (x, y)
        radio = int(max(x2 - x1, y2 - y1) / 2) + 10

        # Corte normalizado (uint8) pasado a RGB para dibujar a color
        rgb = cv2.cvtColor(self.__seg_corte, cv2.COLOR_GRAY2RGB)
        grosor = max(1, rgb.shape[1] // 300)
        escala = rgb.shape[1] / 900
        cv2.circle(rgb, centro, radio, (255, 0, 0), grosor)
        texto = f"{ancho_mm:.1f} x {alto_mm:.1f} mm"
        cv2.putText(rgb, texto, (max(centro[0] - radio, 0), max(centro[1] - radio - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, escala, (255, 255, 0), grosor, cv2.LINE_AA)

        binaria = cv2.cvtColor(self.__seg_binaria, cv2.COLOR_GRAY2RGB)
        cv2.circle(binaria, centro, radio, (255, 128, 0), grosor * 2)

        recorte = rgb[max(y1 - 20, 0):y2 + 20, max(x1 - 20, 0):x2 + 20]

        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(rgb)
        ax1.set_title(f"Elemento {E}: área ≈ {area_mm2:.1f} mm²")
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(binaria)
        ax2.set_title(f"Binaria + apertura (kernel {tam_kernel}, {iteraciones} it.)")
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.imshow(recorte)
        ax3.set_title(f"Región recortada: {ancho_mm:.1f} x {alto_mm:.1f} mm")
        fig.tight_layout()

        nombre_grafico = f"{self.__nombre}_segmentacion_{self.__seg_nombre}.png"
        nombre_imagen = f"{self.__nombre}_circulo_{self.__seg_nombre}.png"
        fig.savefig(nombre_grafico)
        cv2.imwrite(nombre_imagen, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        plt.show()

        print(f"Área del elemento {E}: {area_px} píxeles = {area_mm2:.2f} mm²")
        print(f"Dimensiones: {ancho_mm:.2f} mm de ancho x {alto_mm:.2f} mm de alto")
        return area_mm2, nombre_grafico, nombre_imagen
