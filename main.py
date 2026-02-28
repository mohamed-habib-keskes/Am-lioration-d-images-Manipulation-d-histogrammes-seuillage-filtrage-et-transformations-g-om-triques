from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap
import cv2
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

qtcreator_file = "design (1).ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtcreator_file)


class DesignWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(DesignWindow, self).__init__()
        self.setupUi(self)

        self.image = None
        self.gray_image = None

        # Connexion des boutons
        self.Browse.clicked.connect(self.get_image)
        self.Apply.clicked.connect(self.show_ImgHistEqualized)
        self.Validate_1.clicked.connect(self.show_ImgThresholding)
        self.Validate_2.clicked.connect(self.show_ImgFiltered)
        self.Validate_3.clicked.connect(self.show_ImgAugmented)
        self.ShowHistBtn.clicked.connect(self.show_HistInOriginalBlock)  # Nouveau bouton

    # ===============================
    # Affichage image dans QLabel
    # ===============================
    def makeFigure(self, img_path, widget):
        pixmap = QPixmap(img_path)
        widget.setPixmap(pixmap)
        widget.setScaledContents(True)

    # ===============================
    # Charger image
    # ===============================
    def get_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choisir une image", "", "Images (*.png *.jpg *.jpeg)"
        )

        if path:
            self.image = cv2.imread(path)
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            cv2.imwrite("Gray_Image.png", self.gray_image)
            self.makeFigure("Gray_Image.png", self.OriginalImg)

            self.show_HistOriginal()

    # ===============================
    # Histogramme original
    # ===============================
    def show_HistOriginal(self):
        hist = cv2.calcHist([self.gray_image], [0], None, [256], [0, 256])

        plt.figure()
        plt.plot(hist)
        plt.title("Histogramme Original")
        plt.savefig("Original_Histogram.png")
        plt.close()

        self.makeFigure("Original_Histogram.png", self.OriginalHist)

    # ===============================
    # Afficher histogramme dans bloc image originale
    # ===============================
    def show_HistInOriginalBlock(self):
        if self.gray_image is None:
            return

        hist = cv2.calcHist([self.gray_image], [0], None, [256], [0, 256])

        plt.figure()
        plt.plot(hist)
        plt.title("Histogramme Original")
        plt.savefig("Temp_Hist.png")
        plt.close()

        self.makeFigure("Temp_Hist.png", self.OriginalImg)

    # ===============================
    # Égalisation
    # ===============================
    def show_ImgHistEqualized(self):
        if self.gray_image is None:
            return

        equalized = cv2.equalizeHist(self.gray_image)
        cv2.imwrite("Equalized_Image.png", equalized)
        self.makeFigure("Equalized_Image.png", self.EqualizedImg)

        hist = cv2.calcHist([equalized], [0], None, [256], [0, 256])

        plt.figure()
        plt.plot(hist)
        plt.title("Histogramme Egalisé")
        plt.savefig("Equalized_Histogram.png")
        plt.close()

        self.makeFigure("Equalized_Histogram.png", self.EqualizedHist)

    # ===============================
    # Seuillage
    # ===============================
    def show_ImgThresholding(self):
        if self.gray_image is None:
            return

        if self.BinaryRadio.isChecked():
            _, result = cv2.threshold(
                self.gray_image, 120, 255, cv2.THRESH_BINARY
            )

        elif self.OtsuRadio.isChecked():
            _, result = cv2.threshold(
                self.gray_image, 0, 255, cv2.THRESH_OTSU
            )

        else:
            return

        cv2.imwrite("Thresholding_Image.png", result)
        self.makeFigure("Thresholding_Image.png", self.ThresholdingImg)

    # ===============================
    # Filtrage
    # ===============================
    def show_ImgFiltered(self):
        if self.gray_image is None:
            return

        if self.MeanRadio.isChecked():
            result = cv2.blur(self.gray_image, (11, 11))

        elif self.GaussianRadio.isChecked():
            result = cv2.GaussianBlur(self.gray_image, (15, 15), 10)

        elif self.MedianRadio.isChecked():
            result = cv2.medianBlur(self.gray_image, 13)

        else:
            return

        cv2.imwrite("Filtered_Image.png", result)
        self.makeFigure("Filtered_Image.png", self.FilteredImg)

    # ===============================
    # Opérations géométriques
    # ===============================
    def show_ImgAugmented(self):
        if self.gray_image is None:
            return

        h, w = self.gray_image.shape

        if self.RotationRadio.isChecked():
            centre = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(centre, 45, 1.0)
            result = cv2.warpAffine(self.gray_image, M, (w, h))

        elif self.CropRadio.isChecked():
            result = self.gray_image[0:h // 2, 0:w // 2]

        elif self.ZoomRadio.isChecked():
            s = random.uniform(1.5, 4.0)
            resized = cv2.resize(
                self.gray_image, None, fx=s, fy=s,
                interpolation=cv2.INTER_CUBIC
            )

            new_h, new_w = resized.shape
            start_x = new_w // 2 - w // 2
            start_y = new_h // 2 - h // 2
            result = resized[start_y:start_y + h, start_x:start_x + w]

        else:
            return

        cv2.imwrite("Augmented_Image.png", result)
        self.makeFigure("Augmented_Image.png", self.AugmentedImg)


# ===============================
# Main
# ===============================
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = DesignWindow()
    window.show()
    sys.exit(app.exec_())