#!/usr/bin/env python3

# Copyright (C) 2024  Xiaofeng Yan, Shudong Li
# Xueming Li Lab, Tsinghua University

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from PyQt5.QtWidgets import QMainWindow, QGraphicsScene, QApplication, QSpinBox, QDoubleSpinBox, QPushButton, QComboBox
from PyQt5.QtGui import QPixmap,QImage
from PyQt5 import uic
import os,sys
import numpy as np
import mrcfile
from mpicker_merge_view import Mpicker_merge_view

# Define function to import external files when using PyInstaller.
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS

    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class Mpicker_merge(QMainWindow):
    def __init__(self, fmrc):
        super(Mpicker_merge, self).__init__()
        # Load the ui file
        self.uifile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "merge.ui")
        uic.loadUi(self.uifile_path, self)
        self.graphicsView_1: Mpicker_merge_view
        self.graphicsView_2: Mpicker_merge_view
        self.graphicsView_merge: Mpicker_merge_view
        self.doubleSpinBox_min1: QDoubleSpinBox
        self.doubleSpinBox_max1: QDoubleSpinBox
        self.doubleSpinBox_min2: QDoubleSpinBox
        self.doubleSpinBox_max2: QDoubleSpinBox
        self.spinBox_z1: QSpinBox
        self.spinBox_z2: QSpinBox
        self.spinBox_sum1: QSpinBox
        self.spinBox_sum2: QSpinBox
        self.comboBox_mode1: QComboBox
        self.comboBox_mode2: QComboBox
        self.pushButton_save: QPushButton
        # set graphics views
        self.graphicsScene_1 = QGraphicsScene()
        self.graphicsScene_2 = QGraphicsScene()
        self.graphicsScene_merge = QGraphicsScene()
        self.graphicsView_1.setScene(self.graphicsScene_1)
        self.graphicsView_2.setScene(self.graphicsScene_2)
        self.graphicsView_merge.setScene(self.graphicsScene_merge)
        self.pixmap_merge: QPixmap = None
        # process the input file
        with mrcfile.open(fmrc, mode='r', permissive=True) as mrc:
            data:np.ndarray = mrc.data.copy()
            assert data.ndim == 3, "Input MRC file should be 3D."
            data = data[:,::-1, :] # flip y axis
        self.data = data.astype(np.float32)
        dmin, dmax = np.min(self.data), np.max(self.data)
        if dmin == dmax:
            self.data = self.data * 0.0 + 127.0
        else:
            self.data = (self.data - dmin) / (dmax - dmin) * 255.0
        # set spinbox
        mean = np.mean(self.data)
        std = np.std(self.data)
        spin_min = max(round(mean - 3 * std), 0.0)
        spin_max = min(round(mean + 3 * std), 255.0)
        self.doubleSpinBox_min1.setValue(spin_min)
        self.doubleSpinBox_max1.setValue(spin_max)
        self.doubleSpinBox_min2.setValue(spin_min)
        self.doubleSpinBox_max2.setValue(spin_max)
        self.spinBox_z1.setMaximum(len(self.data))
        self.spinBox_z1.setValue(len(self.data)//2+1)
        self.spinBox_z2.setMaximum(len(self.data))
        self.spinBox_z2.setValue(len(self.data)//2+1)
        self.spinBox_sum1.setMaximum(len(self.data))
        self.spinBox_sum2.setMaximum(len(self.data))
        self.update_image()
        # connect signals
        self.doubleSpinBox_min1.valueChanged.connect(self.update_image)
        self.doubleSpinBox_max1.valueChanged.connect(self.update_image)
        self.doubleSpinBox_min2.valueChanged.connect(self.update_image)
        self.doubleSpinBox_max2.valueChanged.connect(self.update_image)
        self.spinBox_z1.valueChanged.connect(self.update_image)
        self.spinBox_z2.valueChanged.connect(self.update_image)
        self.spinBox_sum1.valueChanged.connect(self.update_image)
        self.spinBox_sum2.valueChanged.connect(self.update_image)
        self.comboBox_mode1.currentTextChanged.connect(self.update_image)
        self.comboBox_mode2.currentTextChanged.connect(self.update_image)
        self.pushButton_save.clicked.connect(self.save_image)
    
    def rescale_image_uint8(self, image, min_val, max_val):
        assert min(min_val, max_val) >= 0 and max(max_val, min_val) <= 255
        if min_val == max_val:
            max_val += 1e-6  # Avoid division by zero
        if min_val > max_val:
            image = 255 - image # Invert the image
            min_val = 255 - min_val
            max_val = 255 - max_val
        image = np.select(
            [(image > min_val) & (image < max_val),
             image <= min_val,
             image >= max_val],
            [255 * (image - min_val) / (max_val - min_val),
             0,
             255])
        return np.round(image).astype(np.uint8)
    
    def image_to_pixmap(self, img):
        img = np.round(img).astype(np.uint8)
        height, width, channels = img.shape
        bytesPerLine = channels * width
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        return pixmap
    
    def get_image(self, z, sum, mode):
        z1 = z - 1
        z2 = min(z - 1 + sum, len(self.data))
        img = self.data[z1:z2, :, :]
        if mode == "mean":
            img = img.mean(axis=0)
        elif mode == "min":
            img = img.min(axis=0)
        elif mode == "max":
            img = img.max(axis=0)
        return img

    def update_image(self):
        img1 = self.get_image(self.spinBox_z1.value(), self.spinBox_sum1.value(), self.comboBox_mode1.currentText())
        img2 = self.get_image(self.spinBox_z2.value(), self.spinBox_sum2.value(), self.comboBox_mode2.currentText())
        # rescale images
        img1 = self.rescale_image_uint8(img1, self.doubleSpinBox_min1.value(), self.doubleSpinBox_max1.value())
        img2 = self.rescale_image_uint8(img2, self.doubleSpinBox_min2.value(), self.doubleSpinBox_max2.value())
        
        # show image 1, megenta
        img1_rgb = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.uint8)
        img1_rgb[:,:,0] = img1
        img1_rgb[:,:,1] = img1
        img1_rgb[:,:,2] = img1
        pixmap1 = self.image_to_pixmap(img1_rgb)
        self.graphicsScene_1.clear()
        self.graphicsScene_1.addPixmap(pixmap1)
        # show image 2
        img2_rgb = np.zeros((img2.shape[0], img2.shape[1], 3), dtype=np.uint8)
        img2_rgb[:,:,0] = img2
        img2_rgb[:,:,1] = img2
        img2_rgb[:,:,2] = img2
        pixmap2 = self.image_to_pixmap(img2_rgb)
        self.graphicsScene_2.clear()
        self.graphicsScene_2.addPixmap(pixmap2)
        # merge images
        merged_img = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.uint8)
        merged_img[:,:,0] = 255 - img1
        merged_img[:,:,1] = 255 - img2
        merged_img[:,:,2] = 255 - img1
        self.pixmap_merge = self.image_to_pixmap(merged_img)
        self.graphicsScene_merge.clear()
        self.graphicsScene_merge.addPixmap(self.pixmap_merge)

    def save_image(self):
        if self.pixmap_merge is None:
            return
        idx = 0
        while os.path.exists(os.path.abspath(f"Merge_{idx:03}.tif")):
            idx += 1
        fname = os.path.abspath(f"Merge_{idx:03}.tif")
        self.pixmap_merge.save(fname, "TIFF")
        print(f"save images as: {fname}")
        return


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: Mpicker_merge.py input.mrc")
        exit()
    fin = sys.argv[1]
    if os.path.exists(fin) is False:
        print(f"File {fin} does not exist.")
        print("Usage: Mpicker_merge.py input.mrc")
        exit()
    app = QApplication(sys.argv)
    window = Mpicker_merge(fin)
    window.show()
    sys.exit(app.exec_())
