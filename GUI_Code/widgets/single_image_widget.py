# widgets/single_image_widget.py

import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog,
    QHBoxLayout, QGroupBox, QMessageBox, QSizePolicy
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import numpy as np
import tensorflow as tf
import cv2
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from models.seg_model import model as build_model
from models.error_regression_model import build_residual_model
from models.ac_estimator import fit_ellipse_ac_mm_pred

def resource_path(relative_path):
    import sys
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


class AspectRatioWidget(QWidget):
    def __init__(self, child_widget, aspect_ratio=508/384, parent=None):
        super().__init__(parent)
        self.child = child_widget
        self.aspect_ratio = aspect_ratio
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(self.child, alignment=Qt.AlignCenter)
        self.setStyleSheet("background-color: #f4f4f4;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def resizeEvent(self, event):
        w = self.width()
        h = int(w / self.aspect_ratio)
        if h > self.height():
            h = self.height()
            w = int(h * self.aspect_ratio)
        self.child.setFixedSize(w, h)


class SingleImageWidget(QWidget):
    def __init__(self, seg_model, residual_model):
        super().__init__()
        self.ac_label = None
        self.image_path = None
        self.ax = None
        self.canvas = None
        self.segment_button = None
        self.load_button = None
        self.image_label = None

        self.seg_model = seg_model
        self.residual_model = residual_model

        self.input_size = (384, 508)
        self.orig_size = (744, 562)

        self.init_ui()


    def init_ui(self):
        self.setWindowTitle("Fetal AC Estimation System")
        main_layout = QVBoxLayout(self)

        # 按钮
        button_layout = QHBoxLayout()
        self.load_button = QPushButton("📂 Load Image")
        self.segment_button = QPushButton("🔬 Run Segmentation + AC Estimation")

        btn_style = """
            QPushButton {
                background-color: #ddd;
                color: #333;
                border: 1px solid #aaa;
                padding: 6px 14px;
                border-radius: 4px;
                box-shadow: inset 2px 2px 5px rgba(0,0,0,0.1),
                            inset -2px -2px 5px rgba(255,255,255,0.7);
            }
            QPushButton:hover {
                background-color: #ccc;
                border-color: #888;
            }
            QPushButton:pressed {
                box-shadow: inset 1px 1px 2px rgba(0,0,0,0.2),
                            inset -1px -1px 2px rgba(255,255,255,0.8);
                border-color: #666;
            }
        """
        self.load_button.setStyleSheet(btn_style)
        self.segment_button.setStyleSheet(btn_style)

        self.load_button.clicked.connect(self.load_image)
        self.segment_button.clicked.connect(self.run_segmentation)

        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.segment_button)

        image_group = QGroupBox("Image Display")
        image_layout = QHBoxLayout()

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        left_wrapper = AspectRatioWidget(self.image_label, aspect_ratio=508/384)
        left_group = QGroupBox("Original Image")
        left_group.setStyleSheet(
            "QGroupBox { background-color: #f4f4f4; border: 1px solid gray; border-radius:4px; margin-top:1ex;}"
            "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding:0 3px;}"
        )
        left_vbox = QVBoxLayout(left_group)
        left_vbox.setContentsMargins(10, 10, 10, 10)
        left_vbox.addWidget(left_wrapper)
        image_layout.addWidget(left_group, 1)

        self.canvas = FigureCanvas(Figure(figsize=(5.08, 3.84)))
        self.canvas.figure.patch.set_facecolor('#f4f4f4')
        self.canvas.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.ax = self.canvas.figure.add_subplot(111)
        self.ax.axis('off')

        canvas_wrapper = AspectRatioWidget(self.canvas, aspect_ratio=508/384)
        canvas_wrapper.setStyleSheet("background-color: transparent;")

        right_group = QGroupBox("Overlay: Original + Mask + Fitted Ellipse")
        right_group.setStyleSheet(
            "QGroupBox { background-color: #f4f4f4; border: 1px solid gray; border-radius:4px; margin-top:1ex;}"
            "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding:0 3px;}"
        )
        right_vbox = QVBoxLayout(right_group)
        right_vbox.setContentsMargins(10, 10, 10, 10)
        right_vbox.addWidget(canvas_wrapper)
        image_layout.addWidget(right_group, 1)

        image_group.setLayout(image_layout)

        self.ac_label = QLabel("AC Result: -")
        self.ac_label.setAlignment(Qt.AlignCenter)
        self.ac_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        self.ac_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        ac_group = QGroupBox("AC Results")
        ac_vbox = QVBoxLayout(ac_group)
        ac_vbox.setContentsMargins(10, 10, 10, 10)
        ac_vbox.addWidget(self.ac_label)

        main_layout.addLayout(button_layout)
        main_layout.addWidget(image_group)
        main_layout.addWidget(ac_group)
        main_layout.setStretch(0, 0)
        main_layout.setStretch(1, 6)
        main_layout.setStretch(2, 3)

        self.setLayout(main_layout)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select an image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.image_path = path
            pixmap = QPixmap(path)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            self.image_label.setStyleSheet("border: 1px solid gray; background-color: #f4f4f4;")

    def run_segmentation(self):
        if not self.image_path:
            QMessageBox.warning(self, "No image", "Please load an image first.")
            return

        # print(">> [1] Start reading image")
        img = tf.io.read_file(self.image_path)
        img = tf.image.decode_png(img, channels=3)
        resized = tf.image.resize(img, self.input_size)
        norm = tf.cast(resized, tf.float32) / 127.5 - 1.0
        inp = tf.expand_dims(norm, axis=0)
        # print(">> [2] Running segmentation model")
        pred = self.seg_model.predict(inp, verbose=0)
        # print(">> [3] Postprocessing mask")
        mask = tf.argmax(pred, axis=-1)[0].numpy().astype(np.uint8)
        mask_resized = cv2.resize(mask, self.orig_size, interpolation=cv2.INTER_NEAREST)
        temp_mask = "temp_pred_mask.png"
        cv2.imwrite(temp_mask, mask_resized)
        # print(">> [4] First ellipse fit with show_plot=True")
        fit_ellipse_ac_mm_pred(mask_path=temp_mask, image_path=self.image_path, show_plot=True, draw_func=self.draw_overlay_on_canvas, spacing=0.28, scale_factor=1, epsilon=0.01)
        # print(">> [5] Second ellipse fit with show_plot=False")
        ac_mm = fit_ellipse_ac_mm_pred(mask_path=temp_mask, image_path=self.image_path, show_plot=False, draw_func=None, spacing=0.28, scale_factor=1, epsilon=0.0015)
        # print(">> [6] Check AC result")
        if ac_mm == 0.0:
            QMessageBox.warning(self, "Fitting Failed", "Ellipse fitting failed. Segmentation may be invalid.")
            return
        if ac_mm < 140 or ac_mm > 350:
            QMessageBox.warning(self, "Suspicious AC", "AC value out of expected range. Please verify the image.")
            return
        # print(">> [7] Run residual correction")
        corrected = self.run_residual_model(self.image_path, temp_mask, ac_mm)
        self.ac_label.setText(f"AC Result → Initial: {ac_mm:.2f} mm | Corrected: {corrected:.2f} mm")
        # print(">> [8] Done. AC:", ac_mm, "Corrected:", corrected)

    def run_residual_model(self, image_path, mask_path, ac_pred):
        IMG_SIZE = (562, 744)
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0

        mask_raw = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask_raw, channels=1)
        mask = tf.image.resize(mask, IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mask = tf.cast(mask, tf.float32)
        mask_bin = tf.where(mask > 127, 1.0, 0.0)

        x = tf.concat([img, mask_bin], axis=-1)
        x = tf.expand_dims(x, axis=0)
        ac_tensor = tf.constant([[ac_pred]], dtype=tf.float32)
        res = self.residual_model.predict([x, ac_tensor], verbose=0)
        return ac_pred + float(res[0][0])

    def draw_overlay_on_canvas(self, overlay_img):
        self.ax.clear()
        self.ax.imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
        self.ax.axis('off')
        self.canvas.draw()
