# widgets/sweep_process_widget.py

import os
import pynvml
import numpy as np
import tensorflow as tf
import cv2
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog,
    QHBoxLayout, QGroupBox, QMessageBox, QSizePolicy, QProgressDialog
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from models.seg_model import model as build_model
from models.error_regression_model import build_residual_model
from models.ac_estimator import fit_ellipse_ac_mm_pred
from single_image_widget import AspectRatioWidget
from validators import load_and_predict_best_frame
from mha_reader import split_mha_into_sweeps, extract_features_from_sweeps


# class MHAProcessingWorker(QObject):
#     finished = pyqtSignal(str)
#
#     def __init__(self, path, sequence_model_path):
#         super().__init__()
#         self.path = path
#         self.sequence_model_path = sequence_model_path
#
#     def run(self):
#         try:
#             base_name = os.path.splitext(os.path.basename(self.path))[0]
#             feature_root = os.path.join("test_data", "mha", "features", base_name)
#             sweep_dirs = [os.path.join(feature_root, d) for d in os.listdir(feature_root) if d.startswith("sweep_")]
#
#             best_score = -np.inf
#             best_frame = None
#
#             for sweep_path in sweep_dirs:
#                 try:
#                     idx, scores = load_and_predict_best_frame(sweep_path, self.sequence_model_path)
#                     if scores[idx] > best_score:
#                         best_score = scores[idx]
#                         parts = os.path.normpath(sweep_path).split(os.sep)
#                         data_path = os.path.join("test_data", "mha", "data", parts[-2], parts[-1])
#                         best_frame = os.path.join(data_path, "images", f"{idx:03d}.png")
#                 except Exception as e:
#                     print(f"[⚠] Error processing {sweep_path}: {e}")
#
#             self.finished.emit(best_frame if best_frame and os.path.exists(best_frame) else "")
#         except Exception as e:
#             print(f"[‼] Worker error: {e}")
#             self.finished.emit("")
class MHAProcessingWorker(QObject):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)  # 新增：用于主线程弹窗

    def __init__(self, path, sequence_model_path):
        super().__init__()
        self.path = path
        self.sequence_model_path = sequence_model_path

    @staticmethod
    def get_total_gpu_memory_gb():
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            pynvml.nvmlShutdown()
            return mem_info.total / (1024 ** 3)  # GB
        except Exception as e:
            print(f"[⚠] GPU memory check failed: {e}")
            return 0

    def run(self):
        try:
            base_name = os.path.splitext(os.path.basename(self.path))[0]
            feature_root = os.path.join("test_data", "mha", "features", base_name)

            if not os.path.exists(feature_root) or not os.listdir(feature_root):
                sweep_dirs = split_mha_into_sweeps(
                    self.path, save_root=feature_root,
                    frames_per_sweep=140, remove_tail=25
                )

                total_gpu_gb = self.get_total_gpu_memory_gb()
                print(f"[GPU] Total memory: {total_gpu_gb:.2f} GB")

                if total_gpu_gb <= 16:
                    self.error.emit("Your GPU does not have enough memory (≥16GB required) to extract features.\n"
                                    "Please try on a machine with more GPU memory.")
                    self.finished.emit("")
                    return

                print("[GPU] > 16GB → Running feature extraction...")
                extract_features_from_sweeps(sweep_dirs)

            sweep_dirs = [os.path.join(feature_root, d)
                          for d in os.listdir(feature_root)
                          if d.startswith("sweep_")]

            best_score = -np.inf
            best_frame = None

            for sweep_path in sweep_dirs:
                try:
                    idx, scores = load_and_predict_best_frame(sweep_path, self.sequence_model_path)
                    if scores[idx] > best_score:
                        best_score = scores[idx]
                        parts = os.path.normpath(sweep_path).split(os.sep)
                        data_path = os.path.join("test_data", "mha", "data", parts[-2], parts[-1])
                        best_frame = os.path.join(data_path, "images", f"{idx:03d}.png")
                except Exception as e:
                    print(f"[⚠] Error processing {sweep_path}: {e}")

            self.finished.emit(best_frame if best_frame and os.path.exists(best_frame) else "")
        except Exception as e:
            print(f"[‼] Worker error: {e}")
            self.finished.emit("")


class SweepProcessWidget(QWidget):
    def __init__(self, seg_model, residual_model):
        super().__init__()
        self.ac_label = None
        self.ax = None
        self.canvas = None
        self.segment_button = None
        self.load_button = None
        self.image_label = None
        self.best_frame_path = None
        self.temp_mask_path = "temp_pred_mask.png"

        self.seg_model = seg_model
        self.residual_model = residual_model

        self.sequence_model_path = "assets/sequence_model.h5"
        self.input_size = (384, 508)
        self.orig_size = (744, 562)

        self.has_gpu_error = False

        self.init_ui()


    def init_ui(self):
        self.setWindowTitle("Fetal AC Estimation System (Sweep Mode)")
        main_layout = QVBoxLayout(self)

        button_layout = QHBoxLayout()
        self.load_button = QPushButton("📂 Load MHA File")
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

        self.load_button.clicked.connect(self.load_mha_file)
        self.segment_button.clicked.connect(self.run_segmentation)

        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.segment_button)

        image_group = QGroupBox("Image Display")
        image_layout = QHBoxLayout()

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        left_wrapper = AspectRatioWidget(self.image_label, aspect_ratio=508 / 384)
        left_group = QGroupBox("Original Image")
        left_group.setStyleSheet(
            "QGroupBox { background-color: #f4f4f4; border: 1px solid gray; border-radius:4px; margin-top:1ex; }"
            "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px; }"
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

        canvas_wrapper = AspectRatioWidget(self.canvas, aspect_ratio=508 / 384)
        canvas_wrapper.setStyleSheet("background-color: transparent;")

        right_group = QGroupBox("Overlay: Original + Mask + Fitted Ellipse")
        right_group.setStyleSheet(
            "QGroupBox { background-color: #f4f4f4; border: 1px solid gray; border-radius:4px; margin-top:1ex; }"
            "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px; }"
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

    def load_mha_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select MHA File", "", "*.mha")
        if not path:
            return

        self.best_frame_path = None
        self.has_gpu_error = False  # 每次加载前先重置

        self.progress_dialog = QProgressDialog("Processing MHA file...", None, 0, 0, self)
        self.progress_dialog.setWindowModality(Qt.ApplicationModal)
        self.progress_dialog.setCancelButton(None)
        self.progress_dialog.setWindowTitle("Please Wait")
        self.progress_dialog.setWindowFlags(self.progress_dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.progress_dialog.show()

        self.thread = QThread()
        self.worker = MHAProcessingWorker(path, self.sequence_model_path)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_mha_processing_done)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.error.connect(self.show_error_message)

        self.thread.start()

    def show_error_message(self, message):
        self.has_gpu_error = True  # 设置显存不足标志
        self.progress_dialog.close()
        QMessageBox.critical(self, "Insufficient GPU Memory", message)

    def on_mha_processing_done(self, best_frame):
        self.progress_dialog.close()

        if self.has_gpu_error:
            self.has_gpu_error = False  # 重置
            return  # 不继续处理，也不弹“无效帧”提示

        if not best_frame:
            QMessageBox.warning(self, "No valid frame", "Could not find a usable key frame.")
            return

        self.best_frame_path = best_frame
        pixmap = QPixmap(self.best_frame_path)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        self.image_label.setStyleSheet("border: 1px solid gray; background-color: #f4f4f4;")

    def run_segmentation(self):
        if not self.best_frame_path:
            QMessageBox.warning(self, "No frame", "Please load a sweep file first.")
            return

        img = tf.io.read_file(self.best_frame_path)
        img = tf.image.decode_png(img, channels=3)
        resized = tf.image.resize(img, self.input_size)
        norm = tf.cast(resized, tf.float32) / 127.5 - 1.0
        inp = tf.expand_dims(norm, axis=0)

        pred = self.seg_model.predict(inp, verbose=0)
        mask = tf.argmax(pred, axis=-1)[0].numpy().astype(np.uint8)
        mask_resized = cv2.resize(mask, self.orig_size, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(self.temp_mask_path, mask_resized)

        fit_ellipse_ac_mm_pred(
            mask_path=self.temp_mask_path,
            image_path=self.best_frame_path,
            show_plot=True,
            draw_func=self.draw_overlay_on_canvas,
            spacing=0.28,
            scale_factor=1,
            epsilon=0.01
        )
        ac_mm = fit_ellipse_ac_mm_pred(
            mask_path=self.temp_mask_path,
            image_path=self.best_frame_path,
            show_plot=False,
            draw_func=None,
            spacing=0.28,
            scale_factor=1,
            epsilon=0.0015
        )

        if ac_mm == 0.0:
            QMessageBox.warning(self, "Fitting Failed", "Ellipse fitting failed. Segmentation may be invalid.")
            return
        if ac_mm < 140 or ac_mm > 350:
            QMessageBox.warning(self, "Suspicious AC", "AC value out of expected range. Please verify.")
            return

        corrected = self.run_residual_model(self.best_frame_path, self.temp_mask_path, ac_mm)
        self.ac_label.setText(f"AC Result → Initial: {ac_mm:.2f} mm | Corrected: {corrected:.2f} mm")

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
