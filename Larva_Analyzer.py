"""
Larva Turning Angle Analyzer
Analyzes turning angles of Drosophila larvae from ImageJ ROI files
"""

import sys
import os
import glob
import re
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSlider, 
                             QFileDialog, QSpinBox, QCheckBox, QGroupBox,
                             QMessageBox, QComboBox, QLineEdit, QScrollArea)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
from scipy import ndimage
from scipy.interpolate import splprep, splev
from skimage import measure
import zipfile
import struct


class LarvaTurningAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Larva Turning Angle Analyzer")
        self.setGeometry(100, 100, 1400, 900)
        
        # Data storage
        self.video_path = None
        self.roi_path = None
        self.cap = None
        self.rois = []
        self.current_frame = 0
        self.total_frames = 0
        self.current_display_frame = None
        
        # TIFF-specific storage
        self.tiff_reader = None
        self.tiff_file_path = None
        self.tiff_stack = None  # Store entire TIFF stack in memory
        
        # Single image mode
        self.single_image_mode = False
        self.single_image = None
        
        # Analysis data
        self.head_positions = {}
        self.tail_positions = {}
        self.body_angles = {}
        self.turning_angles = {}
        self.midline_points = {}
        
        # Segment points for detailed body analysis
        # Format: {frame: {point_name: np.array([x, y])}}
        self.segment_points = {}
        self.segment_labels = {
            'left': ['t1l', 't2l', 't3l', 'a1l', 'a2l', 'a3l', 'a4l', 'a5l', 'a6l', 'a7l'],
            'right': ['t1r', 't2r', 't3r', 'a1r', 'a2r', 'a3r', 'a4r', 'a5r', 'a6r', 'a7r'],
            'midline': ['st1', 'st2', 'st3', 'sa1', 'sa2', 'sa3', 'sa4', 'sa5', 'sa6', 'sa7']
        }
        self.show_segments = False
        self.dragging_segment = None  # Which segment point is being dragged
        
        # Settings
        self.show_roi = True
        self.show_axis = True
        self.show_midline = True
        self.auto_detect_endpoints = True
        self.manual_adjust_mode = False
        self.setting_head = True  # True = setting head, False = setting tail
        
        # Dragging state
        self.dragging = False
        self.drag_target = None  # 'head' or 'tail'
        self.drag_threshold = 25  # pixels - increased for easier grabbing
        
        # Zoom and pan state
        self.zoom_level = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        
        # ROI drawing state
        self.drawing_roi = False
        self.roi_drawing_points = []  # Points being drawn
        self.drawing_mode = False  # Whether in ROI drawing mode
        
        self.init_ui()
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        
        # Left panel - controls
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, stretch=1)
        
        # Middle panel - video display
        middle_panel = self.create_video_panel()
        main_layout.addWidget(middle_panel, stretch=2)
        
        # Right panel - plots
        right_panel = self.create_plot_panel()
        main_layout.addWidget(right_panel, stretch=2)
        
        main_widget.setLayout(main_layout)
        
    def create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        
        # File loading group
        file_group = QGroupBox("Load Files")
        file_layout = QVBoxLayout()
        
        # Load video/image buttons in a horizontal layout
        load_media_layout = QHBoxLayout()
        
        load_video_btn = QPushButton("Load Video/Image")
        load_video_btn.setToolTip("Load video (AVI, MP4, MOV) or single image (PNG, JPG, TIFF)")
        load_video_btn.clicked.connect(self.load_video)
        load_media_layout.addWidget(load_video_btn)
        
        file_layout.addLayout(load_media_layout)
        
        load_roi_btn = QPushButton("Load ROI File (.zip)")
        load_roi_btn.clicked.connect(self.load_rois)
        file_layout.addWidget(load_roi_btn)
        
        load_roi_folder_btn = QPushButton("Load ROIs from Folder")
        load_roi_folder_btn.clicked.connect(self.load_rois_from_folder)
        file_layout.addWidget(load_roi_folder_btn)
        
        # Draw ROI button
        self.draw_roi_btn = QPushButton("✏️ Draw ROI")
        self.draw_roi_btn.setToolTip("Click to draw ROI manually on current frame\n"
                                     "Left-click: Add points\n"
                                     "Right-click: Finish ROI\n"
                                     "Esc: Cancel")
        self.draw_roi_btn.setCheckable(True)
        self.draw_roi_btn.clicked.connect(self.toggle_draw_roi_mode)
        file_layout.addWidget(self.draw_roi_btn)
        
        # Auto-detect ROI button
        auto_roi_btn = QPushButton("🔍 Auto-detect ROI")
        auto_roi_btn.setToolTip("Automatically detect larva outline using threshold\n"
                                "Works on current frame or all frames")
        auto_roi_btn.clicked.connect(self.show_auto_detect_roi_dialog)
        file_layout.addWidget(auto_roi_btn)
        
        # Save drawn ROIs button
        save_roi_btn = QPushButton("💾 Save Drawn ROIs")
        save_roi_btn.setToolTip("Save drawn ROIs to ImageJ-compatible .roi files")
        save_roi_btn.clicked.connect(self.save_drawn_rois)
        file_layout.addWidget(save_roi_btn)
        
        self.file_status_label = QLabel("No files loaded")
        self.file_status_label.setWordWrap(True)
        file_layout.addWidget(self.file_status_label)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Frame navigation group
        nav_group = QGroupBox("Frame Navigation")
        nav_layout = QVBoxLayout()
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self.on_frame_changed)
        nav_layout.addWidget(self.frame_slider)
        
        frame_control_layout = QHBoxLayout()
        self.prev_btn = QPushButton("< Prev")
        self.prev_btn.clicked.connect(self.prev_frame)
        frame_control_layout.addWidget(self.prev_btn)
        
        self.frame_label = QLabel("Frame: 0/0")
        frame_control_layout.addWidget(self.frame_label)
        
        self.next_btn = QPushButton("Next >")
        self.next_btn.clicked.connect(self.next_frame)
        frame_control_layout.addWidget(self.next_btn)
        
        nav_layout.addLayout(frame_control_layout)
        
        # Jump to frame
        jump_layout = QHBoxLayout()
        jump_layout.addWidget(QLabel("Jump to:"))
        self.jump_spinbox = QSpinBox()
        self.jump_spinbox.setMinimum(0)
        self.jump_spinbox.valueChanged.connect(self.jump_to_frame)
        jump_layout.addWidget(self.jump_spinbox)
        nav_layout.addLayout(jump_layout)
        
        # ROI offset control
        offset_layout = QHBoxLayout()
        offset_layout.addWidget(QLabel("ROI Offset:"))
        self.roi_offset_spinbox = QSpinBox()
        self.roi_offset_spinbox.setMinimum(-1000)
        self.roi_offset_spinbox.setMaximum(1000)
        self.roi_offset_spinbox.setValue(0)
        self.roi_offset_spinbox.setToolTip("Adjust if ROI numbering doesn't match frame numbering")
        self.roi_offset_spinbox.valueChanged.connect(self.update_display)
        offset_layout.addWidget(self.roi_offset_spinbox)
        nav_layout.addLayout(offset_layout)
        
        nav_group.setLayout(nav_layout)
        layout.addWidget(nav_group)
        
        # Head/Tail definition group
        endpoint_group = QGroupBox("Head/Tail Definition")
        endpoint_layout = QVBoxLayout()
        
        self.auto_detect_cb = QCheckBox("Auto-detect endpoints")
        self.auto_detect_cb.setChecked(True)
        self.auto_detect_cb.stateChanged.connect(self.on_auto_detect_changed)
        endpoint_layout.addWidget(self.auto_detect_cb)
        
        self.snap_to_roi_cb = QCheckBox("Snap to ROI contour")
        self.snap_to_roi_cb.setChecked(True)
        self.snap_to_roi_cb.setToolTip("When enabled, head/tail points snap to nearest ROI boundary point")
        endpoint_layout.addWidget(self.snap_to_roi_cb)
        
        endpoint_layout.addWidget(QLabel("Detection method:"))
        self.detection_method = QComboBox()
        self.detection_method.addItems(["Anterior = Top", "Anterior = Bottom", 
                                        "Anterior = Left", "Anterior = Right",
                                        "Manual (click on video)"])
        self.detection_method.currentIndexChanged.connect(self.update_display)
        endpoint_layout.addWidget(self.detection_method)
        
        # Manual adjustment mode
        self.manual_adjust_btn = QPushButton("🖱 Click to Set Head/Tail")
        self.manual_adjust_btn.setCheckable(True)
        self.manual_adjust_btn.setStyleSheet("""
            QPushButton:checked {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }
        """)
        self.manual_adjust_btn.toggled.connect(self.toggle_manual_mode)
        endpoint_layout.addWidget(self.manual_adjust_btn)
        
        self.manual_mode_label = QLabel("Manual mode: OFF")
        self.manual_mode_label.setStyleSheet("color: gray;")
        endpoint_layout.addWidget(self.manual_mode_label)
        
        swap_btn = QPushButton("Swap Head/Tail")
        swap_btn.clicked.connect(self.swap_head_tail)
        endpoint_layout.addWidget(swap_btn)
        
        # Clear and re-detect buttons
        clear_layout = QHBoxLayout()
        clear_current_btn = QPushButton("Clear Current")
        clear_current_btn.clicked.connect(self.clear_current_frame)
        clear_current_btn.setToolTip("Remove manual head/tail for current frame")
        clear_layout.addWidget(clear_current_btn)
        
        redetect_btn = QPushButton("Re-detect")
        redetect_btn.clicked.connect(self.redetect_current)
        redetect_btn.setToolTip("Force re-detection for current frame")
        clear_layout.addWidget(redetect_btn)
        endpoint_layout.addLayout(clear_layout)
        
        # Copy to adjacent frames
        copy_layout = QHBoxLayout()
        copy_prev_btn = QPushButton("← Copy to Prev")
        copy_prev_btn.clicked.connect(lambda: self.copy_to_adjacent(-1))
        copy_layout.addWidget(copy_prev_btn)
        copy_next_btn = QPushButton("Copy to Next →")
        copy_next_btn.clicked.connect(lambda: self.copy_to_adjacent(1))
        copy_layout.addWidget(copy_next_btn)
        endpoint_layout.addLayout(copy_layout)
        
        apply_all_btn = QPushButton("Apply to All Frames")
        apply_all_btn.clicked.connect(self.apply_endpoints_to_all)
        apply_all_btn.setToolTip("Auto-detect only frames without manual positions (preserves adjustments)")
        endpoint_layout.addWidget(apply_all_btn)
        
        force_redetect_btn = QPushButton("⚠ Force Re-detect All")
        force_redetect_btn.clicked.connect(self.force_redetect_all)
        force_redetect_btn.setToolTip("WARNING: Overwrites ALL manual adjustments with auto-detection")
        force_redetect_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff9800;
                color: white;
            }
            QPushButton:hover {
                background-color: #f57c00;
            }
        """)
        endpoint_layout.addWidget(force_redetect_btn)
        
        endpoint_group.setLayout(endpoint_layout)
        layout.addWidget(endpoint_group)
        
        # Display options group
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout()
        
        self.show_roi_cb = QCheckBox("Show ROI contour")
        self.show_roi_cb.setChecked(True)
        self.show_roi_cb.stateChanged.connect(self.update_display)
        display_layout.addWidget(self.show_roi_cb)
        
        self.show_axis_cb = QCheckBox("Show body axis")
        self.show_axis_cb.setChecked(True)
        self.show_axis_cb.stateChanged.connect(self.update_display)
        display_layout.addWidget(self.show_axis_cb)
        
        self.show_midline_cb = QCheckBox("Show midline")
        self.show_midline_cb.setChecked(True)
        self.show_midline_cb.stateChanged.connect(self.update_display)
        display_layout.addWidget(self.show_midline_cb)
        
        # Midline method selection
        midline_method_layout = QHBoxLayout()
        midline_method_layout.addWidget(QLabel("Midline method:"))
        self.midline_method = QComboBox()
        self.midline_method.addItems(["Skeleton", "Simple interpolation"])
        self.midline_method.setToolTip("Skeleton: Medial axis (accurate)\nSimple: Straight interpolation (fast)")
        self.midline_method.currentIndexChanged.connect(self.update_display)
        midline_method_layout.addWidget(self.midline_method)
        display_layout.addLayout(midline_method_layout)
        
        # Segment points
        self.show_segments_cb = QCheckBox("Show segment points")
        self.show_segments_cb.setChecked(False)
        self.show_segments_cb.stateChanged.connect(self.on_show_segments_changed)
        self.show_segments_cb.setToolTip("Show T1-T3, A1-A7 segment points on left/right/midline")
        display_layout.addWidget(self.show_segments_cb)
        
        segment_btn_layout = QHBoxLayout()
        init_segments_btn = QPushButton("Initialize Segments")
        init_segments_btn.clicked.connect(self.initialize_segments_current)
        init_segments_btn.setToolTip("Create evenly-spaced segment points for current frame")
        segment_btn_layout.addWidget(init_segments_btn)
        
        apply_segments_btn = QPushButton("Apply to All")
        apply_segments_btn.clicked.connect(self.apply_segments_to_all)
        apply_segments_btn.setToolTip("Initialize segments for all frames with head/tail")
        segment_btn_layout.addWidget(apply_segments_btn)
        display_layout.addLayout(segment_btn_layout)
        
        # Copy segments to adjacent frames
        segment_copy_layout = QHBoxLayout()
        copy_seg_prev_btn = QPushButton("← Copy Segments")
        copy_seg_prev_btn.clicked.connect(lambda: self.copy_segments_to_adjacent(-1))
        copy_seg_prev_btn.setToolTip("Copy segment points to previous frame")
        segment_copy_layout.addWidget(copy_seg_prev_btn)
        copy_seg_next_btn = QPushButton("Copy Segments →")
        copy_seg_next_btn.clicked.connect(lambda: self.copy_segments_to_adjacent(1))
        copy_seg_next_btn.setToolTip("Copy segment points to next frame")
        segment_copy_layout.addWidget(copy_seg_next_btn)
        display_layout.addLayout(segment_copy_layout)
        
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        # Analysis group
        analysis_group = QGroupBox("Analysis")
        analysis_layout = QVBoxLayout()
        
        analyze_all_btn = QPushButton("Analyze All Frames")
        analyze_all_btn.clicked.connect(self.analyze_all_frames)
        analysis_layout.addWidget(analyze_all_btn)
        
        self.analysis_status = QLabel("Not analyzed")
        self.analysis_status.setWordWrap(True)
        analysis_layout.addWidget(self.analysis_status)
        
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)
        
        # Export group
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout()
        
        export_csv_btn = QPushButton("Export to CSV")
        export_csv_btn.clicked.connect(self.export_csv)
        export_layout.addWidget(export_csv_btn)
        
        export_excel_btn = QPushButton("Export to Excel (Multi-Sheet)")
        export_excel_btn.clicked.connect(self.export_excel)
        export_excel_btn.setToolTip("Export to Excel with separate sheets for metadata, angles, and measurements")
        export_layout.addWidget(export_excel_btn)
        
        export_plot_btn = QPushButton("Save Plots")
        export_plot_btn.clicked.connect(self.save_plots)
        export_layout.addWidget(export_plot_btn)
        
        export_video_btn = QPushButton("🎬 Export Annotated Video")
        export_video_btn.clicked.connect(self.export_annotated_video)
        export_video_btn.setToolTip("Create video with ROIs, segments, and angle overlays")
        export_layout.addWidget(export_video_btn)
        
        # Session save/load
        session_layout = QHBoxLayout()
        save_session_btn = QPushButton("💾 Save Session")
        save_session_btn.clicked.connect(self.save_session)
        save_session_btn.setToolTip("Save head/tail positions and settings for later")
        session_layout.addWidget(save_session_btn)
        
        load_session_btn = QPushButton("📂 Load Session")
        load_session_btn.clicked.connect(self.load_session)
        load_session_btn.setToolTip("Load previously saved head/tail positions")
        session_layout.addWidget(load_session_btn)
        export_layout.addLayout(session_layout)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        layout.addStretch()
        panel.setLayout(layout)
        
        # Wrap panel in scroll area so all controls are accessible even with large videos
        scroll_area = QScrollArea()
        scroll_area.setWidget(panel)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setMinimumWidth(300)  # Ensure minimum width for controls
        scroll_area.setMaximumWidth(350)  # Limit maximum width
        
        return scroll_area
        
    def create_video_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("Video Display"))
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("QLabel { background-color: black; }")
        self.video_label.setScaledContents(False)  # Changed to False for manual zoom control
        self.video_label.mousePressEvent = self.on_video_click
        self.video_label.mouseMoveEvent = self.on_video_move
        self.video_label.mouseReleaseEvent = self.on_video_release
        self.video_label.wheelEvent = self.on_video_wheel
        self.video_label.setMouseTracking(True)  # Enable mouse tracking for hover
        layout.addWidget(self.video_label)
        
        # Zoom controls
        zoom_layout = QHBoxLayout()
        zoom_label = QLabel("Zoom: Use mouse wheel | Pan: Right-click + drag")
        zoom_layout.addWidget(zoom_label)
        
        zoom_in_btn = QPushButton("Zoom In (+)")
        zoom_in_btn.clicked.connect(lambda: self.adjust_zoom(1.2))
        zoom_layout.addWidget(zoom_in_btn)
        
        zoom_out_btn = QPushButton("Zoom Out (-)")
        zoom_out_btn.clicked.connect(lambda: self.adjust_zoom(0.8))
        zoom_layout.addWidget(zoom_out_btn)
        
        reset_view_btn = QPushButton("Reset View")
        reset_view_btn.clicked.connect(self.reset_view)
        zoom_layout.addWidget(reset_view_btn)
        
        layout.addLayout(zoom_layout)
        
        # Info label
        self.info_label = QLabel("Use 'Click to Set Head/Tail' button for manual adjustments")
        layout.addWidget(self.info_label)
        
        # Status indicator for current frame
        self.position_status_label = QLabel("Position: Not set")
        self.position_status_label.setStyleSheet("QLabel { padding: 5px; background-color: #f0f0f0; }")
        layout.addWidget(self.position_status_label)
        
        panel.setLayout(layout)
        return panel
        
    def create_plot_panel(self):
        from PyQt5.QtWidgets import QTabWidget, QScrollArea
        
        panel = QWidget()
        main_layout = QVBoxLayout()
        
        # Create tab widget
        tab_widget = QTabWidget()
        
        # TAB 1: Angle/Curvature Plots
        angles_tab = QWidget()
        angles_layout = QVBoxLayout()
        angles_layout.addWidget(QLabel("Angle & Curvature Analysis"))
        
        # Create matplotlib figure for angles
        self.figure = Figure(figsize=(8, 10))
        self.canvas = FigureCanvas(self.figure)
        angles_layout.addWidget(self.canvas)
        
        # Create subplots
        self.ax_angle = self.figure.add_subplot(311)
        self.ax_angle.set_title("Body Angle Over Time")
        self.ax_angle.set_xlabel("Frame")
        self.ax_angle.set_ylabel("Angle (degrees)")
        self.ax_angle.grid(True)
        
        self.ax_turning = self.figure.add_subplot(312)
        self.ax_turning.set_title("Turning Angle (Frame-to-Frame)")
        self.ax_turning.set_xlabel("Frame")
        self.ax_turning.set_ylabel("Turning Angle (degrees)")
        self.ax_turning.grid(True)
        
        self.ax_curvature = self.figure.add_subplot(313)
        self.ax_curvature.set_title("Body Curvature")
        self.ax_curvature.set_xlabel("Frame")
        self.ax_curvature.set_ylabel("Curvature (1/pixels)")
        self.ax_curvature.grid(True)
        
        self.figure.tight_layout()
        angles_tab.setLayout(angles_layout)
        
        # TAB 2: Segment Distance Plots
        distances_tab = QWidget()
        distances_layout = QVBoxLayout()
        distances_layout.addWidget(QLabel("Segment Distances Over Time (Left=Red, Right=Green)"))
        
        # Create scrollable area for segment plots
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout()
        
        # Create matplotlib figure for segment distances - VERTICAL STACKING
        self.segment_figure = Figure(figsize=(10, 24))  # Taller for 11 subplots
        self.segment_canvas = FigureCanvas(self.segment_figure)
        scroll_layout.addWidget(self.segment_canvas)
        
        # Create 11 subplots STACKED VERTICALLY (11 rows x 1 column)
        # Including Head-T1 and A7-Tail distances
        self.segment_distance_axes = []
        segment_pairs = [
            ('head', 't1', 'H-T1'),      # Head to T1
            ('t1', 't2', 'T1-T2'),
            ('t2', 't3', 'T2-T3'),
            ('t3', 'a1', 'T3-A1'),
            ('a1', 'a2', 'A1-A2'),
            ('a2', 'a3', 'A2-A3'),
            ('a3', 'a4', 'A3-A4'),
            ('a4', 'a5', 'A4-A5'),
            ('a5', 'a6', 'A5-A6'),
            ('a6', 'a7', 'A6-A7'),
            ('a7', 'tail', 'A7-T')       # A7 to Tail
        ]
        
        for i, (seg1, seg2, title) in enumerate(segment_pairs):
            # Create subplot in vertical stack (11 rows, 1 column)
            ax = self.segment_figure.add_subplot(11, 1, i+1)
            
            # Labels and title
            ax.set_ylabel("Distance (px)", fontsize=9)
            ax.set_title(f"{title}", fontsize=10, loc='left', fontweight='bold')
            ax.grid(True, alpha=0.3, linewidth=0.5)
            
            # Only show x-axis label and tick labels on bottom plot
            if i == len(segment_pairs) - 1:
                ax.set_xlabel("Frame", fontsize=10)
            else:
                ax.set_xlabel("")
                ax.tick_params(axis='x', labelbottom=False)  # Hide x-tick labels
            
            # Share x-axis for easier comparison
            if i == 0:
                first_ax = ax
            else:
                ax.sharex(first_ax)
            
            self.segment_distance_axes.append((ax, seg1, seg2))
        
        self.segment_figure.tight_layout()
        
        scroll_content.setLayout(scroll_layout)
        scroll.setWidget(scroll_content)
        distances_layout.addWidget(scroll)
        distances_tab.setLayout(distances_layout)
        
        # Add tabs
        tab_widget.addTab(angles_tab, "Angles & Curvature")
        tab_widget.addTab(distances_tab, "Segment Distances")
        
        main_layout.addWidget(tab_widget)
        panel.setLayout(main_layout)
        return panel
        
    def load_video(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Video or Image", "", 
            "All Supported (*.avi *.mp4 *.mov *.tif *.tiff *.png *.jpg *.jpeg *.bmp);;"
            "Video Files (*.avi *.mp4 *.mov);;"
            "TIFF Files (*.tif *.tiff);;"
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if filename:
            self.video_path = filename
            
            # Detect file type
            ext = filename.lower()
            
            # Single image formats
            if ext.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                self.load_single_image(filename)
            # TIFF (could be single or multi-frame)
            elif ext.endswith(('.tif', '.tiff')):
                self.load_tiff_stack(filename)
            # Standard video
            else:
                self.load_standard_video(filename)
    
    def load_single_image(self, filename):
        """Load a single image file"""
        try:
            import imageio.v3 as iio
            
            # Read the image
            image = iio.imread(filename)
            
            # Convert to RGB if needed
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            elif image.shape[2] == 3 and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # PIL/imageio read as RGB, need to convert to BGR for OpenCV
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Store as single frame
            self.single_image = image
            self.single_image_mode = True
            self.total_frames = 1
            self.current_frame = 0
            
            # Clear video capture objects
            self.cap = None
            self.tiff_reader = None
            self.tiff_file_path = None
            self.tiff_stack = None
            
            # Disable frame navigation
            self.frame_slider.setEnabled(False)
            self.frame_slider.setMaximum(0)
            self.jump_spinbox.setEnabled(False)
            self.jump_spinbox.setMaximum(0)
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            
            print(f"Loaded single image: {filename}")
            print(f"Image size: {image.shape[1]}x{image.shape[0]}")
            
            self.update_file_status()
            self.display_frame()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")
            print(f"Error loading image: {e}")
            import traceback
            traceback.print_exc()
    
    def load_tiff_stack(self, filename):
        """Load multi-frame TIFF using imageio or tifffile"""
        # Reset single image mode
        self.single_image_mode = False
        self.single_image = None
        
        try:
            # Try imageio v3 first
            import imageio.v3 as iio
            
            # Read the entire stack to count frames
            # For TIFF stacks, this reads all frames at once
            try:
                # Try to read as a stack
                stack = iio.imread(filename)
                
                # Check if it's a 3D or 4D array (multiple frames)
                if len(stack.shape) >= 3:
                    # First dimension is usually the frame count for image stacks
                    self.total_frames = stack.shape[0]
                    print(f"Loaded TIFF stack with {self.total_frames} frames (imageio v3)")
                    
                    # Store the entire stack in memory (faster for navigation)
                    self.tiff_stack = stack
                    self.tiff_file_path = filename
                    self.cap = None
                    self.tiff_reader = None
                    
                    # Enable and set navigation controls
                    self.frame_slider.setEnabled(True)
                    self.jump_spinbox.setEnabled(True)
                    self.prev_btn.setEnabled(True)
                    self.next_btn.setEnabled(True)
                    
                    # Set maximum values
                    max_val = min(self.total_frames - 1, 2147483647)
                    self.frame_slider.setMaximum(max_val)
                    self.jump_spinbox.setMaximum(max_val)
                    
                    self.update_file_status()
                    self.display_frame()
                    return
                else:
                    # Single frame TIFF
                    self.total_frames = 1
                    print(f"Loaded single-frame TIFF (imageio v3)")
                    self.tiff_stack = np.expand_dims(stack, 0)  # Make it 3D with 1 frame
                    self.tiff_file_path = filename
                    self.cap = None
                    self.tiff_reader = None
                    
                    self.frame_slider.setMaximum(0)
                    self.jump_spinbox.setMaximum(0)
                    
                    self.update_file_status()
                    self.display_frame()
                    return
                    
            except Exception as e:
                print(f"imageio v3 stack read failed: {e}, trying frame-by-frame...")
                # Try reading frame by frame
                frame_count = 0
                try:
                    while True:
                        iio.imread(filename, index=frame_count)
                        frame_count += 1
                except:
                    pass
                
                if frame_count > 0:
                    self.total_frames = frame_count
                    print(f"Loaded TIFF stack with {self.total_frames} frames (imageio v3, counted)")
                    
                    self.tiff_file_path = filename
                    self.tiff_stack = None  # Will read on demand
                    self.cap = None
                    self.tiff_reader = None
                    
                    max_val = min(self.total_frames - 1, 2147483647)
                    self.frame_slider.setMaximum(max_val)
                    self.jump_spinbox.setMaximum(max_val)
                    
                    self.update_file_status()
                    self.display_frame()
                    return
                else:
                    raise Exception("Could not read any frames")
                
        except (ImportError, AttributeError) as e:
            print(f"imageio v3 not available: {e}")
            # Try legacy imageio v2
            try:
                import imageio
                reader = imageio.get_reader(filename)
                
                # Count frames manually for legacy reader
                self.total_frames = 0
                for _ in reader:
                    self.total_frames += 1
                reader.close()
                
                print(f"Loaded TIFF stack with {self.total_frames} frames (imageio v2)")
                
                self.tiff_file_path = filename
                self.tiff_stack = None
                self.cap = None
                self.tiff_reader = None
                
                max_val = min(self.total_frames - 1, 2147483647)
                self.frame_slider.setMaximum(max_val)
                self.jump_spinbox.setMaximum(max_val)
                
                self.update_file_status()
                self.display_frame()
                return
                
            except ImportError:
                print("imageio v2 not available, trying tifffile...")
                # Fallback to tifffile
                self.load_tiff_with_tifffile(filename)
                return
            except Exception as e:
                print(f"imageio v2 failed: {e}, trying tifffile...")
                self.load_tiff_with_tifffile(filename)
                return
        except Exception as e:
            print(f"imageio failed: {e}, trying tifffile...")
            self.load_tiff_with_tifffile(filename)
    
    def load_tiff_with_tifffile(self, filename):
        """Fallback TIFF loading using tifffile library"""
        try:
            import tifffile
            with tifffile.TiffFile(filename) as tif:
                self.total_frames = len(tif.pages)
                print(f"Loaded TIFF stack with {self.total_frames} frames using tifffile")
            
            self.tiff_file_path = filename
            self.cap = None
            self.tiff_reader = None
            
            max_val = min(self.total_frames - 1, 2147483647)
            self.frame_slider.setMaximum(max_val)
            self.jump_spinbox.setMaximum(max_val)
            
            self.update_file_status()
            self.display_frame()
            
        except ImportError:
            QMessageBox.critical(self, "Error", 
                               "Cannot load TIFF files. Please install:\n"
                               "pip install imageio\n"
                               "or\n"
                               "pip install tifffile")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load TIFF: {str(e)}")
    
    def load_standard_video(self, filename):
        """Load standard video formats using OpenCV"""
        # Reset single image mode
        self.single_image_mode = False
        self.single_image = None
        
        self.cap = cv2.VideoCapture(filename)
        
        # Get frame count
        frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        if frame_count <= 0 or frame_count > 2147483647:
            self.total_frames = 1000
            QMessageBox.warning(self, "Warning", 
                              f"Could not determine frame count. Defaulting to {self.total_frames}.")
        else:
            self.total_frames = int(frame_count)
        
        # Enable and set navigation controls
        self.frame_slider.setEnabled(True)
        self.jump_spinbox.setEnabled(True)
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        
        # Set maximum values
        max_val = min(self.total_frames - 1, 2147483647)
        self.frame_slider.setMaximum(max_val)
        self.jump_spinbox.setMaximum(max_val)
        
        self.update_file_status()
        self.display_frame()
            
    def load_rois(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load ROI File", "", "ROI Files (*.zip *.roi)"
        )
        if filename:
            self.roi_path = filename
            self.rois = self.parse_imagej_roi(filename)
            
            self.update_file_status()
            self.update_display()
            
            if self.cap is None and self.tiff_reader is None and self.tiff_file_path is None and self.tiff_stack is None:
                QMessageBox.information(self, "ROIs Loaded", 
                                       f"Loaded {len(self.rois)} ROIs.\n\nLoad a video to visualize them.")
    
    def load_rois_from_folder(self):
        """Load individual ROI files from a folder"""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder Containing ROI Files"
        )
        if folder:
            # Find all .roi files
            roi_files = glob.glob(os.path.join(folder, "*.roi"))
            
            if len(roi_files) == 0:
                QMessageBox.warning(self, "Warning", "No .roi files found in selected folder!")
                return
            
            # Sort files by name - use natural sorting for numbers
            import re
            def natural_sort_key(s):
                return [int(text) if text.isdigit() else text.lower()
                        for text in re.split('([0-9]+)', s)]
            
            roi_files.sort(key=natural_sort_key)
            
            # Show first and last file names for verification
            print(f"First ROI file: {os.path.basename(roi_files[0])}")
            print(f"Last ROI file: {os.path.basename(roi_files[-1])}")
            
            # Parse each ROI file
            self.rois = []
            for roi_file in roi_files:
                with open(roi_file, 'rb') as f:
                    roi_data = self.read_roi_file(f.read())
                    if roi_data is not None:
                        self.rois.append(roi_data)
                    else:
                        # Add None for failed reads to maintain frame correspondence
                        self.rois.append(None)
                        print(f"Warning: Could not read {roi_file}")
            
            self.roi_path = folder
            self.update_file_status()
            self.update_display()
            
            success_msg = f"Loaded {len([r for r in self.rois if r is not None])} ROIs from {len(roi_files)} files"
            success_msg += f"\n\nFirst: {os.path.basename(roi_files[0])}"
            success_msg += f"\nLast: {os.path.basename(roi_files[-1])}"
            
            if self.cap is None and self.tiff_reader is None and self.tiff_file_path is None and self.tiff_stack is None:
                success_msg += "\n\nNote: Load a video to visualize the ROIs."
            else:
                # Check for mismatch
                if len(self.rois) != self.total_frames:
                    success_msg += f"\n\n⚠ WARNING: Frame/ROI mismatch!"
                    success_msg += f"\nVideo frames: {self.total_frames}"
                    success_msg += f"\nROI files: {len(self.rois)}"
                    success_msg += f"\n\nUse 'ROI Offset' to align them."
            
            QMessageBox.information(self, "Success", success_msg)
    
    def toggle_draw_roi_mode(self):
        """Toggle ROI drawing mode on/off"""
        self.drawing_mode = self.draw_roi_btn.isChecked()
        
        if self.drawing_mode:
            # Entering draw mode
            self.roi_drawing_points = []
            self.draw_roi_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
            self.draw_roi_btn.setText("✏️ Drawing... (Right-click to finish)")
            
            # Show instructions
            QMessageBox.information(
                self,
                "Draw ROI Mode",
                "ROI Drawing Instructions:\n\n"
                "• Left-click: Add points to ROI outline\n"
                "• Right-click: Finish and close ROI\n"
                "• Esc key: Cancel drawing\n\n"
                "Tips:\n"
                "- Click around the larva outline\n"
                "- Points will automatically connect\n"
                "- Need at least 3 points to create ROI\n"
                "- Zoom in for precision!\n\n"
                "The ROI will be saved for the current frame."
            )
        else:
            # Exiting draw mode
            self.roi_drawing_points = []
            self.draw_roi_btn.setStyleSheet("")
            self.draw_roi_btn.setText("✏️ Draw ROI")
            self.update_display()
    
    def finish_roi_drawing(self):
        """Complete the ROI drawing and save it"""
        if len(self.roi_drawing_points) < 3:
            QMessageBox.warning(self, "Not Enough Points", 
                              "Need at least 3 points to create an ROI.\n"
                              f"Current points: {len(self.roi_drawing_points)}")
            return
        
        # Convert points to numpy array
        roi_array = np.array(self.roi_drawing_points, dtype=np.int32)
        
        # Calculate current frame's ROI index
        roi_index = self.current_frame + self.roi_offset_spinbox.value()
        
        # Ensure rois list is large enough
        while len(self.rois) <= roi_index:
            self.rois.append(None)
        
        # Save the ROI
        self.rois[roi_index] = roi_array
        
        # Set ROI path to indicate ROIs are loaded
        if not self.roi_path:
            self.roi_path = f"Drawn ROI (Frame {self.current_frame})"
        
        # Exit drawing mode
        self.drawing_mode = False
        self.draw_roi_btn.setChecked(False)
        self.draw_roi_btn.setStyleSheet("")
        self.draw_roi_btn.setText("✏️ Draw ROI")
        self.roi_drawing_points = []
        
        # Update display and status
        self.update_file_status()
        self.update_display()
        
        # Show success message
        QMessageBox.information(
            self,
            "ROI Created",
            f"✓ ROI successfully created for frame {self.current_frame}!\n\n"
            f"Points: {len(roi_array)}\n"
            f"ROI saved to frame index {roi_index}\n\n"
            "You can now:\n"
            "• Auto-detect or manually set head/tail\n"
            "• Initialize segments\n"
            "• Draw more ROIs for other frames\n"
            "• Save session to keep the drawn ROI"
        )
    
    def cancel_roi_drawing(self):
        """Cancel ROI drawing"""
        if not self.drawing_mode:
            return
        
        self.roi_drawing_points = []
        self.drawing_mode = False
        self.draw_roi_btn.setChecked(False)
        self.draw_roi_btn.setStyleSheet("")
        self.draw_roi_btn.setText("✏️ Draw ROI")
        self.update_display()
    
    def save_drawn_rois(self):
        """Save drawn ROIs to ImageJ-compatible .roi files"""
        if len(self.rois) == 0:
            QMessageBox.warning(self, "No ROIs", "No ROIs to save!")
            return
        
        # Ask for save location
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder to Save ROI Files"
        )
        
        if not folder:
            return
        
        # Save each ROI as a separate .roi file
        saved_count = 0
        for frame_idx, roi in enumerate(self.rois):
            if roi is not None and len(roi) > 0:
                filename = os.path.join(folder, f"roi_{frame_idx:04d}.roi")
                self.write_roi_file(filename, roi)
                saved_count += 1
        
        QMessageBox.information(
            self,
            "ROIs Saved",
            f"Successfully saved {saved_count} ROI files to:\n{folder}\n\n"
            "These files are compatible with ImageJ."
        )
    
    def write_roi_file(self, filepath, roi_points):
        """Write ROI points to ImageJ-compatible .roi file format"""
        # Simplified ImageJ ROI format writer (polygon type)
        import struct
        
        n_points = len(roi_points)
        roi_type = 0  # 0 = polygon
        
        # Get bounding box
        x_coords = roi_points[:, 0]
        y_coords = roi_points[:, 1]
        left = int(np.min(x_coords))
        top = int(np.min(y_coords))
        right = int(np.max(x_coords))
        bottom = int(np.max(y_coords))
        width = right - left
        height = bottom - top
        
        with open(filepath, 'wb') as f:
            # Header
            f.write(b'Iout')  # Magic number
            f.write(struct.pack('>h', 228))  # Version
            f.write(struct.pack('>h', roi_type))  # Type
            f.write(struct.pack('>h', top))  # Top
            f.write(struct.pack('>h', left))  # Left
            f.write(struct.pack('>h', bottom))  # Bottom
            f.write(struct.pack('>h', right))  # Right
            f.write(struct.pack('>h', n_points))  # N points
            
            # Padding and coordinates offset
            f.write(struct.pack('>f', 0))  # x1
            f.write(struct.pack('>f', 0))  # y1
            f.write(struct.pack('>f', 0))  # x2
            f.write(struct.pack('>f', 0))  # y2
            f.write(struct.pack('>h', 0))  # stroke width
            f.write(struct.pack('>h', 0))  # shape roi size
            f.write(struct.pack('>h', 0))  # stroke color
            f.write(struct.pack('>h', 0))  # fill color
            f.write(struct.pack('>h', 0))  # subtype
            f.write(struct.pack('>h', 0))  # options
            f.write(struct.pack('>B', 0))  # arrow style
            f.write(struct.pack('>B', 0))  # arrow head size
            f.write(struct.pack('>h', 0))  # rounded rect arc size
            f.write(struct.pack('>i', 0))  # position
            f.write(struct.pack('>i', 64))  # header2 offset
            
            # Coordinates
            for point in roi_points:
                x = int(point[0] - left)
                y = int(point[1] - top)
                f.write(struct.pack('>h', x))
            for point in roi_points:
                x = int(point[0] - left)
                y = int(point[1] - top)
                f.write(struct.pack('>h', y))
    
    def show_auto_detect_roi_dialog(self):
        """Show dialog for auto-detecting ROI with parameters"""
        from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QSlider, QRadioButton, QButtonGroup
        
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Auto-detect ROI")
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout()
        
        # Instructions
        info_label = QLabel(
            "Automatically detect larva outline using threshold.\n\n"
            "Adjust parameters and preview the result:"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Threshold method
        method_group = QGroupBox("Threshold Method")
        method_layout = QVBoxLayout()
        
        self.thresh_method_group = QButtonGroup()
        
        otsu_radio = QRadioButton("Otsu (automatic)")
        otsu_radio.setToolTip("Automatic threshold based on image histogram")
        otsu_radio.setChecked(True)
        self.thresh_method_group.addButton(otsu_radio, 0)
        method_layout.addWidget(otsu_radio)
        
        manual_radio = QRadioButton("Manual threshold")
        manual_radio.setToolTip("Set threshold value manually")
        self.thresh_method_group.addButton(manual_radio, 1)
        method_layout.addWidget(manual_radio)
        
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)
        
        # Manual threshold slider
        thresh_group = QGroupBox("Manual Threshold Value")
        thresh_layout = QVBoxLayout()
        
        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setMinimum(0)
        self.thresh_slider.setMaximum(255)
        self.thresh_slider.setValue(128)
        self.thresh_slider.setEnabled(False)
        thresh_layout.addWidget(self.thresh_slider)
        
        self.thresh_value_label = QLabel("Value: 128")
        thresh_layout.addWidget(self.thresh_value_label)
        
        thresh_group.setLayout(thresh_layout)
        layout.addWidget(thresh_group)
        
        # Enable slider when manual is selected
        def on_method_changed():
            is_manual = self.thresh_method_group.checkedId() == 1
            self.thresh_slider.setEnabled(is_manual)
        
        self.thresh_method_group.buttonClicked.connect(on_method_changed)
        
        # Update label when slider moves
        def on_slider_changed(value):
            self.thresh_value_label.setText(f"Value: {value}")
        
        self.thresh_slider.valueChanged.connect(on_slider_changed)
        
        # Invert option
        self.invert_checkbox = QCheckBox("Invert (for dark larvae on bright background)")
        self.invert_checkbox.setChecked(False)
        layout.addWidget(self.invert_checkbox)
        
        # Min/max size filters
        size_group = QGroupBox("Size Filters")
        size_layout = QVBoxLayout()
        
        min_size_layout = QHBoxLayout()
        min_size_layout.addWidget(QLabel("Min area (pixels²):"))
        self.min_size_spinbox = QSpinBox()
        self.min_size_spinbox.setMinimum(0)
        self.min_size_spinbox.setMaximum(1000000)
        self.min_size_spinbox.setValue(500)
        self.min_size_spinbox.setToolTip("Ignore contours smaller than this")
        min_size_layout.addWidget(self.min_size_spinbox)
        size_layout.addLayout(min_size_layout)
        
        max_size_layout = QHBoxLayout()
        max_size_layout.addWidget(QLabel("Max area (pixels²):"))
        self.max_size_spinbox = QSpinBox()
        self.max_size_spinbox.setMinimum(0)
        self.max_size_spinbox.setMaximum(1000000)
        self.max_size_spinbox.setValue(100000)
        self.max_size_spinbox.setToolTip("Ignore contours larger than this")
        max_size_layout.addWidget(self.max_size_spinbox)
        size_layout.addLayout(max_size_layout)
        
        size_group.setLayout(size_layout)
        layout.addWidget(size_group)
        
        # NEW: Preprocessing options for difficult images
        preprocess_group = QGroupBox("Preprocessing (for low-contrast images)")
        preprocess_layout = QVBoxLayout()
        
        self.enhance_contrast_checkbox = QCheckBox("✓ Enhance contrast (CLAHE)")
        self.enhance_contrast_checkbox.setChecked(True)
        self.enhance_contrast_checkbox.setToolTip("Improves detection for low-contrast images\nRECOMMENDED for difficult images")
        preprocess_layout.addWidget(self.enhance_contrast_checkbox)
        
        self.use_adaptive_checkbox = QCheckBox("Use adaptive threshold (for varying lighting)")
        self.use_adaptive_checkbox.setChecked(False)
        self.use_adaptive_checkbox.setToolTip("Better for images with uneven illumination\nOverrides Otsu/Manual")
        preprocess_layout.addWidget(self.use_adaptive_checkbox)
        
        blur_layout = QHBoxLayout()
        blur_layout.addWidget(QLabel("Blur amount:"))
        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setMinimum(0)
        self.blur_slider.setMaximum(50)
        self.blur_slider.setValue(20)  # 2.0 sigma
        self.blur_slider.setToolTip("Reduces noise before detection\nHigher = more blur")
        blur_layout.addWidget(self.blur_slider)
        self.blur_value_label = QLabel("2.0")
        blur_layout.addWidget(self.blur_value_label)
        preprocess_layout.addLayout(blur_layout)
        
        def on_blur_changed(value):
            sigma = value / 10.0
            self.blur_value_label.setText(f"{sigma:.1f}")
        
        self.blur_slider.valueChanged.connect(on_blur_changed)
        
        preprocess_group.setLayout(preprocess_layout)
        layout.addWidget(preprocess_group)
        
        # Preview button
        preview_btn = QPushButton("🔍 Preview on Current Frame")
        preview_btn.clicked.connect(lambda: self.preview_auto_detect_roi(
            self.thresh_method_group.checkedId() == 0,  # use_otsu
            self.thresh_slider.value(),
            self.invert_checkbox.isChecked(),
            self.min_size_spinbox.value(),
            self.max_size_spinbox.value(),
            self.enhance_contrast_checkbox.isChecked(),
            self.blur_slider.value() / 10.0,
            self.use_adaptive_checkbox.isChecked()
        ))
        layout.addWidget(preview_btn)
        
        # Apply to options
        apply_group = QGroupBox("Apply To")
        apply_layout = QVBoxLayout()
        
        self.apply_mode_group = QButtonGroup()
        
        current_radio = QRadioButton("Current frame only")
        current_radio.setChecked(True)
        self.apply_mode_group.addButton(current_radio, 0)
        apply_layout.addWidget(current_radio)
        
        all_radio = QRadioButton("All frames (batch mode)")
        all_radio.setToolTip("Detect ROI for all frames in video")
        self.apply_mode_group.addButton(all_radio, 1)
        apply_layout.addWidget(all_radio)
        
        apply_group.setLayout(apply_layout)
        layout.addWidget(apply_group)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            # Apply auto-detection
            use_otsu = self.thresh_method_group.checkedId() == 0
            thresh_value = self.thresh_slider.value()
            invert = self.invert_checkbox.isChecked()
            min_size = self.min_size_spinbox.value()
            max_size = self.max_size_spinbox.value()
            enhance_contrast = self.enhance_contrast_checkbox.isChecked()
            blur_sigma = self.blur_slider.value() / 10.0
            use_adaptive = self.use_adaptive_checkbox.isChecked()
            apply_to_all = self.apply_mode_group.checkedId() == 1
            
            if apply_to_all:
                self.auto_detect_roi_batch(use_otsu, thresh_value, invert, min_size, max_size,
                                          enhance_contrast, blur_sigma, use_adaptive)
            else:
                self.auto_detect_roi_single(use_otsu, thresh_value, invert, min_size, max_size,
                                           enhance_contrast, blur_sigma, use_adaptive)
    
    def preview_auto_detect_roi(self, use_otsu, thresh_value, invert, min_size, max_size,
                                 enhance_contrast=True, blur_sigma=2.0, use_adaptive=False):
        """Preview ROI detection on current frame"""
        if self.current_display_frame is None:
            QMessageBox.warning(self, "No Frame", "Load a video or image first!")
            return
        
        # Detect ROI
        roi = self.detect_roi_from_threshold(
            self.current_display_frame, use_otsu, thresh_value, invert, min_size, max_size,
            enhance_contrast, blur_sigma, use_adaptive
        )
        
        if roi is None:
            QMessageBox.warning(
                self, "No Contour Found",
                "Could not detect a valid contour with current parameters.\n\n"
                "Try adjusting:\n"
                "• Threshold value\n"
                "• Invert option\n"
                "• Size filters"
            )
            return
        
        # Show preview with ROI overlay
        preview_frame = self.current_display_frame.copy()
        cv2.polylines(preview_frame, [roi.astype(np.int32)], True, (0, 255, 0), 3)
        
        # Add info text
        area = cv2.contourArea(roi)
        cv2.putText(preview_frame, f"Detected contour area: {area:.0f} px^2", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(preview_frame, f"Points: {len(roi)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display preview
        height, width = preview_frame.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(preview_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image.rgbSwapped())
        
        # Scale to fit
        scaled = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)
        
        QMessageBox.information(
            self, "Preview",
            f"✓ ROI detected successfully!\n\n"
            f"Area: {area:.0f} pixels²\n"
            f"Points: {len(roi)}\n\n"
            "If this looks good, click OK to apply.\n"
            "Otherwise, adjust parameters and preview again."
        )
    
    def detect_roi_from_threshold(self, frame, use_otsu, thresh_value, invert, min_size, max_size, 
                                   enhance_contrast=True, blur_sigma=2.0, use_adaptive=False):
        """Detect ROI contour from image using threshold with enhanced preprocessing"""
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # ENHANCEMENT 1: Contrast enhancement for low-contrast images
        if enhance_contrast:
            # Use CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
        
        # ENHANCEMENT 2: Gaussian blur to reduce noise
        if blur_sigma > 0:
            gray = cv2.GaussianBlur(gray, (5, 5), blur_sigma)
        
        # Apply threshold
        if use_adaptive:
            # ENHANCEMENT 3: Adaptive threshold for varying lighting
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
        elif use_otsu:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)
        
        # Invert if needed
        if invert:
            binary = cv2.bitwise_not(binary)
        
        # ENHANCEMENT 4: More aggressive morphology for difficult images
        # Increased kernel size and iterations for smoother results
        kernel = np.ones((7,7), np.uint8)  # Larger kernel (was 5x5)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=4)  # More iterations (was 3)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=3)   # More iterations (was 2)
        
        # Fill holes in the binary image
        contours_fill, _ = cv2.findContours(binary.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_fill:
            cv2.drawContours(binary, [cnt], 0, 255, -1)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        # Filter by size and find largest valid contour
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_size <= area <= max_size:
                valid_contours.append(contour)
        
        if len(valid_contours) == 0:
            return None
        
        # Get largest valid contour
        largest_contour = max(valid_contours, key=cv2.contourArea)
        
        # ENHANCEMENT: Apply contour smoothing
        # Smooth the contour points using convolution
        if len(largest_contour) > 10:
            # Convert to simple array format
            contour_points = largest_contour.squeeze()
            
            # Apply moving average smoothing
            window_size = 5  # Smooth over 5 neighboring points
            kernel = np.ones(window_size) / window_size
            
            # Smooth x and y coordinates separately
            x_smooth = np.convolve(contour_points[:, 0], kernel, mode='same')
            y_smooth = np.convolve(contour_points[:, 1], kernel, mode='same')
            
            # Handle boundary conditions (wrap around for closed contour)
            if window_size > 1:
                half_window = window_size // 2
                # Re-smooth the boundaries by wrapping
                x_coords = np.concatenate([contour_points[-half_window:, 0], 
                                          contour_points[:, 0], 
                                          contour_points[:half_window, 0]])
                y_coords = np.concatenate([contour_points[-half_window:, 1], 
                                          contour_points[:, 1], 
                                          contour_points[:half_window, 1]])
                
                x_smooth = np.convolve(x_coords, kernel, mode='valid')
                y_smooth = np.convolve(y_coords, kernel, mode='valid')
            
            # Reconstruct contour
            largest_contour = np.column_stack([x_smooth, y_smooth]).astype(np.int32)
            largest_contour = largest_contour.reshape(-1, 1, 2)
        
        # Simplify contour (reduce points while maintaining shape)
        # Increased epsilon for smoother outline (was 0.002)
        epsilon = 0.010 * cv2.arcLength(largest_contour, True)  # Increased from 0.008
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Convert to expected format (N, 2)
        roi_points = approx.squeeze()
        
        # Ensure 2D array
        if len(roi_points.shape) == 1:
            roi_points = roi_points.reshape(-1, 2)
        
        return roi_points
    
    def auto_detect_roi_single(self, use_otsu, thresh_value, invert, min_size, max_size,
                                enhance_contrast=True, blur_sigma=2.0, use_adaptive=False):
        """Auto-detect ROI for current frame"""
        if self.current_display_frame is None:
            QMessageBox.warning(self, "No Frame", "Load a video or image first!")
            return
        
        roi = self.detect_roi_from_threshold(
            self.current_display_frame, use_otsu, thresh_value, invert, min_size, max_size,
            enhance_contrast, blur_sigma, use_adaptive
        )
        
        if roi is None:
            QMessageBox.warning(self, "Detection Failed", 
                              "Could not detect ROI. Try adjusting parameters.\n\n"
                              "For low-contrast images:\n"
                              "• Enable 'Enhance contrast'\n"
                              "• Try 'Adaptive threshold'\n"
                              "• Adjust blur amount")
            return
        
        # Save ROI for current frame
        roi_index = self.current_frame + self.roi_offset_spinbox.value()
        
        # Ensure rois list is large enough
        while len(self.rois) <= roi_index:
            self.rois.append(None)
        
        self.rois[roi_index] = roi
        
        # Set ROI path
        if not self.roi_path:
            self.roi_path = "Auto-detected ROI"
        
        # Update display
        self.update_file_status()
        self.update_display()
        
        area = cv2.contourArea(roi)
        QMessageBox.information(
            self, "ROI Detected",
            f"✓ ROI successfully detected for frame {self.current_frame}!\n\n"
            f"Area: {area:.0f} pixels²\n"
            f"Points: {len(roi)}"
        )
    
    def auto_detect_roi_batch(self, use_otsu, thresh_value, invert, min_size, max_size,
                               enhance_contrast=True, blur_sigma=2.0, use_adaptive=False):
        """Auto-detect ROI for all frames with progress"""
        from PyQt5.QtWidgets import QProgressDialog
        from PyQt5.QtCore import Qt
        
        if self.cap is None and self.tiff_stack is None and not self.single_image_mode:
            QMessageBox.warning(self, "No Video", "Load a video first for batch processing!")
            return
        
        # Confirm
        reply = QMessageBox.question(
            self, "Batch Auto-detect",
            f"Auto-detect ROI for all {self.total_frames} frames?\n\n"
            "This may take a few minutes.\n"
            "You can review results after completion.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
        
        # Create progress dialog
        progress = QProgressDialog("Detecting ROIs...", "Cancel", 0, self.total_frames, self)
        progress.setWindowTitle("Auto-detecting ROIs")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        
        # Ensure rois list is large enough
        while len(self.rois) < self.total_frames:
            self.rois.append(None)
        
        success_count = 0
        failed_frames = []
        
        # Process each frame
        for frame_idx in range(self.total_frames):
            if progress.wasCanceled():
                break
            
            progress.setValue(frame_idx)
            progress.setLabelText(f"Detecting ROI for frame {frame_idx+1}/{self.total_frames}...")
            
            # Get frame
            if self.single_image_mode:
                frame = self.single_image.copy()
            elif self.tiff_stack is not None:
                frame = self.tiff_stack[frame_idx]
                # Convert to BGR if needed
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = self.cap.read()
                if not ret:
                    failed_frames.append(frame_idx)
                    continue
            
            # Detect ROI
            roi = self.detect_roi_from_threshold(frame, use_otsu, thresh_value, invert, min_size, max_size,
                                                enhance_contrast, blur_sigma, use_adaptive)
            
            if roi is not None:
                roi_index = frame_idx + self.roi_offset_spinbox.value()
                if 0 <= roi_index < len(self.rois):
                    self.rois[roi_index] = roi
                    success_count += 1
            else:
                failed_frames.append(frame_idx)
        
        progress.setValue(self.total_frames)
        
        # Set ROI path
        if not self.roi_path:
            self.roi_path = "Auto-detected ROIs (batch)"
        
        # Update display
        self.update_file_status()
        self.update_display()
        
        # Show results
        if len(failed_frames) > 0:
            failed_msg = f"\n\nFailed frames: {len(failed_frames)}\n"
            if len(failed_frames) <= 10:
                failed_msg += f"Frames: {failed_frames}"
            else:
                failed_msg += f"First 10: {failed_frames[:10]}..."
        else:
            failed_msg = ""
        
        QMessageBox.information(
            self, "Batch Detection Complete",
            f"✓ ROI detection complete!\n\n"
            f"Successfully detected: {success_count}/{self.total_frames}{failed_msg}\n\n"
            "You can now:\n"
            "• Review ROIs by navigating frames\n"
            "• Manually adjust any frames\n"
            "• Proceed with analysis"
        )
            
    def parse_imagej_roi(self, filepath):
        """Parse ImageJ ROI file (zip containing multiple .roi files)"""
        rois = []
        
        if filepath.endswith('.zip'):
            with zipfile.ZipFile(filepath, 'r') as zip_file:
                roi_files = [f for f in zip_file.namelist() if f.endswith('.roi')]
                roi_files.sort()  # Sort to maintain frame order
                
                for roi_file in roi_files:
                    with zip_file.open(roi_file) as f:
                        roi_data = self.read_roi_file(f.read())
                        if roi_data is not None:
                            rois.append(roi_data)
        else:
            # Single ROI file
            with open(filepath, 'rb') as f:
                roi_data = self.read_roi_file(f.read())
                if roi_data is not None:
                    rois.append(roi_data)
        
        return rois
    
    def read_roi_file(self, data):
        """Read a single ImageJ .roi file"""
        try:
            # ImageJ ROI file format
            # Header: 4 bytes "Iout"
            if data[:4] != b'Iout':
                return None
            
            # Get ROI type and coordinates
            version = struct.unpack('>h', data[4:6])[0]
            roi_type = struct.unpack('>b', data[6:7])[0]
            top = struct.unpack('>h', data[8:10])[0]
            left = struct.unpack('>h', data[10:12])[0]
            bottom = struct.unpack('>h', data[12:14])[0]
            right = struct.unpack('>h', data[14:16])[0]
            n_coords = struct.unpack('>h', data[16:18])[0]
            
            # Read coordinates
            if n_coords > 0:
                coords_start = 64  # Standard offset for coordinates
                x_coords = []
                y_coords = []
                
                for i in range(n_coords):
                    x = struct.unpack('>h', data[coords_start + i*2:coords_start + i*2 + 2])[0]
                    y = struct.unpack('>h', data[coords_start + n_coords*2 + i*2:coords_start + n_coords*2 + i*2 + 2])[0]
                    x_coords.append(left + x)
                    y_coords.append(top + y)
                
                return np.column_stack([x_coords, y_coords])
            
        except Exception as e:
            print(f"Error reading ROI: {e}")
            return None
            
        return None
    
    def update_file_status(self):
        status = []
        if self.video_path:
            filename = self.video_path.split('/')[-1]
            if self.single_image_mode:
                status.append(f"Image: {filename}")
            else:
                status.append(f"Video: {filename}")
                status.append(f"Frames: {self.total_frames}")
        if self.roi_path:
            status.append(f"ROIs: {len(self.rois)} loaded")
        
        self.file_status_label.setText("\n".join(status) if status else "No files loaded")
        
    def display_frame(self):
        # Handle single image mode
        if self.single_image_mode and self.single_image is not None:
            self.current_display_frame = self.single_image.copy()
            self.update_display()
            return
        
        # Handle in-memory TIFF stack (imageio v3)
        if self.tiff_stack is not None:
            try:
                frame = self.tiff_stack[self.current_frame]
                
                # Convert to BGR for OpenCV compatibility
                if len(frame.shape) == 2:  # Grayscale
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif len(frame.shape) == 3:
                    if frame.shape[2] == 4:  # RGBA
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                    elif frame.shape[2] == 3:
                        # Assume RGB, convert to BGR
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Handle 16-bit images
                if frame.dtype == np.uint16:
                    frame = (frame / 256).astype(np.uint8)
                elif frame.dtype != np.uint8:
                    # Normalize other types to 0-255
                    frame = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
                
                self.current_display_frame = frame
                self.update_display()
                return
                
            except Exception as e:
                print(f"Error reading TIFF frame {self.current_frame} from stack: {e}")
                return
        
        # Handle TIFF files with on-demand reading
        if self.tiff_file_path is not None and self.tiff_stack is None:
            try:
                # Try imageio v3 first
                import imageio.v3 as iio
                frame = iio.imread(self.tiff_file_path, index=self.current_frame)
                
                # Convert to BGR for OpenCV compatibility
                if len(frame.shape) == 2:  # Grayscale
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:  # RGBA
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                elif frame.shape[2] == 3:
                    # Check if RGB, convert to BGR
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Handle 16-bit images
                if frame.dtype == np.uint16:
                    frame = (frame / 256).astype(np.uint8)
                
                self.current_display_frame = frame
                self.update_display()
                return
                
            except (ImportError, AttributeError):
                # Try imageio v2
                try:
                    import imageio
                    reader = imageio.get_reader(self.tiff_file_path)
                    frame = reader.get_data(self.current_frame)
                    reader.close()
                    
                    # Convert to BGR
                    if len(frame.shape) == 2:  # Grayscale
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    elif frame.shape[2] == 4:  # RGBA
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                    elif frame.shape[2] == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    if frame.dtype == np.uint16:
                        frame = (frame / 256).astype(np.uint8)
                    
                    self.current_display_frame = frame
                    self.update_display()
                    return
                    
                except ImportError:
                    pass  # Fall through to tifffile
            
            # Fallback to tifffile
            try:
                import tifffile
                with tifffile.TiffFile(self.tiff_file_path) as tif:
                    frame = tif.pages[self.current_frame].asarray()
                    
                    # Convert to 8-bit BGR
                    if frame.dtype == np.uint16:
                        frame = (frame / 256).astype(np.uint8)
                    
                    if len(frame.shape) == 2:  # Grayscale
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    elif len(frame.shape) == 3:
                        if frame.shape[2] == 4:  # RGBA
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                        elif frame.shape[2] == 3:  # RGB
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    self.current_display_frame = frame
                    self.update_display()
                return
            except Exception as e:
                print(f"Error reading TIFF frame {self.current_frame}: {e}")
                return
        
        # Handle TIFF files with imageio v2 reader (legacy)
        if self.tiff_reader is not None:
            try:
                frame = self.tiff_reader.get_data(self.current_frame)
                # Convert to BGR for OpenCV compatibility
                if len(frame.shape) == 2:  # Grayscale
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:  # RGBA
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                elif frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                if frame.dtype == np.uint16:
                    frame = (frame / 256).astype(np.uint8)
                
                self.current_display_frame = frame
                self.update_display()
                return
            except Exception as e:
                print(f"Error reading TIFF frame {self.current_frame}: {e}")
                return
        
        # Handle standard video
        if self.cap is None:
            return
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        
        if ret:
            self.current_display_frame = frame.copy()
            self.update_display()
            
    def update_display(self):
        # If no video/image loaded yet, just return
        if (self.cap is None and self.tiff_reader is None and self.tiff_file_path is None and 
            self.tiff_stack is None and not self.single_image_mode):
            return
            
        # If no frame has been displayed yet, display the current frame first
        if self.current_display_frame is None:
            self.display_frame()
            return
            
        frame = self.current_display_frame.copy()
        
        # Calculate ROI index with offset
        roi_index = self.current_frame + self.roi_offset_spinbox.value()
        
        # Draw ROI if available and enabled for CURRENT FRAME ONLY
        if 0 <= roi_index < len(self.rois) and self.show_roi_cb.isChecked():
            roi = self.rois[roi_index]
            if roi is not None and len(roi) > 0:
                cv2.polylines(frame, [roi.astype(np.int32)], True, (0, 255, 0), 2)
        
        # Draw ROI being drawn (in drawing mode)
        if self.drawing_mode and len(self.roi_drawing_points) > 0:
            points = np.array(self.roi_drawing_points, dtype=np.int32)
            
            # Draw points
            for point in points:
                cv2.circle(frame, tuple(point), 5, (255, 165, 0), -1)  # Orange dots
            
            # Draw lines connecting points
            if len(points) > 1:
                cv2.polylines(frame, [points], False, (255, 165, 0), 2)  # Orange line
            
            # Draw line from last point to cursor would go here if we tracked cursor
            
            # Show instruction
            cv2.putText(frame, "Drawing ROI - Left click: add point | Right click: finish", 
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 165, 0), 2)
        
        # Check ROI and show snap preview if dragging
        if 0 <= roi_index < len(self.rois) and self.rois[roi_index] is not None:
            roi = self.rois[roi_index]
            if len(roi) > 0:
                
                # If dragging and snap is enabled, show preview of snap target
                if self.dragging and self.snap_to_roi_cb.isChecked():
                    # Show a small circle at the snap target for visual feedback
                    if self.drag_target == 'head' and self.current_frame in self.head_positions:
                        snap_pos = self.head_positions[self.current_frame]
                        cv2.circle(frame, tuple(snap_pos.astype(int)), 4, (255, 255, 0), -1)  # Yellow dot
                    elif self.drag_target == 'tail' and self.current_frame in self.tail_positions:
                        snap_pos = self.tail_positions[self.current_frame]
                        cv2.circle(frame, tuple(snap_pos.astype(int)), 4, (255, 255, 0), -1)  # Yellow dot
                
                # Auto-detect or retrieve head/tail (but not if dragging)
                if self.auto_detect_cb.isChecked() and not self.dragging:
                    # Only auto-detect if positions don't exist yet
                    if self.current_frame not in self.head_positions or self.current_frame not in self.tail_positions:
                        head, tail = self.detect_endpoints(roi)
                        self.head_positions[self.current_frame] = head
                        self.tail_positions[self.current_frame] = tail
                
                # Draw head and tail for CURRENT FRAME ONLY
                if self.current_frame in self.head_positions and self.current_frame in self.tail_positions:
                    head = self.head_positions[self.current_frame]
                    tail = self.tail_positions[self.current_frame]
                    
                    # Draw head (red circle)
                    cv2.circle(frame, tuple(head.astype(int)), 8, (0, 0, 255), -1)
                    cv2.putText(frame, "H", tuple((head + 15).astype(int)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # If dragging head, show larger highlight
                    if self.dragging and self.drag_target == 'head':
                        cv2.circle(frame, tuple(head.astype(int)), 12, (0, 0, 255), 2)
                    
                    # Draw tail (blue circle)
                    cv2.circle(frame, tuple(tail.astype(int)), 8, (255, 0, 0), -1)
                    cv2.putText(frame, "T", tuple((tail + 15).astype(int)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
                    # If dragging tail, show larger highlight
                    if self.dragging and self.drag_target == 'tail':
                        cv2.circle(frame, tuple(tail.astype(int)), 12, (255, 0, 0), 2)
                    
                    # Draw body axis if enabled
                    if self.show_axis_cb.isChecked():
                        cv2.line(frame, tuple(head.astype(int)), tuple(tail.astype(int)), 
                                (255, 255, 0), 2)
                        
                        # Calculate and display angle
                        angle = self.calculate_body_angle(head, tail)
                        self.body_angles[self.current_frame] = angle
                        
                        # Draw angle info (OpenCV doesn't support ° symbol, use 'deg')
                        cv2.putText(frame, f"Angle: {angle:.1f} deg", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Draw reference line (horizontal) for angle visualization
                        ref_length = 30
                        head_int = tuple(head.astype(int))
                        ref_end = (int(head[0] + ref_length), int(head[1]))
                        cv2.line(frame, head_int, ref_end, (128, 128, 128), 1)
                        cv2.putText(frame, "0°", (ref_end[0] + 5, ref_end[1] + 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
                        
                        # Add angle arc visualization
                        dx = tail[0] - head[0]
                        dy = tail[1] - head[1]
                        angle_rad = np.arctan2(dy, dx)
                        
                        # Draw small arc
                        arc_radius = 40
                        if abs(angle_rad) > 0.1:  # Only draw if angle is significant
                            # Draw arc from 0 to angle
                            start_angle = 0
                            end_angle = int(np.degrees(angle_rad))
                            
                            cv2.ellipse(frame, head_int, (arc_radius, arc_radius), 0,
                                       start_angle, end_angle, (255, 200, 0), 1)
                    
                    # Draw midline if enabled for CURRENT FRAME ONLY
                    if self.show_midline_cb.isChecked():
                        midline = self.calculate_midline(roi, head, tail)
                        if midline is not None:
                            self.midline_points[self.current_frame] = midline
                            cv2.polylines(frame, [midline.astype(np.int32)], False, (255, 0, 255), 2)
        
        # Draw segment points if enabled
        if self.show_segments and self.current_frame in self.segment_points:
            self.draw_segment_points(frame, self.current_frame)
        
        # Convert to QImage
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # Apply zoom and pan transformations
        if self.zoom_level != 1.0 or self.pan_offset_x != 0 or self.pan_offset_y != 0:
            # Get label size
            label_size = self.video_label.size()
            
            # Calculate zoomed size
            zoomed_w = int(w * self.zoom_level)
            zoomed_h = int(h * self.zoom_level)
            
            # Scale pixmap
            scaled_pixmap = pixmap.scaled(zoomed_w, zoomed_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Create a pixmap the size of the label
            display_pixmap = QPixmap(label_size)
            display_pixmap.fill(Qt.black)
            
            # Calculate position with pan offset
            from PyQt5.QtGui import QPainter
            painter = QPainter(display_pixmap)
            
            # Center the zoomed image with pan offset
            x_pos = int((label_size.width() - zoomed_w) / 2 + self.pan_offset_x)
            y_pos = int((label_size.height() - zoomed_h) / 2 + self.pan_offset_y)
            
            painter.drawPixmap(x_pos, y_pos, scaled_pixmap)
            painter.end()
            
            self.video_label.setPixmap(display_pixmap)
        else:
            # No zoom/pan, just display normally
            self.video_label.setPixmap(pixmap)
        
        # Update frame label with alignment info
        roi_index = self.current_frame + self.roi_offset_spinbox.value()
        roi_info = ""
        if 0 <= roi_index < len(self.rois):
            if self.rois[roi_index] is not None:
                roi_info = f" | ROI[{roi_index}]: {len(self.rois[roi_index])} pts"
            else:
                roi_info = f" | ROI[{roi_index}]: None"
        else:
            roi_info = f" | ROI[{roi_index}]: Out of range"
        
        zoom_info = f" | Zoom: {self.zoom_level:.1f}x" if self.zoom_level != 1.0 else ""
        self.frame_label.setText(f"Frame: {self.current_frame + 1}/{self.total_frames}{roi_info}{zoom_info}")
        
        # Update position status
        if self.current_frame in self.head_positions and self.current_frame in self.tail_positions:
            # Check if manually set (simple heuristic: if auto-detect is off or manual mode was used)
            if not self.auto_detect_cb.isChecked() or self.manual_adjust_mode:
                self.position_status_label.setText("Position: ✓ Set (drag to adjust)")
                self.position_status_label.setStyleSheet("QLabel { padding: 5px; background-color: #c8e6c9; color: #2e7d32; }")
            else:
                self.position_status_label.setText("Position: Auto-detected")
                self.position_status_label.setStyleSheet("QLabel { padding: 5px; background-color: #fff9c4; color: #f57f17; }")
        else:
            self.position_status_label.setText("Position: Not set (will auto-detect)")
            self.position_status_label.setStyleSheet("QLabel { padding: 5px; background-color: #ffcdd2; color: #c62828; }")
        
        # Update plots if we have angle/curvature data
        if len(self.body_angles) > 0 or len(self.midline_points) > 0:
            self.update_plots()
        
    def detect_endpoints(self, roi):
        """Detect head and tail positions based on selected method"""
        method = self.detection_method.currentText()
        
        if "Top" in method:
            # Head is at the top (minimum y)
            idx_head = np.argmin(roi[:, 1])
            idx_tail = np.argmax(roi[:, 1])
        elif "Bottom" in method:
            # Head is at the bottom (maximum y)
            idx_head = np.argmax(roi[:, 1])
            idx_tail = np.argmin(roi[:, 1])
        elif "Left" in method:
            # Head is at the left (minimum x)
            idx_head = np.argmin(roi[:, 0])
            idx_tail = np.argmax(roi[:, 0])
        elif "Right" in method:
            # Head is at the right (maximum x)
            idx_head = np.argmax(roi[:, 0])
            idx_tail = np.argmin(roi[:, 0])
        else:
            # Manual - use existing or centroid
            if self.current_frame in self.head_positions:
                return self.head_positions[self.current_frame], self.tail_positions[self.current_frame]
            else:
                # Default to top/bottom
                idx_head = np.argmin(roi[:, 1])
                idx_tail = np.argmax(roi[:, 1])
        
        head = roi[idx_head]
        tail = roi[idx_tail]
        
        return head, tail
    
    def calculate_body_angle(self, head, tail):
        """
        Calculate angle of body axis relative to horizontal
        
        Method: Uses arctangent2 (atan2) to calculate angle
        - Measures from horizontal right (0°) counterclockwise
        - Positive angles: counterclockwise from horizontal
        - Negative angles: clockwise from horizontal
        
        Calculation:
        dx = head_x - tail_x  (horizontal displacement)
        dy = head_y - tail_y  (vertical displacement)  
        angle = atan2(dy, dx) * 180/π
        
        Examples:
        - Horizontal right: 0°
        - Up: +90°
        - Horizontal left: ±180°
        - Down: -90°
        
        Note: Y-axis increases downward in images, so positive dy means
        head is below tail, giving negative angle
        """
        dx = head[0] - tail[0]
        dy = head[1] - tail[1]
        angle = np.degrees(np.arctan2(dy, dx))
        return angle
    
    def calculate_midline(self, roi, head, tail):
        """Calculate midline - choose method based on user selection"""
        method = self.midline_method.currentText()
        
        if method == "Simple interpolation":
            return self.calculate_simple_midline(roi, head, tail)
        else:
            # Try skeleton method, fall back to simple if it fails
            midline = self.calculate_skeleton_midline(roi, head, tail)
            if midline is None:
                print("Skeleton method failed, using simple interpolation")
                return self.calculate_simple_midline(roi, head, tail)
            return midline
    
    def calculate_simple_midline(self, roi, head, tail):
        """Simple midline by following ROI shape between head and tail"""
        try:
            # Create points along head-tail axis
            n_points = 30
            
            # Calculate axis
            axis_vector = tail - head
            axis_length = np.linalg.norm(axis_vector)
            if axis_length < 1:
                return None
            
            axis_unit = axis_vector / axis_length
            
            # For each point along axis, find the centroid of nearby ROI points
            midline_points = []
            
            # FORCE first point to be HEAD
            midline_points.append(head.copy())
            
            # Calculate intermediate points
            for i in range(1, n_points - 1):
                t = i / (n_points - 1)
                # Point on the axis
                axis_point = head + t * axis_vector
                
                # Find ROI points perpendicular to this axis point
                # Project all ROI points onto the axis
                projections = np.dot(roi - head, axis_unit)
                target_projection = t * axis_length
                
                # Find points near this projection
                dist_along_axis = np.abs(projections - target_projection)
                window = axis_length / (n_points - 1) * 1.5  # Overlap windows
                
                nearby_mask = dist_along_axis < window
                nearby_points = roi[nearby_mask]
                
                if len(nearby_points) > 0:
                    # Use centroid of nearby points
                    midline_point = np.mean(nearby_points, axis=0)
                    midline_points.append(midline_point)
                else:
                    # Fallback to axis point
                    midline_points.append(axis_point)
            
            # FORCE last point to be TAIL
            midline_points.append(tail.copy())
            
            if len(midline_points) < 3:
                return None
            
            midline_points = np.array(midline_points)
            
            # Smooth with spline, but ensure endpoints are preserved
            try:
                # Use internal points for spline, then add endpoints
                if len(midline_points) > 4:
                    internal_points = midline_points[1:-1]  # Exclude endpoints
                    tck, u = splprep([internal_points[:, 0], internal_points[:, 1]], s=30, k=min(3, len(internal_points)-1))
                    u_new = np.linspace(0, 1, 98)  # 98 internal points for better resolution
                    x_new, y_new = splev(u_new, tck)
                    smooth_internal = np.column_stack([x_new, y_new])
                    
                    # Combine: head + smoothed internal + tail
                    result = np.vstack([head.reshape(1, 2), smooth_internal, tail.reshape(1, 2)])
                    return result
                else:
                    return midline_points
            except:
                return midline_points
            
        except Exception as e:
            print(f"Error in simple midline calculation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_skeleton_midline(self, roi, head, tail):
        """Calculate midline of the larva body using skeleton"""
        try:
            # Get bounding box
            x_min, y_min = roi.min(axis=0).astype(int)
            x_max, y_max = roi.max(axis=0).astype(int)
            
            # Add padding
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(self.current_display_frame.shape[1], x_max + padding)
            y_max = min(self.current_display_frame.shape[0], y_max + padding)
            
            # Create binary mask of the larva
            mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
            roi_shifted = roi - np.array([x_min, y_min])
            cv2.fillPoly(mask, [roi_shifted.astype(np.int32)], 255)
            
            # Apply aggressive morphological closing to connect gaps
            kernel_large = np.ones((5, 5), np.uint8)
            kernel_small = np.ones((3, 3), np.uint8)
            
            # First, dilate to close gaps
            mask = cv2.dilate(mask, kernel_large, iterations=3)
            mask = cv2.erode(mask, kernel_large, iterations=3)
            
            # Then clean up
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
            
            # Fill any remaining holes
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask, contours, -1, 255, -1)
            
            # Find connected components and keep only the largest
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            if num_labels < 2:  # Only background
                return None
            
            # Get largest component (excluding background which is label 0)
            largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == largest_component).astype(np.uint8) * 255
            
            # Skeletonize to get medial axis
            from skimage.morphology import skeletonize
            skeleton = skeletonize(mask > 0)
            
            # Get skeleton points
            skel_points = np.column_stack(np.where(skeleton))
            
            if len(skel_points) < 5:
                print(f"Too few skeleton points: {len(skel_points)}")
                return None
            
            # Convert back to original coordinates (swap x,y because np.where returns row,col)
            skel_points = np.column_stack([skel_points[:, 1] + x_min, skel_points[:, 0] + y_min])
            
            # Order skeleton points from head to tail using better path finding
            # Find point closest to head
            distances_to_head = np.linalg.norm(skel_points - head, axis=1)
            start_idx = np.argmin(distances_to_head)
            
            # Find point closest to tail for validation
            distances_to_tail = np.linalg.norm(skel_points - tail, axis=1)
            end_idx = np.argmin(distances_to_tail)
            
            # Build path from head to tail
            ordered_points = [skel_points[start_idx]]
            remaining = set(range(len(skel_points)))
            remaining.remove(start_idx)
            
            # Greedy path finding with larger distance threshold
            max_step_distance = 15  # Increased to bridge small gaps
            current_idx = start_idx
            
            while remaining and len(ordered_points) < len(skel_points):
                current = skel_points[current_idx]
                
                # Find nearest remaining point
                distances = np.array([np.linalg.norm(skel_points[i] - current) for i in remaining])
                
                if len(distances) == 0:
                    break
                
                min_dist_idx = np.argmin(distances)
                min_dist = distances[min_dist_idx]
                
                # If too far, try to bridge the gap
                if min_dist > max_step_distance:
                    # Check if we can reach tail
                    if end_idx in remaining:
                        nearest_idx = end_idx
                    else:
                        break
                else:
                    nearest_idx = list(remaining)[min_dist_idx]
                
                ordered_points.append(skel_points[nearest_idx])
                remaining.remove(nearest_idx)
                current_idx = nearest_idx
                
                # Stop if we reached near the tail
                if nearest_idx == end_idx:
                    break
            
            ordered_points = np.array(ordered_points)
            
            if len(ordered_points) < 5:
                print(f"Path too short: {len(ordered_points)} points")
                return None
            
            # Smooth the ordered skeleton with spline
            if len(ordered_points) > 10:
                # Sample fewer points for smoother curve
                step = max(1, len(ordered_points) // 30)
                sampled = ordered_points[::step]
                
                if len(sampled) < 4:
                    # FORCE endpoints to head/tail
                    result = np.vstack([head.reshape(1, 2), sampled, tail.reshape(1, 2)])
                    return result
                
                try:
                    # Use higher smoothing factor, but preserve endpoints
                    # Use internal points for spline, then add endpoints
                    tck, u = splprep([sampled[:, 0], sampled[:, 1]], s=100, k=min(3, len(sampled)-1))
                    u_new = np.linspace(0, 1, 98)  # 98 internal points for better resolution
                    x_new, y_new = splev(u_new, tck)
                    smooth_internal = np.column_stack([x_new, y_new])
                    
                    # FORCE: head + smoothed path + tail
                    midline = np.vstack([head.reshape(1, 2), smooth_internal, tail.reshape(1, 2)])
                    return midline
                except Exception as e:
                    print(f"Spline smoothing failed: {e}")
                    # If spline fails, return with forced endpoints
                    result = np.vstack([head.reshape(1, 2), sampled, tail.reshape(1, 2)])
                    return result
            else:
                # FORCE endpoints for short paths
                result = np.vstack([head.reshape(1, 2), ordered_points, tail.reshape(1, 2)])
                return result
            
        except Exception as e:
            print(f"Error in skeleton midline calculation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def snap_to_roi(self, point):
        """Snap a point to the nearest point on the current ROI contour"""
        roi_index = self.current_frame + self.roi_offset_spinbox.value()
        
        if not (0 <= roi_index < len(self.rois)) or self.rois[roi_index] is None:
            return point  # No ROI, return original point
        
        roi = self.rois[roi_index]
        
        # Find nearest point on ROI
        distances = np.linalg.norm(roi - point, axis=1)
        nearest_idx = np.argmin(distances)
        snapped_point = roi[nearest_idx]
        
        return snapped_point
    
    def snap_to_roi(self, point):
        """Snap a point to the nearest point on the ROI contour"""
        roi_index = self.current_frame + self.roi_offset_spinbox.value()
        
        if not (0 <= roi_index < len(self.rois)) or self.rois[roi_index] is None:
            return point  # No ROI, return original point
        
        roi = self.rois[roi_index]
        
        # Find nearest point on ROI
        distances = np.linalg.norm(roi - point, axis=1)
        nearest_idx = np.argmin(distances)
        snapped_point = roi[nearest_idx]
        
        return snapped_point
    
    def snap_segment_to_roi(self, point, segment_label):
        """Snap a segment point to ROI or midline, constrained between its neighbors"""
        roi_index = self.current_frame + self.roi_offset_spinbox.value()
        
        if not (0 <= roi_index < len(self.rois)) or self.rois[roi_index] is None:
            return point
        
        if self.current_frame not in self.segment_points:
            return self.snap_to_roi(point)  # Fall back to regular snapping
        
        roi = self.rois[roi_index]
        segments = self.segment_points[self.current_frame]
        
        # Determine which side this segment is on and find neighbors
        is_midline = False
        if segment_label in self.segment_labels['left']:
            labels_list = self.segment_labels['left']
        elif segment_label in self.segment_labels['right']:
            labels_list = self.segment_labels['right']
        elif segment_label in self.segment_labels['midline']:
            labels_list = self.segment_labels['midline']
            is_midline = True
        else:
            return self.snap_to_roi(point)  # Unknown label, use regular snapping
        
        # MIDLINE POINTS: Constrain to calculated midline, not ROI
        if is_midline:
            # Get current head and tail to calculate midline
            if self.current_frame not in self.head_positions or self.current_frame not in self.tail_positions:
                return point
            
            head = self.head_positions[self.current_frame]
            tail = self.tail_positions[self.current_frame]
            
            # Calculate midline based on current head/tail
            midline = self.calculate_midline(roi, head, tail)
            if midline is None or len(midline) < 2:
                return point
            
            # Find neighbors on midline
            try:
                idx = labels_list.index(segment_label)
            except ValueError:
                return point
            
            prev_pos = None
            next_pos = None
            
            # For first point (st1), use head as previous neighbor
            if idx == 0:
                prev_pos = head
            elif idx > 0 and labels_list[idx-1] in segments:
                prev_pos = segments[labels_list[idx-1]]
            
            # For last point (sa7), use tail as next neighbor
            if idx == len(labels_list) - 1:
                next_pos = tail
            elif idx < len(labels_list) - 1 and labels_list[idx+1] in segments:
                next_pos = segments[labels_list[idx+1]]
            
            # Constrain to midline segment between neighbors
            if prev_pos is not None and next_pos is not None:
                # Find nearest midline indices for neighbors
                prev_distances = np.linalg.norm(midline - prev_pos, axis=1)
                next_distances = np.linalg.norm(midline - next_pos, axis=1)
                
                prev_idx = np.argmin(prev_distances)
                next_idx = np.argmin(next_distances)
                
                # Get midline segment between neighbors (no wrapping for midline)
                start_idx = min(prev_idx, next_idx)
                end_idx = max(prev_idx, next_idx)
                
                valid_midline_points = midline[start_idx:end_idx+1]
                
                if len(valid_midline_points) > 0:
                    # IMPROVED: Ensure spatial density
                    if len(valid_midline_points) > 1:
                        # Check spacing between consecutive points
                        dists = np.linalg.norm(np.diff(valid_midline_points, axis=0), axis=1)
                        avg_spacing = np.mean(dists)
                        max_spacing = np.max(dists)
                        
                        # If spacing is too large, interpolate
                        desired_spacing = 3.0  # pixels
                        
                        if avg_spacing > desired_spacing or max_spacing > desired_spacing * 2:
                            # Interpolate to create denser point set
                            interpolated_points = [valid_midline_points[0]]
                            
                            for i in range(len(valid_midline_points) - 1):
                                start = valid_midline_points[i]
                                end = valid_midline_points[i + 1]
                                dist = np.linalg.norm(end - start)
                                
                                if dist > desired_spacing:
                                    # Add intermediate points
                                    n_intermediate = int(np.ceil(dist / desired_spacing))
                                    for j in range(1, n_intermediate):
                                        t = j / n_intermediate
                                        interp_point = start + t * (end - start)
                                        interpolated_points.append(interp_point)
                                
                                interpolated_points.append(end)
                            
                            valid_midline_points = np.array(interpolated_points)
                    
                    distances_to_point = np.linalg.norm(valid_midline_points - point, axis=1)
                    nearest_idx = np.argmin(distances_to_point)
                    return valid_midline_points[nearest_idx]
            
            # If no neighbors, snap to nearest point on midline
            distances = np.linalg.norm(midline - point, axis=1)
            nearest_idx = np.argmin(distances)
            return midline[nearest_idx]
        
        # LEFT/RIGHT POINTS: Constrain to ROI as before
        # Find index in the list
        try:
            idx = labels_list.index(segment_label)
        except ValueError:
            return self.snap_to_roi(point)
        
        # Get neighbor positions
        prev_pos = None
        next_pos = None
        
        # Get current head and tail positions (needed for first/last segments)
        if self.current_frame in self.head_positions and self.current_frame in self.tail_positions:
            head = self.head_positions[self.current_frame]
            tail = self.tail_positions[self.current_frame]
        else:
            head = None
            tail = None
        
        # For first point (t1l or t1r), use head as previous neighbor
        if idx == 0 and head is not None:
            prev_pos = head
        elif idx > 0 and labels_list[idx-1] in segments:
            prev_pos = segments[labels_list[idx-1]]
        
        # For last point (a7l or a7r), use tail as next neighbor
        if idx == len(labels_list) - 1 and tail is not None:
            next_pos = tail
        elif idx < len(labels_list) - 1 and labels_list[idx+1] in segments:
            next_pos = segments[labels_list[idx+1]]
        
        # If we have both neighbors, constrain to ROI segment between them
        if prev_pos is not None and next_pos is not None:
            # Find nearest ROI indices for both neighbors
            prev_distances = np.linalg.norm(roi - prev_pos, axis=1)
            next_distances = np.linalg.norm(roi - next_pos, axis=1)
            
            prev_idx = np.argmin(prev_distances)
            next_idx = np.argmin(next_distances)
            
            # Calculate both possible paths around the contour
            # Path 1: prev_idx -> next_idx (forward)
            if prev_idx <= next_idx:
                path1_indices = list(range(prev_idx, next_idx + 1))
            else:
                path1_indices = list(range(prev_idx, len(roi))) + list(range(0, next_idx + 1))
            
            # Path 2: prev_idx -> next_idx (backward)  
            if prev_idx >= next_idx:
                path2_indices = list(range(next_idx, prev_idx + 1))
            else:
                path2_indices = list(range(next_idx, len(roi))) + list(range(0, prev_idx + 1))
            
            # Choose the shorter path
            if len(path1_indices) <= len(path2_indices):
                valid_indices = path1_indices
            else:
                valid_indices = path2_indices
            
            # Get the actual ROI points along this path
            if len(valid_indices) > 0:
                valid_roi_points = roi[valid_indices]
                
                # IMPROVED: Ensure spatial density, not just index count
                # Check if points are spatially sparse
                if len(valid_roi_points) > 1:
                    # Calculate distances between consecutive points
                    dists = np.linalg.norm(np.diff(valid_roi_points, axis=0), axis=1)
                    avg_spacing = np.mean(dists)
                    max_spacing = np.max(dists)
                    
                    # If spacing is too large, interpolate intermediate points
                    desired_spacing = 3.0  # pixels - want points every ~3 pixels
                    
                    if avg_spacing > desired_spacing or max_spacing > desired_spacing * 2:
                        # Interpolate to create denser point set
                        interpolated_points = [valid_roi_points[0]]
                        
                        for i in range(len(valid_roi_points) - 1):
                            start = valid_roi_points[i]
                            end = valid_roi_points[i + 1]
                            dist = np.linalg.norm(end - start)
                            
                            if dist > desired_spacing:
                                # Add intermediate points
                                n_intermediate = int(np.ceil(dist / desired_spacing))
                                for j in range(1, n_intermediate):
                                    t = j / n_intermediate
                                    interp_point = start + t * (end - start)
                                    interpolated_points.append(interp_point)
                            
                            interpolated_points.append(end)
                        
                        valid_roi_points = np.array(interpolated_points)
                
                # Find nearest point to drag position
                distances_to_point = np.linalg.norm(valid_roi_points - point, axis=1)
                nearest_in_valid = np.argmin(distances_to_point)
                return valid_roi_points[nearest_in_valid]
        
        # If no neighbors or constraint failed, use regular snapping
        return self.snap_to_roi(point)
    
    def swap_head_tail(self):
        """Swap head and tail positions"""
        if self.current_frame in self.head_positions and self.current_frame in self.tail_positions:
            temp = self.head_positions[self.current_frame].copy()
            self.head_positions[self.current_frame] = self.tail_positions[self.current_frame]
            self.tail_positions[self.current_frame] = temp
            self.update_display()
    
    def clear_current_frame(self):
        """Remove manual head/tail positions for current frame"""
        if self.current_frame in self.head_positions:
            del self.head_positions[self.current_frame]
        if self.current_frame in self.tail_positions:
            del self.tail_positions[self.current_frame]
        self.info_label.setText("✓ Cleared. Will auto-detect on next update.")
        self.update_display()
    
    def redetect_current(self):
        """Force re-detection for current frame"""
        roi_index = self.current_frame + self.roi_offset_spinbox.value()
        if 0 <= roi_index < len(self.rois) and self.rois[roi_index] is not None:
            head, tail = self.detect_endpoints(self.rois[roi_index])
            self.head_positions[self.current_frame] = head
            self.tail_positions[self.current_frame] = tail
            self.info_label.setText("✓ Re-detected head/tail")
            self.update_display()
        else:
            QMessageBox.warning(self, "Warning", "No ROI available for current frame!")
    
    def apply_endpoints_to_all(self):
        """Apply current endpoint detection method to all frames WITHOUT manual positions"""
        if len(self.rois) == 0:
            QMessageBox.warning(self, "Warning", "No ROIs loaded!")
            return
        
        manually_set = set()
        auto_detected = 0
        
        for i in range(len(self.rois)):
            if self.rois[i] is not None:
                # Only auto-detect if BOTH head AND tail are missing
                # This preserves any manual adjustments
                if i not in self.head_positions or i not in self.tail_positions:
                    head, tail = self.detect_endpoints(self.rois[i])
                    self.head_positions[i] = head
                    self.tail_positions[i] = tail
                    auto_detected += 1
                else:
                    manually_set.add(i)
        
        message = f"Auto-detected: {auto_detected} frames\n"
        if len(manually_set) > 0:
            message += f"Preserved manual adjustments: {len(manually_set)} frames"
        
        QMessageBox.information(self, "Success", message)
        self.update_display()
    
    def force_redetect_all(self):
        """Force re-detection on ALL frames, overwriting manual adjustments"""
        if len(self.rois) == 0:
            QMessageBox.warning(self, "Warning", "No ROIs loaded!")
            return
        
        # Count how many manual adjustments will be lost
        manual_count = len([i for i in range(len(self.rois)) 
                           if i in self.head_positions and i in self.tail_positions])
        
        # Confirm with user
        reply = QMessageBox.question(
            self, 
            "Confirm Force Re-detect",
            f"This will OVERWRITE all manual adjustments!\n\n"
            f"Frames with positions: {manual_count}\n\n"
            f"Are you sure you want to re-detect ALL frames?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
        
        # Clear all positions and re-detect
        self.head_positions = {}
        self.tail_positions = {}
        
        for i in range(len(self.rois)):
            if self.rois[i] is not None:
                head, tail = self.detect_endpoints(self.rois[i])
                self.head_positions[i] = head
                self.tail_positions[i] = tail
        
        QMessageBox.information(self, "Success", 
                               f"Re-detected all {len(self.rois)} frames\n"
                               f"All manual adjustments have been overwritten")
        self.update_display()
    
    def draw_segment_points(self, frame, frame_idx):
        """Draw segment points and connecting lines"""
        if frame_idx not in self.segment_points:
            return
        
        segments = self.segment_points[frame_idx]
        
        # Colors for different sides
        left_color = (0, 255, 255)  # Cyan for left
        right_color = (255, 165, 0)  # Orange for right  
        midline_color = (255, 255, 0)  # Yellow for midline
        line_color = (200, 200, 200)  # Gray for connecting lines
        
        # Draw connecting lines first (so points appear on top)
        for i in range(10):
            left_label = self.segment_labels['left'][i]
            right_label = self.segment_labels['right'][i]
            mid_label = self.segment_labels['midline'][i]
            
            if left_label in segments and right_label in segments and mid_label in segments:
                left_pt = segments[left_label].astype(int)
                right_pt = segments[right_label].astype(int)
                mid_pt = segments[mid_label].astype(int)
                
                # Draw line across: left -> midline -> right
                cv2.line(frame, tuple(left_pt), tuple(mid_pt), line_color, 1)
                cv2.line(frame, tuple(mid_pt), tuple(right_pt), line_color, 1)
        
        # Draw points
        for label, point in segments.items():
            pt = tuple(point.astype(int))
            
            # Determine color based on label
            if label.endswith('l'):
                color = left_color
            elif label.endswith('r'):
                color = right_color
            else:  # midline (st or sa)
                color = midline_color
            
            # Draw circle
            radius = 4
            if self.dragging_segment == label:
                radius = 6  # Larger when dragging
                cv2.circle(frame, pt, radius + 2, color, 2)  # Outer ring
            
            cv2.circle(frame, pt, radius, color, -1)
            
            # Draw label
            label_pos = (pt[0] + 8, pt[1] - 8)
            cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                       0.3, color, 1, cv2.LINE_AA)
    
    def on_show_segments_changed(self):
        """Handle show segments checkbox change"""
        self.show_segments = self.show_segments_cb.isChecked()
        self.update_display()
    
    def initialize_segments_current(self):
        """Initialize segment points for current frame"""
        if self.current_frame not in self.head_positions or self.current_frame not in self.tail_positions:
            QMessageBox.warning(self, "Warning", "Set head and tail first!")
            return
        
        roi_index = self.current_frame + self.roi_offset_spinbox.value()
        if not (0 <= roi_index < len(self.rois)) or self.rois[roi_index] is None:
            QMessageBox.warning(self, "Warning", "No ROI available for current frame!")
            return
        
        roi = self.rois[roi_index]
        head = self.head_positions[self.current_frame]
        tail = self.tail_positions[self.current_frame]
        
        # Initialize segment points
        segments = self.calculate_segment_points(roi, head, tail)
        
        if segments:
            self.segment_points[self.current_frame] = segments
            self.show_segments = True
            self.show_segments_cb.setChecked(True)
            self.update_display()
            
            # Update segment distance plots
            self.update_segment_distance_plots()
            
            total_points = sum(len(v) for v in segments.values())
            self.info_label.setText(f"✓ Initialized {total_points} segment points")
        else:
            QMessageBox.warning(self, "Warning", "Failed to initialize segments!")
    
    def calculate_segment_points(self, roi, head, tail):
        """Calculate evenly-spaced segment points along left, right, and midline"""
        try:
            # Calculate the actual midline first
            midline = self.calculate_midline(roi, head, tail)
            if midline is None or len(midline) < 10:
                print("Failed to calculate midline for segments")
                return None
            
            # Find where head and tail map to on the ROI contour
            head_distances = np.linalg.norm(roi - head, axis=1)
            tail_distances = np.linalg.norm(roi - tail, axis=1)
            head_idx = np.argmin(head_distances)
            tail_idx = np.argmin(tail_distances)
            
            # Determine which path along ROI is the body (shorter path)
            # Calculate both paths around the contour
            if head_idx <= tail_idx:
                path1 = list(range(head_idx, tail_idx + 1))
            else:
                path1 = list(range(head_idx, len(roi))) + list(range(0, tail_idx + 1))
            
            if head_idx >= tail_idx:
                path2 = list(range(tail_idx, head_idx + 1))
            else:
                path2 = list(range(tail_idx, len(roi))) + list(range(0, head_idx + 1))
            
            # Use the shorter path as the body contour
            if len(path1) <= len(path2):
                left_path = path1
                right_path = list(reversed(path2))  # Other side goes backwards
            else:
                left_path = list(reversed(path2))
                right_path = path1
            
            # Get actual ROI points for each side
            left_roi_points = roi[left_path]
            right_roi_points = roi[right_path]
            
            # Sample 10 evenly-spaced points along each side
            # IMPORTANT: Don't sample at exact endpoints (head/tail)
            # Offset inward slightly so segments are distinct from head/tail landmarks
            n_segments = 10
            segments = {}
            left_labels = self.segment_labels['left']
            right_labels = self.segment_labels['right']
            midline_labels = self.segment_labels['midline']
            
            for i in range(n_segments):
                # Position along body with offset from endpoints
                # Instead of 0.0, 0.111, 0.222, ..., 0.888, 1.0
                # Use:    0.05, 0.15, 0.25, ..., 0.85, 0.95
                # This gives 10 points evenly spaced between head and tail
                # but offset 5% inward from each endpoint
                t = 0.05 + (i / (n_segments - 1)) * 0.9  # Maps to 0.05 -> 0.95
                
                # Sample from left side
                left_idx = int(t * (len(left_roi_points) - 1))
                left_idx = min(left_idx, len(left_roi_points) - 1)
                left_point = left_roi_points[left_idx]
                
                # Sample from right side
                right_idx = int(t * (len(right_roi_points) - 1))
                right_idx = min(right_idx, len(right_roi_points) - 1)
                right_point = right_roi_points[right_idx]
                
                # Sample from midline
                midline_idx = int(t * (len(midline) - 1))
                midline_idx = min(midline_idx, len(midline) - 1)
                midline_point = midline[midline_idx]
                
                # Store with labels
                segments[left_labels[i]] = left_point
                segments[right_labels[i]] = right_point
                segments[midline_labels[i]] = midline_point
            
            return segments
            
        except Exception as e:
            print(f"Error calculating segment points: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def find_roi_point_along_line(self, start, end, roi):
        """Find where a line from start to end intersects the ROI"""
        try:
            # Sample points along the line
            n_samples = 50
            line_points = np.array([start + t * (end - start) for t in np.linspace(0, 1, n_samples)])
            
            # Find closest ROI point to each line point
            min_dist = float('inf')
            best_roi_point = None
            
            for line_point in line_points:
                distances = np.linalg.norm(roi - line_point, axis=1)
                min_idx = np.argmin(distances)
                if distances[min_idx] < min_dist:
                    min_dist = distances[min_idx]
                    best_roi_point = roi[min_idx]
                    
                # If we found a point very close to ROI, use it
                if min_dist < 5:
                    break
            
            return best_roi_point
        except:
            return None
    
    def calculate_segment_distances(self, frame_idx):
        """Calculate distances between adjacent segment points using arc length along contours"""
        if frame_idx not in self.segment_points:
            return {}
        
        segments = self.segment_points[frame_idx]
        distances = {}
        
        # Get ROI and midline for arc length calculations
        roi_index = frame_idx + self.roi_offset_spinbox.value()
        roi = None
        midline = None
        
        if 0 <= roi_index < len(self.rois) and self.rois[roi_index] is not None:
            roi = self.rois[roi_index]
            
            # Get midline if head/tail exist
            if frame_idx in self.head_positions and frame_idx in self.tail_positions:
                head = self.head_positions[frame_idx]
                tail = self.tail_positions[frame_idx]
                midline = self.calculate_midline(roi, head, tail)
        
        # Left side distances - arc length along ROI
        for i in range(len(self.segment_labels['left']) - 1):
            curr_label = self.segment_labels['left'][i]
            next_label = self.segment_labels['left'][i+1]
            
            if curr_label in segments and next_label in segments:
                if roi is not None:
                    # Calculate arc length along ROI contour
                    arc_len = self.calculate_arc_length_on_contour(
                        segments[curr_label], segments[next_label], roi
                    )
                    distances[f"{curr_label}_{next_label}_dist"] = arc_len
                else:
                    # Fallback to straight-line if no ROI
                    dist = np.linalg.norm(segments[next_label] - segments[curr_label])
                    distances[f"{curr_label}_{next_label}_dist"] = dist
        
        # Right side distances - arc length along ROI
        for i in range(len(self.segment_labels['right']) - 1):
            curr_label = self.segment_labels['right'][i]
            next_label = self.segment_labels['right'][i+1]
            
            if curr_label in segments and next_label in segments:
                if roi is not None:
                    # Calculate arc length along ROI contour
                    arc_len = self.calculate_arc_length_on_contour(
                        segments[curr_label], segments[next_label], roi
                    )
                    distances[f"{curr_label}_{next_label}_dist"] = arc_len
                else:
                    # Fallback to straight-line if no ROI
                    dist = np.linalg.norm(segments[next_label] - segments[curr_label])
                    distances[f"{curr_label}_{next_label}_dist"] = dist
        
        # Midline distances - arc length along midline
        for i in range(len(self.segment_labels['midline']) - 1):
            curr_label = self.segment_labels['midline'][i]
            next_label = self.segment_labels['midline'][i+1]
            
            if curr_label in segments and next_label in segments:
                if midline is not None and len(midline) > 0:
                    # Calculate arc length along midline
                    arc_len = self.calculate_arc_length_on_path(
                        segments[curr_label], segments[next_label], midline
                    )
                    distances[f"{curr_label}_{next_label}_dist"] = arc_len
                else:
                    # Fallback to straight-line if no midline
                    dist = np.linalg.norm(segments[next_label] - segments[curr_label])
                    distances[f"{curr_label}_{next_label}_dist"] = dist
        
        # Cross-sectional widths (e.g., t1l-t1r, a1l-a1r)
        # Keep as straight-line - this measures body width, not contour length
        for i in range(len(self.segment_labels['left'])):
            left_label = self.segment_labels['left'][i]
            right_label = self.segment_labels['right'][i]
            
            if left_label in segments and right_label in segments:
                dist = np.linalg.norm(segments[right_label] - segments[left_label])
                # Extract segment name (t1, t2, a1, etc.)
                seg_name = left_label[:-1]  # Remove 'l' suffix
                distances[f"{seg_name}_width"] = dist
        
        return distances
    
    def calculate_arc_length_on_contour(self, point1, point2, roi):
        """Calculate arc length along ROI contour between two points"""
        try:
            # Find nearest ROI indices for both points
            distances1 = np.linalg.norm(roi - point1, axis=1)
            distances2 = np.linalg.norm(roi - point2, axis=1)
            
            idx1 = np.argmin(distances1)
            idx2 = np.argmin(distances2)
            
            # Calculate both possible paths around the contour
            if idx1 <= idx2:
                path1_indices = list(range(idx1, idx2 + 1))
            else:
                path1_indices = list(range(idx1, len(roi))) + list(range(0, idx2 + 1))
            
            if idx1 >= idx2:
                path2_indices = list(range(idx2, idx1 + 1))
            else:
                path2_indices = list(range(idx2, len(roi))) + list(range(0, idx1 + 1))
            
            # Use the shorter path
            if len(path1_indices) <= len(path2_indices):
                path_indices = path1_indices
            else:
                path_indices = path2_indices
            
            # Get the actual points along this path
            if len(path_indices) < 2:
                # Very close points, return straight-line distance
                return np.linalg.norm(point2 - point1)
            
            path_points = roi[path_indices]
            
            # Calculate arc length by summing distances between consecutive points
            arc_length = 0.0
            for i in range(len(path_points) - 1):
                arc_length += np.linalg.norm(path_points[i+1] - path_points[i])
            
            return arc_length
            
        except Exception as e:
            print(f"Error calculating arc length on contour: {e}")
            # Fallback to straight-line distance
            return np.linalg.norm(point2 - point1)
    
    def calculate_arc_length_on_path(self, point1, point2, path):
        """Calculate arc length along a path (e.g., midline) between two points"""
        try:
            # Find nearest path indices for both points
            distances1 = np.linalg.norm(path - point1, axis=1)
            distances2 = np.linalg.norm(path - point2, axis=1)
            
            idx1 = np.argmin(distances1)
            idx2 = np.argmin(distances2)
            
            # Get the segment of path between the two indices (no wrapping for paths)
            start_idx = min(idx1, idx2)
            end_idx = max(idx1, idx2)
            
            if start_idx == end_idx:
                # Same point, return straight-line distance
                return np.linalg.norm(point2 - point1)
            
            path_segment = path[start_idx:end_idx+1]
            
            # Calculate arc length by summing distances between consecutive points
            arc_length = 0.0
            for i in range(len(path_segment) - 1):
                arc_length += np.linalg.norm(path_segment[i+1] - path_segment[i])
            
            return arc_length
            
        except Exception as e:
            print(f"Error calculating arc length on path: {e}")
            # Fallback to straight-line distance
            return np.linalg.norm(point2 - point1)
    
    def apply_segments_to_all(self):
        """Apply segment initialization to all frames with head/tail"""
        if len(self.head_positions) == 0:
            QMessageBox.warning(self, "Warning", "No head/tail positions set!")
            return
        
        reply = QMessageBox.question(
            self,
            "Initialize Segments",
            f"Initialize segment points for {len(self.head_positions)} frames?\n\n"
            f"This will create T1-T3, A1-A7 points on each frame.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.No:
            return
        
        success_count = 0
        for frame in self.head_positions.keys():
            if frame in self.tail_positions:
                roi_index = frame + self.roi_offset_spinbox.value()
                if 0 <= roi_index < len(self.rois) and self.rois[roi_index] is not None:
                    roi = self.rois[roi_index]
                    head = self.head_positions[frame]
                    tail = self.tail_positions[frame]
                    
                    segments = self.calculate_segment_points(roi, head, tail)
                    if segments:
                        self.segment_points[frame] = segments
                        success_count += 1
        
        self.show_segments = True
        self.show_segments_cb.setChecked(True)
        self.update_display()
        
        # Update segment distance plots
        self.update_segment_distance_plots()
        
        QMessageBox.information(
            self,
            "Success",
            f"Initialized segments for {success_count} frames"
        )
    
    def analyze_all_frames(self):
        """Analyze all frames and calculate angles"""
        if len(self.rois) == 0:
            QMessageBox.warning(self, "Warning", "No ROIs loaded!")
            return
        
        # First ensure all endpoints are detected (preserves manual adjustments)
        manually_set_before = len(self.head_positions)
        self.apply_endpoints_to_all()
        manually_set_after = manually_set_before  # apply_endpoints_to_all preserves manual ones
        
        # Calculate body angles
        analyzed_count = 0
        for i in range(len(self.rois)):
            if i in self.head_positions and i in self.tail_positions:
                head = self.head_positions[i]
                tail = self.tail_positions[i]
                self.body_angles[i] = self.calculate_body_angle(head, tail)
                analyzed_count += 1
                
                # Calculate midline if needed
                if self.rois[i] is not None:
                    midline = self.calculate_midline(self.rois[i], head, tail)
                    if midline is not None:
                        self.midline_points[i] = midline
        
        # Calculate turning angles (frame-to-frame changes)
        angles = [self.body_angles[i] for i in sorted(self.body_angles.keys())]
        for i in range(1, len(angles)):
            angle_diff = angles[i] - angles[i-1]
            # Normalize to [-180, 180]
            while angle_diff > 180:
                angle_diff -= 360
            while angle_diff < -180:
                angle_diff += 360
            self.turning_angles[i] = angle_diff
        
        self.analysis_status.setText(f"✓ Analyzed {analyzed_count} frames (preserved manual adjustments)")
        self.update_plots()
        
    def update_plots(self):
        """Update all analysis plots"""
        if len(self.body_angles) == 0:
            return
        
        # Clear previous plots
        self.ax_angle.clear()
        self.ax_turning.clear()
        self.ax_curvature.clear()
        
        # Body angle over time
        frames = sorted(self.body_angles.keys())
        angles = [self.body_angles[f] for f in frames]
        
        # Check if single frame mode
        is_single_frame = len(frames) == 1
        
        if is_single_frame:
            # For single frame, show as a large marker with value
            self.ax_angle.plot(frames, angles, 'bo', markersize=15, label=f'Angle: {angles[0]:.1f}°')
            self.ax_angle.axhline(y=angles[0], color='b', linestyle='--', alpha=0.3)
            # Add text annotation
            self.ax_angle.text(frames[0], angles[0], f'  {angles[0]:.1f}°', 
                             fontsize=12, va='center', ha='left', color='blue', fontweight='bold')
            self.ax_angle.set_xlim(-0.5, 0.5)  # Center the point
            self.ax_angle.set_title("Body Angle (Single Frame)")
        else:
            # Multiple frames, show as line
            self.ax_angle.plot(frames, angles, 'b-', linewidth=2, marker='o')
            self.ax_angle.set_title("Body Angle Over Time")
        
        self.ax_angle.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        self.ax_angle.set_xlabel("Frame")
        self.ax_angle.set_ylabel("Angle (degrees)")
        self.ax_angle.grid(True, alpha=0.3)
        self.ax_angle.legend()
        
        # Turning angle (frame-to-frame)
        if len(self.turning_angles) > 0:
            turn_frames = sorted(self.turning_angles.keys())
            turn_angles = [self.turning_angles[f] for f in turn_frames]
            
            if len(turn_frames) == 1:
                # Single turning angle measurement
                self.ax_turning.plot(turn_frames, turn_angles, 'ro', markersize=15, 
                                   label=f'Δ Angle: {turn_angles[0]:.1f}°')
                self.ax_turning.axhline(y=turn_angles[0], color='r', linestyle='--', alpha=0.3)
                self.ax_turning.text(turn_frames[0], turn_angles[0], f'  {turn_angles[0]:.1f}°',
                                   fontsize=12, va='center', ha='left', color='red', fontweight='bold')
                self.ax_turning.set_xlim(turn_frames[0] - 0.5, turn_frames[0] + 0.5)
                self.ax_turning.set_title("Turning Angle (Single Measurement)")
            else:
                # Multiple measurements
                self.ax_turning.plot(turn_frames, turn_angles, 'r-', linewidth=2, marker='o')
                self.ax_turning.set_title("Turning Angle (Frame-to-Frame Change)")
            
            self.ax_turning.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            self.ax_turning.set_xlabel("Frame")
            self.ax_turning.set_ylabel("Turning Angle (degrees)")
            self.ax_turning.grid(True, alpha=0.3)
            self.ax_turning.legend()
        else:
            # No turning angles (e.g., single frame)
            self.ax_turning.text(0.5, 0.5, 'N/A - Single Frame\n(Need multiple frames for turning angle)',
                               ha='center', va='center', transform=self.ax_turning.transAxes,
                               fontsize=11, color='gray', style='italic')
            self.ax_turning.set_title("Turning Angle (N/A)")
            self.ax_turning.set_xlabel("Frame")
            self.ax_turning.set_ylabel("Turning Angle (degrees)")
        
        # Body curvature (if midlines are available)
        if len(self.midline_points) > 0:
            curvatures = []
            curv_frames = []
            
            for frame in sorted(self.midline_points.keys()):
                midline = self.midline_points[frame]
                if len(midline) > 5:
                    # Calculate curvature from midline
                    curvature = self.calculate_curvature(midline)
                    curvatures.append(curvature)
                    curv_frames.append(frame)
            
            if len(curvatures) > 0:
                if len(curvatures) == 1:
                    # Single curvature measurement
                    self.ax_curvature.plot(curv_frames, curvatures, 'go', markersize=15,
                                         label=f'Curvature: {curvatures[0]:.3f}')
                    self.ax_curvature.axhline(y=curvatures[0], color='g', linestyle='--', alpha=0.3)
                    self.ax_curvature.text(curv_frames[0], curvatures[0], f'  {curvatures[0]:.3f}',
                                         fontsize=12, va='center', ha='left', color='green', fontweight='bold')
                    self.ax_curvature.set_xlim(curv_frames[0] - 0.5, curv_frames[0] + 0.5)
                    self.ax_curvature.set_title("Body Curvature (Single Frame)")
                else:
                    # Multiple measurements
                    self.ax_curvature.plot(curv_frames, curvatures, 'g-', linewidth=2, marker='o')
                    self.ax_curvature.set_title("Body Curvature Over Time")
                
                self.ax_curvature.set_xlabel("Frame")
                self.ax_curvature.set_ylabel("Mean Curvature (1/pixels)")
                self.ax_curvature.grid(True, alpha=0.3)
                self.ax_curvature.legend()
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def update_segment_distance_plots(self):
        """Update segment distance plots showing left vs right side distances"""
        if len(self.segment_points) == 0:
            # Clear all subplots and show message
            for i, (ax, seg1, seg2) in enumerate(self.segment_distance_axes):
                ax.clear()
                ax.text(0.5, 0.5, 'No segment data\n(Initialize segments first)',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=9, color='gray', style='italic')
                
                # Y-axis label only on bottom plot
                if i == len(self.segment_distance_axes) - 1:
                    ax.set_ylabel("Distance (px)", fontsize=9)
                    
                ax.grid(True, alpha=0.3, linewidth=0.5)
                
                # Label on left
                title = f"{seg1.upper()}-{seg2.upper()}"
                ax.text(0.02, 0.95, title, transform=ax.transAxes, 
                       fontsize=10, fontweight='bold', va='top', color='lightgray')
                
                # Only show x-label on bottom
                if i == len(self.segment_distance_axes) - 1:
                    ax.set_xlabel("Frame", fontsize=10)
                    
            self.segment_figure.tight_layout()
            self.segment_canvas.draw()
            return
        
        # Get all frames with segment data
        frames = sorted(self.segment_points.keys())
        
        if len(frames) == 0:
            return
        
        # FIRST PASS: Collect all distances to determine global y-axis limits
        all_distances = []
        
        for plot_idx, (ax, seg1, seg2) in enumerate(self.segment_distance_axes):
            left_distances = []
            right_distances = []
            
            for frame in frames:
                segments = self.segment_points[frame]
                
                # Handle special cases: 'head' and 'tail'
                if seg1 == 'head':
                    seg1_left_pos = self.head_positions.get(frame)
                    seg1_right_pos = self.head_positions.get(frame)
                elif seg1 == 'tail':
                    seg1_left_pos = self.tail_positions.get(frame)
                    seg1_right_pos = self.tail_positions.get(frame)
                else:
                    seg1_left_pos = segments.get(f"{seg1}l")
                    seg1_right_pos = segments.get(f"{seg1}r")
                
                if seg2 == 'head':
                    seg2_left_pos = self.head_positions.get(frame)
                    seg2_right_pos = self.head_positions.get(frame)
                elif seg2 == 'tail':
                    seg2_left_pos = self.tail_positions.get(frame)
                    seg2_right_pos = self.tail_positions.get(frame)
                else:
                    seg2_left_pos = segments.get(f"{seg2}l")
                    seg2_right_pos = segments.get(f"{seg2}r")
                
                if seg1_left_pos is not None and seg2_left_pos is not None:
                    dist_left = np.linalg.norm(seg2_left_pos - seg1_left_pos)
                    left_distances.append(dist_left)
                    all_distances.append(dist_left)
                
                if seg1_right_pos is not None and seg2_right_pos is not None:
                    dist_right = np.linalg.norm(seg2_right_pos - seg1_right_pos)
                    right_distances.append(dist_right)
                    all_distances.append(dist_right)
        
        # Calculate global y-axis limits with some padding
        if len(all_distances) > 0:
            y_min = np.min(all_distances)
            y_max = np.max(all_distances)
            y_range = y_max - y_min
            y_padding = y_range * 0.05  # 5% padding
            global_ylim = (y_min - y_padding, y_max + y_padding)
        else:
            global_ylim = None
        
        # SECOND PASS: Plot with consistent y-axis limits
        for plot_idx, (ax, seg1, seg2) in enumerate(self.segment_distance_axes):
            ax.clear()
            
            # Collect distances for left and right sides ONLY
            left_distances = []
            right_distances = []
            valid_frames = []
            
            for frame in frames:
                segments = self.segment_points[frame]
                
                # Handle special cases: 'head' and 'tail'
                # For head/tail, we use the actual head_positions/tail_positions
                # For segments, we use 'l' and 'r' suffixes
                
                # Get seg1 positions (left and right)
                if seg1 == 'head':
                    # Head position (use same for both "left" and "right")
                    seg1_left_pos = self.head_positions.get(frame)
                    seg1_right_pos = self.head_positions.get(frame)
                elif seg1 == 'tail':
                    seg1_left_pos = self.tail_positions.get(frame)
                    seg1_right_pos = self.tail_positions.get(frame)
                else:
                    # Regular segment - use left and right labels
                    seg1_left_pos = segments.get(f"{seg1}l")
                    seg1_right_pos = segments.get(f"{seg1}r")
                
                # Get seg2 positions (left and right)
                if seg2 == 'head':
                    seg2_left_pos = self.head_positions.get(frame)
                    seg2_right_pos = self.head_positions.get(frame)
                elif seg2 == 'tail':
                    seg2_left_pos = self.tail_positions.get(frame)
                    seg2_right_pos = self.tail_positions.get(frame)
                else:
                    # Regular segment - use left and right labels
                    seg2_left_pos = segments.get(f"{seg2}l")
                    seg2_right_pos = segments.get(f"{seg2}r")
                
                # Calculate left distance
                if seg1_left_pos is not None and seg2_left_pos is not None:
                    dist_left = np.linalg.norm(seg2_left_pos - seg1_left_pos)
                    left_distances.append(dist_left)
                else:
                    left_distances.append(np.nan)
                
                # Calculate right distance
                if seg1_right_pos is not None and seg2_right_pos is not None:
                    dist_right = np.linalg.norm(seg2_right_pos - seg1_right_pos)
                    right_distances.append(dist_right)
                else:
                    right_distances.append(np.nan)
                
                valid_frames.append(frame)
            
            # Plot the data
            if len(valid_frames) > 0:
                # Check if we have any valid data
                has_left = not all(np.isnan(left_distances))
                has_right = not all(np.isnan(right_distances))
                
                if has_left or has_right:
                    if len(valid_frames) == 1:
                        # Single frame - show as markers
                        if has_left:
                            ax.plot(valid_frames, left_distances, 'ro', markersize=8, 
                                   label=f'L: {left_distances[0]:.1f}', zorder=3)
                        if has_right:
                            ax.plot(valid_frames, right_distances, 'go', markersize=8, 
                                   label=f'R: {right_distances[0]:.1f}', zorder=3)
                        ax.set_xlim(valid_frames[0] - 0.5, valid_frames[0] + 0.5)
                    else:
                        # Multiple frames - show as smooth lines (like reference image)
                        if has_left:
                            ax.plot(valid_frames, left_distances, 'r-', linewidth=1.5, 
                                   label='Left', alpha=0.9, zorder=3)
                        if has_right:
                            ax.plot(valid_frames, right_distances, 'b-', linewidth=1.5, 
                                   label='Right', alpha=0.9, zorder=3)
                    
                    # Format plot - cleaner style like reference image
                    title = f"{seg1.upper()}-{seg2.upper()}"
                    ax.text(0.02, 0.95, title, transform=ax.transAxes, 
                           fontsize=10, fontweight='bold', va='top')
                    
                    # Clean axis styling
                    ax.grid(True, alpha=0.3, linewidth=0.5, zorder=1)
                    
                    # Apply global y-axis limits for easy comparison across all plots
                    if global_ylim is not None:
                        ax.set_ylim(global_ylim)
                    
                    # Y-axis label only on BOTTOM plot
                    if plot_idx == len(self.segment_distance_axes) - 1:
                        ax.set_ylabel("Distance (px)", fontsize=9)
                    
                    # Legend only on TOP plot
                    if plot_idx == 0:
                        ax.legend(loc='upper right', fontsize=7, framealpha=0.8)
                    
                    # Only show x-label and x-tick labels on bottom subplot
                    if plot_idx == len(self.segment_distance_axes) - 1:
                        ax.set_xlabel("Frame", fontsize=10)
                        ax.tick_params(axis='x', labelbottom=True)
                    else:
                        ax.set_xlabel("")
                        ax.tick_params(axis='x', labelbottom=False)  # Hide x-tick labels
                    
                    # Remove top and right spines for cleaner look
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    
                else:
                    ax.text(0.5, 0.5, f'No valid data for {seg1}-{seg2}',
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=8, color='gray', style='italic')
                    
                    # Apply global y-axis limits for consistency
                    if global_ylim is not None:
                        ax.set_ylim(global_ylim)
                    
                    # Y-axis label only on bottom plot
                    if plot_idx == len(self.segment_distance_axes) - 1:
                        ax.set_ylabel("Distance (px)", fontsize=9)
                    
                    # Hide x-tick labels except on bottom
                    if plot_idx != len(self.segment_distance_axes) - 1:
                        ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.text(0.5, 0.5, 'No data',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=9, color='gray', style='italic')
                
                # Apply global y-axis limits for consistency
                if global_ylim is not None:
                    ax.set_ylim(global_ylim)
                
                # Y-axis label only on bottom plot
                if plot_idx == len(self.segment_distance_axes) - 1:
                    ax.set_ylabel("Distance (px)", fontsize=9)
                
                # Hide x-tick labels except on bottom
                if plot_idx != len(self.segment_distance_axes) - 1:
                    ax.tick_params(axis='x', labelbottom=False)
        
        self.segment_figure.tight_layout()
        self.segment_canvas.draw()
    
    def calculate_curvature(self, midline):
        """Calculate mean curvature of midline"""
        if len(midline) < 5:
            return 0
        
        # Calculate second derivative (approximation of curvature)
        dx = np.gradient(midline[:, 0])
        dy = np.gradient(midline[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # Curvature formula
        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
        
        # Return mean curvature
        return np.mean(curvature[1:-1])  # Exclude endpoints
    
    def save_session(self):
        """Save current head/tail positions and settings to a session file"""
        if len(self.head_positions) == 0 and len(self.tail_positions) == 0:
            QMessageBox.warning(self, "Warning", "No head/tail positions to save!")
            return
        
        # Generate default filename in video's directory
        base_name = self.get_base_filename()
        default_filename = f"{base_name}_session.json"
        default_path = self.get_export_path(default_filename)
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Session", default_path, "Session Files (*.json)"
        )
        
        if filename:
            import json
            import os
            
            # Prepare session data
            session_data = {
                'version': '1.1',  # Updated version to include segments
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'video_file': os.path.basename(self.video_path) if self.video_path else 
                              (os.path.basename(self.tiff_file_path) if self.tiff_file_path else None),
                'roi_path': self.roi_path,
                'total_frames': self.total_frames,
                'settings': {
                    'roi_offset': self.roi_offset_spinbox.value(),
                    'detection_method': self.detection_method.currentText(),
                    'midline_method': self.midline_method.currentText(),
                    'snap_to_roi': self.snap_to_roi_cb.isChecked(),
                    'auto_detect': self.auto_detect_cb.isChecked()
                },
                'head_positions': {int(k): v.tolist() for k, v in self.head_positions.items()},
                'tail_positions': {int(k): v.tolist() for k, v in self.tail_positions.items()},
                'segment_points': {
                    int(frame): {label: pos.tolist() for label, pos in segments.items()}
                    for frame, segments in self.segment_points.items()
                }
            }
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            manual_count = len(self.head_positions)
            QMessageBox.information(
                self, 
                "Success", 
                f"Session saved!\n\n"
                f"Saved {manual_count} head/tail positions\n"
                f"File: {os.path.basename(filename)}\n\n"
                f"You can reload this session later to restore your manual adjustments."
            )
    
    def load_session(self):
        """Load head/tail positions and settings from a session file"""
        # Start in video's directory if available
        video_dir = self.get_video_directory()
        start_dir = video_dir if video_dir else ""
        
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Session", start_dir, "Session Files (*.json)"
        )
        
        if filename:
            import json
            import os
            
            try:
                with open(filename, 'r') as f:
                    session_data = json.load(f)
                
                # Verify version (support both 1.0 and 1.1)
                version = session_data.get('version', '1.0')
                if version not in ['1.0', '1.1']:
                    QMessageBox.warning(self, "Warning", f"Unknown session file version: {version}")
                    return
                
                # Check if video matches (warning only, not blocking)
                saved_video = session_data.get('video_file')
                current_video = os.path.basename(self.video_path) if self.video_path else \
                               (os.path.basename(self.tiff_file_path) if self.tiff_file_path else None)
                
                if saved_video and current_video and saved_video != current_video:
                    reply = QMessageBox.question(
                        self,
                        "Video Mismatch",
                        f"Session was saved for: {saved_video}\n"
                        f"Current video is: {current_video}\n\n"
                        f"The positions might not align correctly.\n"
                        f"Do you want to load anyway?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No
                    )
                    if reply == QMessageBox.No:
                        return
                
                # Load settings
                if 'settings' in session_data:
                    settings = session_data['settings']
                    self.roi_offset_spinbox.setValue(settings.get('roi_offset', 0))
                    
                    # Set detection method
                    method_text = settings.get('detection_method', 'Anterior = Top')
                    idx = self.detection_method.findText(method_text)
                    if idx >= 0:
                        self.detection_method.setCurrentIndex(idx)
                    
                    # Set midline method
                    midline_text = settings.get('midline_method', 'Skeleton')
                    idx = self.midline_method.findText(midline_text)
                    if idx >= 0:
                        self.midline_method.setCurrentIndex(idx)
                    
                    self.snap_to_roi_cb.setChecked(settings.get('snap_to_roi', True))
                    self.auto_detect_cb.setChecked(settings.get('auto_detect', True))
                
                # Load positions
                head_data = session_data.get('head_positions', {})
                tail_data = session_data.get('tail_positions', {})
                
                self.head_positions = {int(k): np.array(v) for k, v in head_data.items()}
                self.tail_positions = {int(k): np.array(v) for k, v in tail_data.items()}
                
                # Load segment points if available (version 1.1+)
                if 'segment_points' in session_data:
                    segment_data = session_data['segment_points']
                    self.segment_points = {
                        int(frame): {label: np.array(pos) for label, pos in segments.items()}
                        for frame, segments in segment_data.items()
                    }
                    if len(self.segment_points) > 0:
                        self.show_segments = True
                        self.show_segments_cb.setChecked(True)
                
                # Update display
                self.update_display()
                
                # Update segment distance plots if segments were loaded
                if len(self.segment_points) > 0:
                    self.update_segment_distance_plots()
                
                loaded_count = len(self.head_positions)
                segment_count = len(self.segment_points)
                segment_info = f"\nLoaded {segment_count} frames with segment points" if segment_count > 0 else ""
                QMessageBox.information(
                    self,
                    "Success",
                    f"Session loaded!\n\n"
                    f"Loaded {loaded_count} head/tail positions{segment_info}\n"
                    f"Saved on: {session_data.get('timestamp', 'Unknown')}\n\n"
                    f"Your manual adjustments have been restored!"
                )
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to load session file:\n{str(e)}"
                )
    
    def get_base_filename(self):
        """Get base filename from loaded video for exports"""
        # Try different video path variables
        video_file = None
        if self.video_path:
            video_file = self.video_path
        elif self.tiff_file_path:
            video_file = self.tiff_file_path
        
        if video_file:
            # Extract filename without extension
            import os
            basename = os.path.basename(video_file)
            name_without_ext = os.path.splitext(basename)[0]
            return name_without_ext
        else:
            return "larva_analysis"
    
    def get_video_directory(self):
        """Get the directory containing the loaded video"""
        import os
        if self.video_path:
            return os.path.dirname(self.video_path)
        elif self.tiff_file_path:
            return os.path.dirname(self.tiff_file_path)
        else:
            return ""  # Current directory
    
    def get_export_path(self, filename):
        """Get full export path (video directory + filename)"""
        import os
        video_dir = self.get_video_directory()
        if video_dir:
            return os.path.join(video_dir, filename)
        else:
            return filename
    
    def export_csv(self):
        """Export analysis results to CSV"""
        if len(self.body_angles) == 0:
            QMessageBox.warning(self, "Warning", "No analysis data to export!")
            return
        
        # Generate default filename in video's directory
        base_name = self.get_base_filename()
        default_filename = f"{base_name}_turning_analysis.csv"
        default_path = self.get_export_path(default_filename)
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", default_path, "CSV Files (*.csv)"
        )
        
        if filename:
            # Prepare data
            data = []
            for frame in sorted(self.body_angles.keys()):
                row = {
                    'Frame': frame,
                    'Body_Angle_deg': self.body_angles[frame],
                    'Turning_Angle_deg': self.turning_angles.get(frame, np.nan)
                }
                
                # Add head/tail coordinates
                if frame in self.head_positions:
                    row['Head_X'] = self.head_positions[frame][0]
                    row['Head_Y'] = self.head_positions[frame][1]
                if frame in self.tail_positions:
                    row['Tail_X'] = self.tail_positions[frame][0]
                    row['Tail_Y'] = self.tail_positions[frame][1]
                
                # Add curvature if available
                if frame in self.midline_points:
                    midline = self.midline_points[frame]
                    row['Mean_Curvature'] = self.calculate_curvature(midline)
                
                # Add segment point coordinates
                if frame in self.segment_points:
                    segments = self.segment_points[frame]
                    # Add all segment coordinates
                    for label in self.segment_labels['left']:
                        if label in segments:
                            row[f'{label}_X'] = segments[label][0]
                            row[f'{label}_Y'] = segments[label][1]
                    for label in self.segment_labels['right']:
                        if label in segments:
                            row[f'{label}_X'] = segments[label][0]
                            row[f'{label}_Y'] = segments[label][1]
                    for label in self.segment_labels['midline']:
                        if label in segments:
                            row[f'{label}_X'] = segments[label][0]
                            row[f'{label}_Y'] = segments[label][1]
                    
                    # Add segment distances
                    distances = self.calculate_segment_distances(frame)
                    row.update(distances)
                
                data.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Write metadata as comments at top of file, then data
            import os
            try:
                with open(filename, 'w') as f:
                    # Write metadata
                    f.write(f"# Larva Turning Angle Analysis\n")
                    f.write(f"# Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    if self.video_path:
                        f.write(f"# Video: {os.path.basename(self.video_path)}\n")
                    elif self.tiff_file_path:
                        f.write(f"# Video: {os.path.basename(self.tiff_file_path)}\n")
                    if self.roi_path:
                        f.write(f"# ROIs: {os.path.basename(self.roi_path)}\n")
                    f.write(f"# Total frames analyzed: {len(self.body_angles)}\n")
                    f.write(f"# ROI offset: {self.roi_offset_spinbox.value()}\n")
                    f.write(f"# Detection method: {self.detection_method.currentText()}\n")
                    f.write(f"# Midline method: {self.midline_method.currentText()}\n")
                    has_segments = len(self.segment_points) > 0
                    f.write(f"# Segment points included: {has_segments}\n")
                    if has_segments:
                        f.write(f"# Segments: T1-T3 (thorax), A1-A7 (abdomen) on left (l), right (r), and midline (st/sa)\n")
                    f.write("#\n")
                    
                    # Write data
                    df.to_csv(f, index=False)
                
                segment_info = f" (with {len(self.segment_points)} frames of segment data)" if len(self.segment_points) > 0 else ""
                QMessageBox.information(self, "Success", f"Data exported to:\n{filename}{segment_info}")
                
            except PermissionError:
                QMessageBox.critical(
                    self,
                    "Permission Error",
                    f"Cannot write to file:\n{filename}\n\n"
                    f"The file may be open in Excel or another program.\n"
                    f"Please close the file and try again."
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Export Error",
                    f"Failed to export CSV:\n{str(e)}"
                )
    
    def export_excel(self):
        """Export analysis results to Excel with multiple sheets"""
        if len(self.body_angles) == 0:
            QMessageBox.warning(self, "Warning", "No analysis data to export!")
            return
        
        # Generate default filename in video's directory
        base_name = self.get_base_filename()
        default_filename = f"{base_name}_analysis.xlsx"
        default_path = self.get_export_path(default_filename)
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Excel", default_path, "Excel Files (*.xlsx)"
        )
        
        if filename:
            try:
                # Try to import openpyxl
                try:
                    import openpyxl
                except ImportError:
                    QMessageBox.critical(
                        self, 
                        "Missing Package",
                        "openpyxl is required for Excel export.\n\n"
                        "Install with: pip install openpyxl"
                    )
                    return
                
                import os
                
                # Prepare metadata
                metadata_dict = {
                    'Parameter': [
                        'Generated',
                        'Video File',
                        'ROI Path',
                        'Total Frames',
                        'ROI Offset',
                        'Detection Method',
                        'Midline Method',
                        'Segments Included'
                    ],
                    'Value': [
                        pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                        os.path.basename(self.video_path) if self.video_path else 
                            (os.path.basename(self.tiff_file_path) if self.tiff_file_path else 'N/A'),
                        os.path.basename(self.roi_path) if self.roi_path else 'N/A',
                        len(self.body_angles),
                        self.roi_offset_spinbox.value(),
                        self.detection_method.currentText(),
                        self.midline_method.currentText(),
                        'Yes' if len(self.segment_points) > 0 else 'No'
                    ]
                }
                df_metadata = pd.DataFrame(metadata_dict)
                
                # Prepare angles and curvature data
                angles_data = []
                for frame in sorted(self.body_angles.keys()):
                    row = {
                        'Frame': frame,
                        'Body_Angle_deg': self.body_angles[frame],
                        'Turning_Angle_deg': self.turning_angles.get(frame, np.nan),
                        'Head_X': self.head_positions[frame][0] if frame in self.head_positions else np.nan,
                        'Head_Y': self.head_positions[frame][1] if frame in self.head_positions else np.nan,
                        'Tail_X': self.tail_positions[frame][0] if frame in self.tail_positions else np.nan,
                        'Tail_Y': self.tail_positions[frame][1] if frame in self.tail_positions else np.nan,
                        'Mean_Curvature': np.nan
                    }
                    
                    if frame in self.midline_points:
                        midline = self.midline_points[frame]
                        row['Mean_Curvature'] = self.calculate_curvature(midline)
                    
                    angles_data.append(row)
                
                df_angles = pd.DataFrame(angles_data)
                
                # Prepare segment coordinates and distances separately
                segment_coords_data = []
                segment_distances_data = []
                
                for frame in sorted(self.body_angles.keys()):
                    if frame in self.segment_points:
                        segments = self.segment_points[frame]
                        
                        # Coordinates sheet
                        coords_row = {'Frame': frame}
                        for label in self.segment_labels['left']:
                            if label in segments:
                                coords_row[f'{label}_X'] = segments[label][0]
                                coords_row[f'{label}_Y'] = segments[label][1]
                        for label in self.segment_labels['right']:
                            if label in segments:
                                coords_row[f'{label}_X'] = segments[label][0]
                                coords_row[f'{label}_Y'] = segments[label][1]
                        for label in self.segment_labels['midline']:
                            if label in segments:
                                coords_row[f'{label}_X'] = segments[label][0]
                                coords_row[f'{label}_Y'] = segments[label][1]
                        segment_coords_data.append(coords_row)
                        
                        # Distances sheet
                        distances_row = {'Frame': frame}
                        distances = self.calculate_segment_distances(frame)
                        distances_row.update(distances)
                        segment_distances_data.append(distances_row)
                
                # Create Excel writer
                with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                    # Write metadata sheet
                    df_metadata.to_excel(writer, sheet_name='Metadata', index=False)
                    
                    # Write angles/curvature sheet
                    df_angles.to_excel(writer, sheet_name='Angles_Curvature', index=False)
                    
                    # Write segment sheets if data exists
                    if segment_coords_data:
                        df_seg_coords = pd.DataFrame(segment_coords_data)
                        df_seg_coords.to_excel(writer, sheet_name='Segment_Coordinates', index=False)
                    
                    if segment_distances_data:
                        df_seg_distances = pd.DataFrame(segment_distances_data)
                        df_seg_distances.to_excel(writer, sheet_name='Segment_Distances', index=False)
                
                sheet_count = 2  # Metadata + Angles
                if segment_coords_data:
                    sheet_count += 2  # Coordinates + Distances
                
                sheets_list = "• Metadata\n• Angles_Curvature"
                if segment_coords_data:
                    sheets_list += "\n• Segment_Coordinates\n• Segment_Distances"
                
                QMessageBox.information(
                    self, 
                    "Success", 
                    f"Excel file exported with {sheet_count} sheets:\n"
                    f"{sheets_list}\n\n"
                    f"File: {filename}"
                )
                
            except PermissionError:
                QMessageBox.critical(
                    self,
                    "Permission Error",
                    f"Cannot write to file:\n{filename}\n\n"
                    f"The file may be open in Excel or another program.\n"
                    f"Please close the file and try again."
                )
            except Exception as e:
                QMessageBox.critical(
                    self, 
                    "Export Error",
                    f"Failed to export Excel file:\n{str(e)}"
                )
    
    def save_plots(self):
        """Save analysis plots to file"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QCheckBox, QDialogButtonBox
        
        # Check what data is available
        has_angles = len(self.body_angles) > 0
        has_segments = len(self.segment_points) > 0
        
        if not has_angles and not has_segments:
            QMessageBox.warning(self, "Warning", "No plots to save!\nAnalyze frames or initialize segments first.")
            return
        
        # Create dialog to choose which plots to save
        dialog = QDialog(self)
        dialog.setWindowTitle("Save Plots")
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("Select which plots to save:"))
        
        # Checkboxes for each plot type
        angles_cb = QCheckBox("Angles & Curvature Plots")
        angles_cb.setEnabled(has_angles)
        angles_cb.setChecked(has_angles)
        if not has_angles:
            angles_cb.setToolTip("No angle data available - analyze frames first")
        layout.addWidget(angles_cb)
        
        segments_cb = QCheckBox("Segment Distance Plots")
        segments_cb.setEnabled(has_segments)
        segments_cb.setChecked(has_segments)
        if not has_segments:
            segments_cb.setToolTip("No segment data available - initialize segments first")
        layout.addWidget(segments_cb)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() == QDialog.Accepted:
            save_angles = angles_cb.isChecked()
            save_segments = segments_cb.isChecked()
            
            if not save_angles and not save_segments:
                QMessageBox.warning(self, "Warning", "Please select at least one plot type to save!")
                return
            
            # Generate default filename
            base_name = self.get_base_filename()
            
            # Get save location
            video_dir = self.get_video_directory()
            start_dir = video_dir if video_dir else ""
            
            saved_files = []
            
            # Save angles/curvature plots
            if save_angles:
                default_filename = f"{base_name}_angles_plots.png"
                default_path = os.path.join(start_dir, default_filename) if start_dir else default_filename
                
                filename, _ = QFileDialog.getSaveFileName(
                    self, "Save Angles & Curvature Plots", default_path, 
                    "PNG Files (*.png);;PDF Files (*.pdf)"
                )
                
                if filename:
                    self.figure.savefig(filename, dpi=300, bbox_inches='tight')
                    saved_files.append(filename)
            
            # Save segment distance plots
            if save_segments:
                default_filename = f"{base_name}_segment_distances.png"
                default_path = os.path.join(start_dir, default_filename) if start_dir else default_filename
                
                filename, _ = QFileDialog.getSaveFileName(
                    self, "Save Segment Distance Plots", default_path,
                    "PNG Files (*.png);;PDF Files (*.pdf)"
                )
                
                if filename:
                    self.segment_figure.savefig(filename, dpi=300, bbox_inches='tight')
                    saved_files.append(filename)
            
            # Show success message
            if saved_files:
                file_list = "\n".join([os.path.basename(f) for f in saved_files])
                QMessageBox.information(
                    self, "Success", 
                    f"Saved {len(saved_files)} plot file(s):\n\n{file_list}"
                )
    
    def export_annotated_video(self):
        """Export video with ROI, segments, and angle overlays"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QCheckBox, QDialogButtonBox, QProgressDialog, QSpinBox
        from PyQt5.QtCore import Qt
        
        # Check if video is loaded
        if self.cap is None and self.tiff_stack is None and not self.single_image_mode:
            QMessageBox.warning(self, "No Video", "Load a video first!")
            return
        
        if self.single_image_mode:
            QMessageBox.warning(self, "Single Image", "Cannot export video from single image!")
            return
        
        # Create dialog for export options
        dialog = QDialog(self)
        dialog.setWindowTitle("Export Annotated Video")
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("Select what to include in the video:"))
        
        # Overlay options
        roi_cb = QCheckBox("Show ROI contours")
        roi_cb.setChecked(True)
        roi_cb.setEnabled(len(self.rois) > 0)
        layout.addWidget(roi_cb)
        
        head_tail_cb = QCheckBox("Show head/tail markers")
        head_tail_cb.setChecked(True)
        head_tail_cb.setEnabled(len(self.head_positions) > 0)
        layout.addWidget(head_tail_cb)
        
        segments_cb = QCheckBox("Show segment points")
        segments_cb.setChecked(len(self.segment_points) > 0)
        segments_cb.setEnabled(len(self.segment_points) > 0)
        layout.addWidget(segments_cb)
        
        midline_cb = QCheckBox("Show midline")
        midline_cb.setChecked(False)
        midline_cb.setEnabled(len(self.midline_points) > 0)
        layout.addWidget(midline_cb)
        
        angle_cb = QCheckBox("Show body angle text")
        angle_cb.setChecked(True)
        angle_cb.setEnabled(len(self.body_angles) > 0)
        layout.addWidget(angle_cb)
        
        # FPS option
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("Output FPS:"))
        fps_spinbox = QSpinBox()
        fps_spinbox.setMinimum(1)
        fps_spinbox.setMaximum(120)
        fps_spinbox.setValue(30)
        fps_spinbox.setToolTip("Frames per second for output video")
        fps_layout.addWidget(fps_spinbox)
        layout.addLayout(fps_layout)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() != QDialog.Accepted:
            return
        
        # Get options
        show_roi = roi_cb.isChecked()
        show_head_tail = head_tail_cb.isChecked()
        show_segments = segments_cb.isChecked()
        show_midline = midline_cb.isChecked()
        show_angle = angle_cb.isChecked()
        output_fps = fps_spinbox.value()
        
        # Get save location
        base_name = self.get_base_filename()
        default_filename = f"{base_name}_annotated.mp4"
        video_dir = self.get_video_directory()
        default_path = os.path.join(video_dir, default_filename) if video_dir else default_filename
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Annotated Video", default_path,
            "MP4 Video (*.mp4);;AVI Video (*.avi)"
        )
        
        if not filename:
            return
        
        # Create progress dialog
        progress = QProgressDialog("Exporting annotated video...", "Cancel", 0, self.total_frames, self)
        progress.setWindowTitle("Exporting Video")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        
        try:
            # Get video properties
            if self.tiff_stack is not None:
                height, width = self.tiff_stack[0].shape[:2]
            else:
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Determine codec based on file extension
            if filename.lower().endswith('.avi'):
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # Create video writer
            out = cv2.VideoWriter(filename, fourcc, output_fps, (width, height))
            
            if not out.isOpened():
                QMessageBox.critical(self, "Error", "Failed to create video file!")
                return
            
            # Process each frame
            for frame_idx in range(self.total_frames):
                if progress.wasCanceled():
                    break
                
                progress.setValue(frame_idx)
                progress.setLabelText(f"Processing frame {frame_idx+1}/{self.total_frames}...")
                
                # Get frame
                if self.tiff_stack is not None:
                    frame = self.tiff_stack[frame_idx].copy()
                    if len(frame.shape) == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    elif len(frame.shape) == 3 and frame.shape[2] == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = self.cap.read()
                    if not ret:
                        continue
                
                # Apply overlays
                roi_index = frame_idx + self.roi_offset_spinbox.value()
                
                # Draw ROI
                if show_roi and 0 <= roi_index < len(self.rois) and self.rois[roi_index] is not None:
                    roi = self.rois[roi_index]
                    cv2.polylines(frame, [roi.astype(np.int32)], True, (0, 255, 0), 2)
                
                # Draw head/tail
                if show_head_tail:
                    if frame_idx in self.head_positions:
                        head = self.head_positions[frame_idx]
                        cv2.circle(frame, tuple(head.astype(int)), 8, (0, 0, 255), -1)
                        cv2.putText(frame, "H", tuple((head + [10, -10]).astype(int)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    if frame_idx in self.tail_positions:
                        tail = self.tail_positions[frame_idx]
                        cv2.circle(frame, tuple(tail.astype(int)), 8, (255, 0, 0), -1)
                        cv2.putText(frame, "T", tuple((tail + [10, -10]).astype(int)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Draw midline
                if show_midline and frame_idx in self.midline_points:
                    midline = self.midline_points[frame_idx]
                    for i in range(len(midline) - 1):
                        pt1 = tuple(midline[i].astype(int))
                        pt2 = tuple(midline[i+1].astype(int))
                        cv2.line(frame, pt1, pt2, (255, 0, 255), 2)
                
                # Draw segments
                if show_segments and frame_idx in self.segment_points:
                    segments = self.segment_points[frame_idx]
                    
                    # Draw segment points
                    for label, pos in segments.items():
                        color = (255, 255, 0) if label.startswith('s') else \
                               (0, 255, 255) if label.endswith('l') else (0, 165, 255)
                        cv2.circle(frame, tuple(pos.astype(int)), 4, color, -1)
                    
                    # Draw cross-sectional lines (through midline, matching GUI display)
                    for seg in ['t1', 't2', 't3', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7']:
                        left_key = f"{seg}l"
                        right_key = f"{seg}r"
                        mid_key = f"s{seg}"
                        
                        # Draw lines through midline: left -> midline -> right
                        if left_key in segments and right_key in segments and mid_key in segments:
                            left_pt = tuple(segments[left_key].astype(int))
                            mid_pt = tuple(segments[mid_key].astype(int))
                            right_pt = tuple(segments[right_key].astype(int))
                            
                            # Two lines: left->mid and mid->right (matches GUI)
                            cv2.line(frame, left_pt, mid_pt, (200, 200, 200), 1)
                            cv2.line(frame, mid_pt, right_pt, (200, 200, 200), 1)
                
                # Draw angle text
                if show_angle and frame_idx in self.body_angles:
                    angle = self.body_angles[frame_idx]
                    cv2.putText(frame, f"Angle: {angle:.1f} deg", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Frame: {frame_idx}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Write frame
                out.write(frame)
            
            # Clean up
            out.release()
            progress.setValue(self.total_frames)
            
            QMessageBox.information(
                self, "Success",
                f"Annotated video exported successfully!\n\n"
                f"Location: {os.path.basename(filename)}\n"
                f"Frames: {self.total_frames}\n"
                f"FPS: {output_fps}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export video:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def on_frame_changed(self, value):

        self.current_frame = value
        self.display_frame()
    
    def prev_frame(self):
        if self.current_frame > 0:
            self.frame_slider.setValue(self.current_frame - 1)
    
    def next_frame(self):
        if self.current_frame < self.total_frames - 1:
            self.frame_slider.setValue(self.current_frame + 1)
    
    def jump_to_frame(self, value):
        self.frame_slider.setValue(value)
    
    def on_auto_detect_changed(self):
        self.update_display()
    
    def on_video_click(self, event):
        """Handle mouse press on video"""
        if self.current_display_frame is None:
            return
        
        from PyQt5.QtCore import Qt
        
        # Handle ROI drawing mode
        if self.drawing_mode:
            if event.button() == Qt.LeftButton:
                # Add point to ROI
                x, y = self.screen_to_frame_coords(event.pos().x(), event.pos().y())
                self.roi_drawing_points.append([x, y])
                self.update_display()
                return
            elif event.button() == Qt.RightButton:
                # Finish ROI
                self.finish_roi_drawing()
                return
        
        # Handle right-click for panning (when not drawing)
        if event.button() == Qt.RightButton:
            self.panning = True
            self.pan_start_x = event.pos().x()
            self.pan_start_y = event.pos().y()
            self.video_label.setCursor(Qt.ClosedHandCursor)
            return
        
        # Get click coordinates with zoom/pan
        x, y = self.screen_to_frame_coords(event.pos().x(), event.pos().y())
        click_pos = np.array([x, y])
        
        # Check if clicking near segment points first (if visible)
        if self.show_segments and self.current_frame in self.segment_points:
            segments = self.segment_points[self.current_frame]
            for label, pos in segments.items():
                dist = np.linalg.norm(click_pos - pos)
                if dist < self.drag_threshold / self.zoom_level:  # Adjust threshold for zoom
                    self.dragging = True
                    self.drag_target = 'segment'
                    self.dragging_segment = label
                    self.video_label.setCursor(Qt.ClosedHandCursor)
                    self.info_label.setText(f"🖱 Dragging {label}...")
                    print(f"Started dragging segment {label} (distance: {dist:.1f})")
                    return
        
        # Check if clicking near existing head/tail (for dragging)
        if self.current_frame in self.head_positions:
            head_pos = self.head_positions[self.current_frame]
            dist = np.linalg.norm(click_pos - head_pos)
            if dist < self.drag_threshold / self.zoom_level:
                self.dragging = True
                self.drag_target = 'head'
                self.video_label.setCursor(Qt.ClosedHandCursor)
                self.info_label.setText("🖱 Dragging HEAD...")
                print(f"Started dragging HEAD (distance: {dist:.1f})")
                return
        
        if self.current_frame in self.tail_positions:
            tail_pos = self.tail_positions[self.current_frame]
            dist = np.linalg.norm(click_pos - tail_pos)
            if dist < self.drag_threshold / self.zoom_level:
                self.dragging = True
                self.drag_target = 'tail'
                self.video_label.setCursor(Qt.ClosedHandCursor)
                self.info_label.setText("🖱 Dragging TAIL...")
                print(f"Started dragging TAIL (distance: {dist:.1f})")
                return
        
        # If manual mode and not dragging, set new points
        if self.manual_adjust_mode:
            # Start with click position
            final_pos = click_pos
            snap_info = ""
            
            # Apply snapping if enabled
            if self.snap_to_roi_cb.isChecked():
                final_pos = self.snap_to_roi(click_pos)
                snap_distance = np.linalg.norm(final_pos - click_pos)
                
                # Check if ROI was available
                roi_index = self.current_frame + self.roi_offset_spinbox.value()
                if 0 <= roi_index < len(self.rois) and self.rois[roi_index] is not None:
                    snap_info = f" (snapped {snap_distance:.1f}px to ROI)"
                else:
                    snap_info = " (no ROI available)"
            else:
                snap_info = " (snapping disabled)"
            
            if self.setting_head:
                self.head_positions[self.current_frame] = final_pos
                self.info_label.setText(f"✓ Head set at ({int(final_pos[0])}, {int(final_pos[1])}){snap_info}. Click to set TAIL.")
                self.setting_head = False
                print(f"Set HEAD at ({int(final_pos[0])}, {int(final_pos[1])}){snap_info}")
            else:
                self.tail_positions[self.current_frame] = final_pos
                self.info_label.setText(f"✓ Tail set at ({int(final_pos[0])}, {int(final_pos[1])}){snap_info}. Hover to drag or click for HEAD.")
                self.setting_head = True
                print(f"Set TAIL at ({int(final_pos[0])}, {int(final_pos[1])}){snap_info}")
            
            self.update_display()
    
    def on_video_move(self, event):
        """Handle mouse move for dragging, panning, and hover"""
        if self.current_display_frame is None:
            return
        
        from PyQt5.QtCore import Qt
        
        # Handle panning
        if self.panning:
            delta_x = event.pos().x() - self.pan_start_x
            delta_y = event.pos().y() - self.pan_start_y
            self.pan_offset_x += delta_x
            self.pan_offset_y += delta_y
            self.pan_start_x = event.pos().x()
            self.pan_start_y = event.pos().y()
            self.update_display()
            return
        
        # Get mouse coordinates with zoom/pan
        x, y = self.screen_to_frame_coords(event.pos().x(), event.pos().y())
        mouse_pos = np.array([x, y], dtype=float)
        
        # If dragging, update position with ROI snapping
        if self.dragging:
            # Start with mouse position
            final_pos = mouse_pos
            
            # Apply snapping if enabled
            if self.snap_to_roi_cb.isChecked():
                # Use constrained snapping for segments
                if self.drag_target == 'segment':
                    final_pos = self.snap_segment_to_roi(mouse_pos, self.dragging_segment)
                else:
                    final_pos = self.snap_to_roi(mouse_pos)
                
                snap_distance = np.linalg.norm(final_pos - mouse_pos)
                
                if self.drag_target == 'head':
                    self.head_positions[self.current_frame] = final_pos
                    self.info_label.setText(f"🖱 HEAD → ROI ({int(final_pos[0])}, {int(final_pos[1])}) [snap: {snap_distance:.1f}px]")
                elif self.drag_target == 'tail':
                    self.tail_positions[self.current_frame] = final_pos
                    self.info_label.setText(f"🖱 TAIL → ROI ({int(final_pos[0])}, {int(final_pos[1])}) [snap: {snap_distance:.1f}px]")
                elif self.drag_target == 'segment':
                    self.segment_points[self.current_frame][self.dragging_segment] = final_pos
                    self.info_label.setText(f"🖱 {self.dragging_segment} → ROI (constrained) ({int(final_pos[0])}, {int(final_pos[1])}) [snap: {snap_distance:.1f}px]")
            else:
                # Free positioning without snapping
                if self.drag_target == 'head':
                    self.head_positions[self.current_frame] = final_pos
                    self.info_label.setText(f"🖱 Dragging HEAD (free) ({x}, {y})")
                elif self.drag_target == 'tail':
                    self.tail_positions[self.current_frame] = final_pos
                    self.info_label.setText(f"🖱 Dragging TAIL (free) ({x}, {y})")
                elif self.drag_target == 'segment':
                    self.segment_points[self.current_frame][self.dragging_segment] = final_pos
                    self.info_label.setText(f"🖱 Dragging {self.dragging_segment} (free) ({x}, {y})")
            
            # Force immediate update
            self.update_display()
            QApplication.processEvents()  # Process events immediately
        else:
            # Check if hovering over segment points, head or tail to change cursor
            from PyQt5.QtCore import Qt
            near_marker = False
            
            # Adjust threshold for zoom
            adjusted_threshold = self.drag_threshold / self.zoom_level
            
            # Check segment points first (if visible)
            if self.show_segments and self.current_frame in self.segment_points:
                segments = self.segment_points[self.current_frame]
                for label, pos in segments.items():
                    if np.linalg.norm(mouse_pos - pos) < adjusted_threshold:
                        self.video_label.setCursor(Qt.OpenHandCursor)
                        near_marker = True
                        break
            
            if not near_marker and self.current_frame in self.head_positions:
                head_pos = self.head_positions[self.current_frame]
                if np.linalg.norm(mouse_pos - head_pos) < adjusted_threshold:
                    self.video_label.setCursor(Qt.OpenHandCursor)
                    near_marker = True
            
            if not near_marker and self.current_frame in self.tail_positions:
                tail_pos = self.tail_positions[self.current_frame]
                if np.linalg.norm(mouse_pos - tail_pos) < adjusted_threshold:
                    self.video_label.setCursor(Qt.OpenHandCursor)
                    near_marker = True
            
            if not near_marker:
                self.video_label.setCursor(Qt.ArrowCursor)
    
    def on_video_release(self, event):
        """Handle mouse release"""
        from PyQt5.QtCore import Qt
        
        # Handle panning release
        if event.button() == Qt.RightButton:
            self.panning = False
            self.video_label.setCursor(Qt.ArrowCursor)
            return
        
        if self.dragging:
            self.dragging = False
            target = self.drag_target if self.drag_target != 'segment' else self.dragging_segment
            was_segment = self.drag_target == 'segment'
            self.drag_target = None
            self.dragging_segment = None
            self.video_label.setCursor(Qt.ArrowCursor)
            if target:
                self.info_label.setText(f"✓ {target.upper()} position updated")
            self.update_display()
            
            # Update segment distance plots if we dragged a segment
            if was_segment:
                self.update_segment_distance_plots()
    
    def keyPressEvent(self, event):
        """Handle keyboard events"""
        from PyQt5.QtCore import Qt
        
        # Esc key cancels ROI drawing
        if event.key() == Qt.Key_Escape and self.drawing_mode:
            self.cancel_roi_drawing()
            event.accept()
        else:
            super().keyPressEvent(event)
    
    def on_video_wheel(self, event):
        """Handle mouse wheel for zooming"""
        if self.current_display_frame is None:
            return
        
        # Get mouse position before zoom
        label_size = self.video_label.size()
        mouse_x = event.pos().x()
        mouse_y = event.pos().y()
        
        # Zoom factor
        from PyQt5.QtCore import Qt
        if event.angleDelta().y() > 0:
            zoom_factor = 1.15  # Zoom in
        else:
            zoom_factor = 0.85  # Zoom out
        
        # Update zoom level
        old_zoom = self.zoom_level
        self.zoom_level *= zoom_factor
        self.zoom_level = max(0.5, min(self.zoom_level, 10.0))  # Limit zoom range
        
        # Adjust pan to zoom around mouse position
        if old_zoom != self.zoom_level:
            # Convert mouse position to frame coordinates
            frame_h, frame_w = self.current_display_frame.shape[:2]
            frame_x = (mouse_x - label_size.width()/2 - self.pan_offset_x) / old_zoom + frame_w/2
            frame_y = (mouse_y - label_size.height()/2 - self.pan_offset_y) / old_zoom + frame_h/2
            
            # Recalculate pan offset to keep same point under mouse
            self.pan_offset_x = mouse_x - label_size.width()/2 - (frame_x - frame_w/2) * self.zoom_level
            self.pan_offset_y = mouse_y - label_size.height()/2 - (frame_y - frame_h/2) * self.zoom_level
        
        self.update_display()
        self.info_label.setText(f"Zoom: {self.zoom_level:.1f}x")
    
    def adjust_zoom(self, factor):
        """Adjust zoom by a factor"""
        self.zoom_level *= factor
        self.zoom_level = max(0.5, min(self.zoom_level, 10.0))
        self.update_display()
        self.info_label.setText(f"Zoom: {self.zoom_level:.1f}x")
    
    def reset_view(self):
        """Reset zoom and pan to default"""
        self.zoom_level = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.update_display()
        self.info_label.setText("View reset")
    
    def screen_to_frame_coords(self, screen_x, screen_y):
        """Convert screen coordinates to frame coordinates with zoom/pan"""
        if self.current_display_frame is None:
            return screen_x, screen_y
        
        label_size = self.video_label.size()
        frame_h, frame_w = self.current_display_frame.shape[:2]
        
        # Account for zoom and pan
        frame_x = (screen_x - label_size.width()/2 - self.pan_offset_x) / self.zoom_level + frame_w/2
        frame_y = (screen_y - label_size.height()/2 - self.pan_offset_y) / self.zoom_level + frame_h/2
        
        # Clamp to frame bounds
        frame_x = max(0, min(frame_w - 1, frame_x))
        frame_y = max(0, min(frame_h - 1, frame_y))
        
        return int(frame_x), int(frame_y)
    
    def toggle_manual_mode(self, enabled):
        """Toggle manual adjustment mode"""
        self.manual_adjust_mode = enabled
        
        if enabled:
            self.manual_mode_label.setText("Manual mode: ON - Click on video!")
            self.manual_mode_label.setStyleSheet("color: green; font-weight: bold;")
            self.info_label.setText("Click on video to set HEAD first, then TAIL")
            self.setting_head = True
        else:
            self.manual_mode_label.setText("Manual mode: OFF")
            self.manual_mode_label.setStyleSheet("color: gray;")
            self.info_label.setText("Manual mode disabled")
    
    def copy_to_adjacent(self, direction):
        """Copy current head/tail to adjacent frame (direction: -1 for prev, +1 for next)"""
        if self.current_frame not in self.head_positions or self.current_frame not in self.tail_positions:
            QMessageBox.warning(self, "Warning", "No head/tail positions set for current frame!")
            return
        
        target_frame = self.current_frame + direction
        
        if target_frame < 0 or target_frame >= self.total_frames:
            QMessageBox.warning(self, "Warning", "Target frame out of range!")
            return
        
        # Get positions from current frame
        head_pos = self.head_positions[self.current_frame].copy()
        tail_pos = self.tail_positions[self.current_frame].copy()
        
        # Navigate to target frame first (so snap_to_roi uses correct ROI)
        old_frame = self.current_frame
        self.frame_slider.setValue(target_frame)
        
        # Apply ROI snapping if enabled
        snap_info = ""
        if self.snap_to_roi_cb.isChecked():
            # Snap to target frame's ROI
            snapped_head = self.snap_to_roi(head_pos)
            snapped_tail = self.snap_to_roi(tail_pos)
            
            head_snap_dist = np.linalg.norm(snapped_head - head_pos)
            tail_snap_dist = np.linalg.norm(snapped_tail - tail_pos)
            
            self.head_positions[target_frame] = snapped_head
            self.tail_positions[target_frame] = snapped_tail
            
            snap_info = f" (snapped: H={head_snap_dist:.1f}px, T={tail_snap_dist:.1f}px)"
        else:
            # Direct copy without snapping
            self.head_positions[target_frame] = head_pos
            self.tail_positions[target_frame] = tail_pos
            snap_info = " (no snapping)"
        
        direction_str = "previous" if direction == -1 else "next"
        self.info_label.setText(f"✓ Copied from frame {old_frame} to {direction_str} frame{snap_info}")
        
        # Update display to show new positions
        self.update_display()
    
    def copy_segments_to_adjacent(self, direction):
        """Copy current segment points to adjacent frame (direction: -1 for prev, +1 for next)"""
        if self.current_frame not in self.segment_points:
            QMessageBox.warning(self, "Warning", "No segment points set for current frame!")
            return
        
        target_frame = self.current_frame + direction
        
        if target_frame < 0 or target_frame >= self.total_frames:
            QMessageBox.warning(self, "Warning", "Target frame out of range!")
            return
        
        # Get segment positions from current frame
        current_segments = self.segment_points[self.current_frame]
        
        # Navigate to target frame first (so snap_to_roi uses correct ROI)
        old_frame = self.current_frame
        self.frame_slider.setValue(target_frame)
        
        # Copy and optionally snap all segment points
        new_segments = {}
        total_snap_dist = 0
        
        # First copy all positions to target frame so constrained snapping works
        if self.snap_to_roi_cb.isChecked():
            # Create initial copy in target frame
            self.segment_points[target_frame] = {label: pos.copy() for label, pos in current_segments.items()}
            
            # Now snap each segment point with constraints
            for label, pos in current_segments.items():
                snapped_pos = self.snap_segment_to_roi(pos.copy(), label)
                new_segments[label] = snapped_pos
                total_snap_dist += np.linalg.norm(snapped_pos - pos)
            
            avg_snap_dist = total_snap_dist / len(current_segments)
            snap_info = f" (avg constrained snap: {avg_snap_dist:.1f}px)"
        else:
            # Direct copy without snapping
            new_segments = {label: pos.copy() for label, pos in current_segments.items()}
            snap_info = " (no snapping)"
        
        self.segment_points[target_frame] = new_segments
        
        direction_str = "previous" if direction == -1 else "next"
        self.info_label.setText(f"✓ Copied {len(new_segments)} segments from frame {old_frame} to {direction_str} frame{snap_info}")
        
        # Update display to show new positions
        self.update_display()
        
        # Update segment distance plots
        self.update_segment_distance_plots()


def main():
    app = QApplication(sys.argv)
    window = LarvaTurningAnalyzer()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
