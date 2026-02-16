# Larva Turning Angle Analyzer

PyQt5 application for analyzing *Drosophila* larva locomotion from video microscopy.
![Application Screenshot](images/Capture.PNG)

## Installation
```bash
pip install PyQt5 opencv-python numpy scipy matplotlib scikit-image pandas openpyxl pillow
```

## Quick Start
```bash
python larva_turning_analyzer.py
```

1. Load video (AVI/MP4/MOV/TIFF) and ROIs (ImageJ .zip)
2. Set head/tail positions (auto-detect or manual)
3. Initialize segments for detailed tracking (optional)
4. Click "Analyze All Frames"
5. Export data, plots, or annotated video

## Features

- **Body Measurements:** Angle, turning angle, curvature
- **Segment Tracking:** 30-point system (T1-T3, A1-A7)
- **Distance Analysis:** 11 segment pairs with unified scale
- **ROI Tools:** Auto-detect (CLAHE, adaptive threshold), manual draw, ImageJ import
- **Visualization:** Real-time overlays, zoom/pan
- **Export:** Excel, CSV, PNG/PDF plots, MP4/AVI video

## Measurements

- **Body Angle:** Head-tail orientation (-180° to 180°)
- **Turning Angle:** Frame-to-frame directional changes
- **Curvature:** Mean along midline (1/pixels)
- **Segment Distances:** Arc length along left/right/midline

## Outputs

### Excel Multi-Sheet
- Metadata (video info, settings, timestamp)
- Frame_Analysis (angles, curvature, head/tail positions)
- Segment_Measurements (all 11 segment pair distances)

### Plots  
- Angles & Curvature (3 plots: body angle, turning angle, curvature)
- Segment Distances (11 stacked plots: HEAD-T1 through A7-TAIL)
- Format: PNG (300 DPI) or PDF (vector)

### Annotated Video
- Customizable overlays: ROI, head/tail, segments, midline, angle text
- Format: MP4 (recommended) or AVI
- Adjustable FPS

## Tips

- **Left panel scrollable** - Scroll down to see Export buttons
- **Low-contrast images** - Enable CLAHE and adaptive threshold in auto-detect
- **Manual mode** - Uncheck "Auto-detect endpoints" for problem frames
- **Save sessions** - Preserve work with JSON save/load (File menu)
- **Unified scale** - Segment distance plots share y-axis for easy comparison

## Color Code

- **Red:** Head marker, Left side traces/points
- **Blue:** Tail marker
- **Green:** ROI contour, Right side traces/points
- **Yellow:** Midline segment points
- **Magenta:** Midline curve, Body axis
- **Gray:** Cross-sectional lines (left→midline→right)

## Workflow Example
```
1. Load Files
   ├─ Video: video.mp4
   └─ ROIs: rois.zip (from ImageJ)

2. Set Head/Tail
   ├─ Method: "Anterior = Top"
   ├─ Auto-detect: ☑
   ├─ Snap to ROI: ☑
   └─ Apply to All Frames

3. Initialize Segments (Optional)
   ├─ Click "Initialize Segments"
   ├─ Apply to All
   └─ Review segment positions

4. Analyze
   └─ Click "Analyze All Frames"

5. Review Plots
   ├─ Tab 1: Angles & Curvature
   └─ Tab 2: Segment Distances

6. Export
   ├─ Excel: Multi-sheet with all data
   ├─ Plots: PNG or PDF
   ├─ Video: Annotated MP4
   └─ Session: Save for later
```

## Troubleshooting

- **Can't see Export buttons:** Scroll down in left panel
- **ROIs not aligned:** Adjust ROI Offset spinbox
- **Low-contrast detection:** Enable CLAHE + adaptive threshold, blur 3.5-4.5
- **Head/tail swapped:** Change detection method or use manual mode
- **Empty plots:** Click "Analyze All Frames" first

## Documentation

Detailed guides included:
- `ANGLE_CALCULATION_EXPLAINED.md` - How angles and curvature are calculated
- `SEGMENT_DISTANCE_TRACKING.md` - 30-point segment system
- `AUTO_DETECT_ROI_GUIDE.md` - ROI detection with CLAHE
- `VIDEO_EXPORT_FEATURE.md` - Annotated video export
- `UNITS_AND_PLOTS_EXPLAINED.md` - Measurement units and conversions

## System Requirements

- Python 3.7+
- 2+ GB RAM
- Works on Windows, macOS, Linux

## License

MIT License - Free to use, modify, and distribute

## Citation

If you use this software in your research, please cite:
```
Larva Turning Angle Analyzer v2.0
```

---

**Happy Analyzing!** 🐛📊✨
