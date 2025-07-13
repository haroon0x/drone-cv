# Architecture

YOLOv8 is used as the baseline. It is a single-stage object detector with a CSPDarknet backbone, PANet neck, and detection head. It is fast and accurate for real-time drone-based person detection.

## Planned Custom Modules

- Extra detection head for small objects (P2 layer)
- Multi-scale Feature Aggregation Module (MFAM)
- Improved Efficient Multi-scale Attention (IEMA)
- Dimension-Aware Selective Integration (DASI)

These modules are inspired by MASF-YOLO and will be implemented as custom PyTorch layers and integrated into the YOLOv8 backbone or head for improved small object detection in aerial images. 