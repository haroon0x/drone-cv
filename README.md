# drone-cv

## Project Overview
A computer vision system for accurate, real-time person detection from drone imagery at 50m altitude. Utilizes state-of-the-art deep learning models (YOLOv5/YOLOv8, MASF-YOLO) and is designed for robust performance in challenging aerial scenarios.

## Features
- Detects persons from UAV images at high altitude
- Supports small object detection and complex backgrounds
- Real-time inference capability

## Project Structure
- data: datasets and YOLO-formatted data
- papers: research papers and documentation
- scripts/data: dataset download and preparation
- scripts/train: training scripts
- scripts/eval: evaluation scripts
- scripts/infer: inference scripts

## Setup
1. Clone this repository
2. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```
3. Download or prepare datasets (see design.md for recommendations)

## Usage
- Use scripts in scripts/data to prepare datasets
- Use scripts in scripts/train to train models
- Use scripts in scripts/eval to evaluate models
- Use scripts in scripts/infer for inference

## References
- See design.md for research background and technical details.