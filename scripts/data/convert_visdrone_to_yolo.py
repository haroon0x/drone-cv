import os
det_dir = 'data/VisDrone2019/VisDrone2019-DET-train/annotations'
img_dir = 'data/VisDrone2019/VisDrone2019-DET-train/images'
yolo_img_dir = 'data/yolo/visdrone/images'
yolo_label_dir = 'data/yolo/visdrone/labels'
os.makedirs(yolo_img_dir, exist_ok=True)
os.makedirs(yolo_label_dir, exist_ok=True)
for fname in os.listdir(det_dir):
    if fname.endswith('.txt'):
        print('Found annotation', fname) 