import cv2
import torch
import torchvision
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import numpy as np
import time

def yolo_detect(image_path):
    # 加载最新的YOLO模型 (YOLOv8)
    model = YOLO('yolov8n.pt')
    
    # 读取图像
    image = cv2.imread(image_path)
    
    # 检查图像是否成功加载
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None, {}
    
    # 记录开始时间
    start_time = time.time()
    
    # 进行检测
    results = model(image)
    
    # 处理结果
    detections = results[0].boxes.data
    object_count = {}
    
    for detection in detections:
        class_id = int(detection[5])
        class_name = model.names[class_id]
        confidence = detection[4]
        x1, y1, x2, y2 = detection[:4]
        
        # 绘制边界框
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # 计数
        if class_name in object_count:
            object_count[class_name] += 1
        else:
            object_count[class_name] = 1
    
    # 计算运行时间
    end_time = time.time()
    run_time = end_time - start_time
    
    return image, object_count, run_time

def faster_rcnn_detect(image_path):
    # 加载最新的Faster R-CNN模型
    model = fasterrcnn_resnet50_fpn_v2(pretrained=True)
    model.eval()
    
    # 读取图像
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 预处理图像
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    input_image = transform(image_rgb)
    
    # 记录开始时间
    start_time = time.time()
    
    # 进行检测
    with torch.no_grad():
        prediction = model([input_image])
    
    # 处理结果
    object_count = {}
    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']
    
    # 过滤低置信度的检测结果
    mask = scores > 0.5
    boxes = boxes[mask]
    labels = labels[mask]
    
    # 定义COCO数据集的类别
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    # 绘制边界框
    result_image = draw_bounding_boxes(input_image, boxes, labels=[f"{COCO_INSTANCE_CATEGORY_NAMES[label]}" for label in labels])
    result_image = np.array(to_pil_image(result_image))
    
    # 计数
    for label in labels:
        class_name = COCO_INSTANCE_CATEGORY_NAMES[label]
        if class_name in object_count:
            object_count[class_name] += 1
        else:
            object_count[class_name] = 1
    
    # 计算运行时间
    end_time = time.time()
    run_time = end_time - start_time
    
    return result_image, object_count, run_time

def main():
    image_path = 'test10.jpg'  # 请确保此路径正确
    
    # YOLO检测
    yolo_image, yolo_count, yolo_time = yolo_detect(image_path)
    if yolo_image is not None:
        cv2.imshow('YOLO Detection', yolo_image)
        print("YOLO检测结果:")
        for obj, count in yolo_count.items():
            print(f"{obj}: {count}")
        print(f"YOLO运行时间: {yolo_time:.2f}秒")
    else:
        print("YOLO检测失败")
    
    # Faster R-CNN检测
    # rcnn_image, rcnn_count, rcnn_time = faster_rcnn_detect(image_path)
    # if rcnn_image is not None:
    #     cv2.imshow('Faster R-CNN Detection', rcnn_image)
    #     print("\nFaster R-CNN检测结果:")
    #     for obj, count in rcnn_count.items():
    #         print(f"{obj}: {count}")
    #     print(f"Faster R-CNN运行时间: {rcnn_time:.2f}秒")
    # else:
    #     print("Faster R-CNN检测失败")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
