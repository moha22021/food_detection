import cv2
import numpy as np
import json
import torch
from torchvision import models, transforms
from PIL import Image

# Load pre-trained model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# COCO class names
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A',
    'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Define food classes
food_classes = [
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake'
]

food_class_indices = [COCO_INSTANCE_CATEGORY_NAMES.index(cls) for cls in food_classes]

transform = transforms.Compose([
    transforms.ToTensor(),
])

def estimate_weight(class_name, area):
    average_weights = {
        'apple': 182,
        'banana': 118,
        'sandwich': 150,
        'orange': 131,
        'broccoli': 91,
        'carrot': 61,
        'hot dog': 100,
        'pizza': 150,
        'donut': 50,
        'cake': 200
    }
    base_area = 10000
    weight = average_weights.get(class_name, 100) * (area / base_area)
    return weight

def detect_food(image_path='image.png'):
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image)
        input_tensor = image_tensor.unsqueeze(0)

        # Perform detection
        with torch.no_grad():
            outputs = model(input_tensor)

        # Process detections
        detected_items = []
        for idx, score in enumerate(outputs[0]['scores']):
            if score > 0.5:
                class_id = outputs[0]['labels'][idx].item()
                if class_id in food_class_indices:
                    class_name = COCO_INSTANCE_CATEGORY_NAMES[class_id]
                    box = outputs[0]['boxes'][idx].tolist()
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    weight = estimate_weight(class_name, area)
                    detected_items.append({
                        'class_name': class_name,
                        'box': box,
                        'score': score.item(),
                        'estimated_weight': weight
                    })

        result = {item['class_name']: item['estimated_weight'] for item in detected_items}
        return result, detected_items

    except FileNotFoundError:
        print("Error: 'image.png' not found in the current directory")
        return {}, []
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return {}, []

def visualize_detections(detections, image_path='image.png'):
    try:
        image = cv2.imread(image_path)
        for item in detections:
            box = item['box']
            class_name = item['class_name']
            estimated_weight = item['estimated_weight']
            # Draw bounding box
            cv2.rectangle(image, 
                         (int(box[0]), int(box[1])), 
                         (int(box[2]), int(box[3])), 
                         (0, 255, 0), 2)
            # Put label
            label = f"{class_name}: {estimated_weight:.2f}g"
            cv2.putText(image, label, 
                       (int(box[0]), int(box[1])-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 2)
        
        # Display the image
        cv2.imshow('Detected Food Items', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error visualizing detections: {str(e)}")

def main():
    # Detect food items
    result, detections = detect_food()
    
    # Print results
    if result:
        print("\nDetected Food Items and Weights:")
        print(json.dumps(result, indent=4))
        
        # Visualize detections
        visualize_detections(detections)
    else:
        print("No food items detected or error occurred")

if __name__ == "__main__":
    main()