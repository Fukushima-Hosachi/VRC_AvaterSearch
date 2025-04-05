import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'   #エラー対策

from ultralytics import YOLO
# Create a new YOLO model from scratch
model = YOLO('C:/Users/PC/Desktop/VRC_AvaterSearch/VRCAvaterSearcher/Lib/site-packages/ultralytics/cfg/models/v8/yolov8.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('C:/Users/PC/Desktop/VRC_AvaterSearch/yolov8n.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='VRCAvaterSearcher/datasets/coco8.yaml', epochs=30,batch=4)

# Evaluate the model's performance on the validation set
results = model.val()
