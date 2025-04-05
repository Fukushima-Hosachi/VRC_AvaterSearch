#これを使用すると推論ができます

from ultralytics import YOLO

# コンフィグは学習時と同じコンフィグでよい
model = YOLO('./VRCAvaterSearcher/Lib/site-packages/ultralytics/cfg/models/v8/yolov8.yaml')  # build a new model from scratch

# bestのモデルを使う
model = YOLO("./runs/detect/train3/weights/best.pt")  # load a pretrained model (recommended for training)

results = model("./VRCAvaterSearcher/datasets/images/train/",save=True)  # predict on an image
