from ultralytics import YOLO
import numpy as np


class YoloPersonDetector:
    PERSON_CLASS_ID = 0

    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect_person(self, frame: np.ndarray) -> tuple[list[list[int]], list[float]]:
        results = self.model(frame, verbose=False)
        
        boxes = []
        confidences = []
        
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id == self.PERSON_CLASS_ID:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    conf = float(box.conf[0])
                    boxes.append(xyxy)
                    confidences.append(conf)
        
        return boxes, confidences

