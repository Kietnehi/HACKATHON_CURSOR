import cv2
import os
import time
from datetime import datetime

from detection.yolo_person_detector import YoloPersonDetector
from segmentation.sam3_segmenter import SAM3Segmenter
from danger.danger_analysis import compute_danger_score
from notify.sos_notifier import send_sos
from utils.drawing import draw_person_boxes, overlay_mask, draw_danger_score
from utils.video_stream import VideoStream


OUTPUT_DIR = "alerts"
YOLO_MODEL = "yolov8n.pt"
VIDEO_SOURCE = 0


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading YOLO person detector...")
    yolo = YoloPersonDetector(model_path=YOLO_MODEL)
    
    print("Loading SAM3 segmenter...")
    sam3 = SAM3Segmenter()
    
    print("Opening video stream...")
    stream = VideoStream(source=VIDEO_SOURCE)
    
    last_sos_time = 0
    sos_cooldown = 30
    
    try:
        for frame in stream.frames():
            display_frame = frame.copy()
            
            person_boxes, confidences = yolo.detect_person(frame)
            
            if person_boxes:
                water_mask = sam3.segment_water(frame)
                roof_mask = sam3.segment_roof(frame)
                
                display_frame = overlay_mask(display_frame, water_mask, color=(255, 100, 0), alpha=0.3)
                display_frame = overlay_mask(display_frame, roof_mask, color=(0, 100, 255), alpha=0.3)
                
                max_danger_score = 0
                max_danger_level = "safe"
                
                for person_box in person_boxes:
                    person_mask = sam3.extract_person_mask(frame, person_box)
                    
                    danger_score, danger_level = compute_danger_score(
                        person_mask, water_mask, roof_mask, person_box
                    )
                    
                    if danger_score > max_danger_score:
                        max_danger_score = danger_score
                        max_danger_level = danger_level
                    
                    if danger_level == "critical":
                        current_time = time.time()
                        if current_time - last_sos_time > sos_cooldown:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            frame_path = os.path.join(OUTPUT_DIR, f"alert_{timestamp}.jpg")
                            cv2.imwrite(frame_path, frame)
                            
                            send_sos(
                                "CRITICAL: Person detected in flood on roof!",
                                danger_score,
                                frame_path
                            )
                            last_sos_time = current_time
                
                display_frame = draw_person_boxes(display_frame, person_boxes, confidences)
                display_frame = draw_danger_score(display_frame, max_danger_score, max_danger_level)
            else:
                display_frame = draw_danger_score(display_frame, 0, "safe")
            
            cv2.imshow("Roof Survival Detection", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"snapshot_{timestamp}.jpg"), display_frame)
                print(f"Snapshot saved")
    
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

