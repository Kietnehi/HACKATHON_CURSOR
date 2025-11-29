import cv2
import numpy as np


def draw_person_boxes(
    frame: np.ndarray,
    boxes: list,
    confidences: list,
    color: tuple = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    output = frame.copy()
    
    for box, conf in zip(boxes, confidences):
        x1, y1, x2, y2 = box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        label = f"Person: {conf:.2f}"
        cv2.putText(output, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return output


def overlay_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple = (255, 0, 0),
    alpha: float = 0.4
) -> np.ndarray:
    output = frame.copy()
    
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    
    mask_binary = mask > 127
    overlay = output.copy()
    overlay[mask_binary] = color
    
    output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)
    return output


def draw_danger_score(
    frame: np.ndarray,
    score: int,
    danger_level: str,
    position: tuple = (10, 30)
) -> np.ndarray:
    output = frame.copy()
    
    color_map = {
        "safe": (0, 255, 0),
        "warning": (0, 255, 255),
        "high": (0, 165, 255),
        "critical": (0, 0, 255)
    }
    color = color_map.get(danger_level, (255, 255, 255))
    
    text = f"Danger: {score}/100 [{danger_level.upper()}]"
    cv2.putText(output, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    bar_x, bar_y = position[0], position[1] + 20
    bar_width = 200
    bar_height = 15
    
    cv2.rectangle(output, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
    fill_width = int(bar_width * score / 100)
    cv2.rectangle(output, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
    cv2.rectangle(output, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 1)
    
    return output

