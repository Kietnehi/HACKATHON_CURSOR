import numpy as np


def compute_danger_score(
    person_mask: np.ndarray,
    water_mask: np.ndarray,
    roof_mask: np.ndarray,
    bbox: list
) -> tuple[int, str]:
    score = 0
    x1, y1, x2, y2 = bbox
    bbox_height = y2 - y1
    
    person_binary = person_mask > 127
    water_binary = water_mask > 127
    roof_binary = roof_mask > 127
    
    water_overlap = np.logical_and(person_binary, water_binary)
    if np.any(water_overlap):
        score += 60
    
    roof_edge = _get_edge_region(roof_binary, edge_width=10)
    roof_edge_touch = np.logical_and(person_binary, roof_edge)
    if np.any(roof_edge_touch):
        score += 20
    
    person_region = person_binary[y1:y2, x1:x2] if y2 <= person_binary.shape[0] and x2 <= person_binary.shape[1] else person_binary
    water_region = water_binary[y1:y2, x1:x2] if y2 <= water_binary.shape[0] and x2 <= water_binary.shape[1] else water_binary
    
    water_in_bbox = np.logical_and(person_region.shape == water_region.shape, True)
    if water_in_bbox:
        try:
            water_rows = np.any(water_region, axis=1)
            water_height = np.sum(water_rows)
            if bbox_height > 0 and water_height / bbox_height > 0.4:
                score += 20
        except:
            pass
    
    score = max(0, min(100, score))
    
    if score <= 30:
        danger_level = "safe"
    elif score <= 60:
        danger_level = "warning"
    elif score <= 80:
        danger_level = "high"
    else:
        danger_level = "critical"
    
    return score, danger_level


def _get_edge_region(mask: np.ndarray, edge_width: int = 10) -> np.ndarray:
    from scipy import ndimage
    
    eroded = ndimage.binary_erosion(mask, iterations=edge_width)
    edge = np.logical_and(mask, ~eroded)
    return edge

