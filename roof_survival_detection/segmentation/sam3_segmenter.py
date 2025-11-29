import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import SamModel, SamProcessor


class SAM3Segmenter:
    def __init__(self, sam_model_path: str = "facebook/sam-vit-base", 
                 grounding_model_path: str = "IDEA-Research/grounding-dino-tiny"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.sam_model = SamModel.from_pretrained(sam_model_path).to(self.device)
        self.sam_processor = SamProcessor.from_pretrained(sam_model_path)
        
        self.grounding_processor = AutoProcessor.from_pretrained(grounding_model_path)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            grounding_model_path
        ).to(self.device)

    def _get_grounding_boxes(self, frame: np.ndarray, text_prompt: str) -> list:
        image = Image.fromarray(frame)
        inputs = self.grounding_processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)
        
        results = self.grounding_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.25,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]]
        )
        
        if len(results) > 0 and len(results[0]["boxes"]) > 0:
            return results[0]["boxes"].cpu().numpy().tolist()
        return []

    def _segment_with_boxes(self, frame: np.ndarray, boxes: list) -> np.ndarray:
        if not boxes:
            return np.zeros(frame.shape[:2], dtype=np.uint8)
        
        image = Image.fromarray(frame)
        input_boxes = [boxes]
        
        inputs = self.sam_processor(
            image, 
            input_boxes=input_boxes, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.sam_model(**inputs)
        
        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )
        
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for mask_set in masks:
            for mask in mask_set:
                mask_np = mask.squeeze().numpy().astype(np.uint8) * 255
                if mask_np.ndim == 3:
                    mask_np = mask_np[0]
                combined_mask = np.maximum(combined_mask, mask_np)
        
        return combined_mask

    def segment_water(self, frame: np.ndarray) -> np.ndarray:
        boxes = self._get_grounding_boxes(frame, "flood water")
        return self._segment_with_boxes(frame, boxes)

    def segment_roof(self, frame: np.ndarray) -> np.ndarray:
        boxes = self._get_grounding_boxes(frame, "roof")
        return self._segment_with_boxes(frame, boxes)

    def extract_person_mask(self, frame: np.ndarray, person_box: list) -> np.ndarray:
        x1, y1, x2, y2 = person_box
        
        image = Image.fromarray(frame)
        input_boxes = [[[x1, y1, x2, y2]]]
        
        inputs = self.sam_processor(
            image, 
            input_boxes=input_boxes, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.sam_model(**inputs)
        
        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )
        
        if masks and len(masks[0]) > 0:
            mask = masks[0][0].squeeze().numpy()
            if mask.ndim == 3:
                mask = mask[0]
            return (mask * 255).astype(np.uint8)
        
        return np.zeros(frame.shape[:2], dtype=np.uint8)

