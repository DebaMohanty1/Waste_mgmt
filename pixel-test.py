from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import cv2
import numpy as np
from PIL import Image

def perform_inference(image_path):
    # Path to the saved model weights
    model_weights_path = "model_final.pth"
    yaml_config_path = "config.yaml"

    # Create a new Detectron2 config
    cfg = get_cfg()
    cfg.merge_from_file(yaml_config_path)
    cfg.MODEL.WEIGHTS = model_weights_path
    cfg.MODEL.DEVICE = "cpu"  # Use "cuda" if you have a GPU

    # Manually set class names based on your dataset
    class_names = ["Bin", "NBD"]

    # Create a predictor for inference
    predictor = DefaultPredictor(cfg)
    confidence_threshold = 0.95

    # Read the input image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform inference
    outputs = predictor(image_rgb)

    instances = outputs["instances"]

    # Dictionary to store mask areas for each class
    mask_areas = {"Bin": 0, "NBD": 0}
    for i in range(len(instances)):
        score = instances.scores[i].item()
        if score >= confidence_threshold:
            box = instances.pred_boxes.tensor.cpu().numpy()[i].astype(int)
            label = instances.pred_classes[i].item()
            class_label = class_names[label]
            score = instances.scores[i].item()

            # Get the mask for the current instance
            mask = instances.pred_masks[i].cpu().numpy()

            # Update the total area for the corresponding class
            mask_areas[class_label] += np.sum(mask)

    return mask_areas

def dpi_mm(metadata,x):
    print(metadata)
    dpi = metadata['jfif_density']
    pixels_per_mm = (dpi[0] / 25.4) * x
    return pixels_per_mm

def main(image_path):
    # Perform inference
    mask_areas = perform_inference(image_path)

    # Calculate areas in millimeters
    image_metadata = Image.open(image_path).info
    bin_mm = round(dpi_mm(image_metadata, mask_areas['Bin']), 2)
    nbd_mm = round(dpi_mm(image_metadata, mask_areas['NBD']), 2)

    # Calculate contamination percentage
    contamination_percent = round((nbd_mm / bin_mm) * 100, 0)

    # Print results
    print("*" * 40)
    print(f"The Bin area is \t\t{bin_mm} mm")
    print(f"The NBD area is \t\t{nbd_mm} mm")
    print(f"The contamination percent is \t{contamination_percent} %")
    print("*" * 40)

if __name__ == "__main__":
    # Provide the path to the input image here
    input_image_path = "test_png.PNG"
    main(input_image_path)
