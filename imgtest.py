import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

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
confidence_threshold = 0.90

# Path to the input image
input_image_path = 'test_png.PNG'

# Read the input image
image = cv2.imread(input_image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform inference
outputs = predictor(image_rgb)

# Visualize the predictions with labels and masks
v = Visualizer(image_rgb, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.5)
instances = outputs["instances"]

# Filter instances with confidence scores above 95%
filtered_instances = instances[instances.scores >= confidence_threshold]

# Draw visualizations on the input image
v = v.draw_instance_predictions(filtered_instances.to("cpu"))
output_image = v.get_image()[:, :, ::-1]  # Convert BGR to RGB

# Resize output_image to match the height of image_rgb
output_image_resized = cv2.resize(output_image, (image_rgb.shape[1], image_rgb.shape[0]))

# Concatenate input and output images horizontally
concatenated_image = np.concatenate((image_rgb, output_image_resized), axis=1)

# Define the maximum width or height for resizing
max_width = 800
max_height = 600

# Calculate the scale factor based on the maximum width or height
scale_factor = min(max_width / concatenated_image.shape[1], max_height / concatenated_image.shape[0])

# Resize the concatenated image
concatenated_image_resized = cv2.resize(concatenated_image, None, fx=scale_factor, fy=scale_factor)

# Display the concatenated image
cv2.imshow('Input and Output', concatenated_image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Dictionary to store mask areas for each class
mask_areas = {"Bin": 0, "NBD": 0}
for i in range(len(filtered_instances)):
    score = filtered_instances.scores[i].item()
    box = filtered_instances.pred_boxes.tensor.cpu().numpy()[i].astype(int)
    label = filtered_instances.pred_classes[i].item()
    class_label = class_names[label]

    # Get the mask for the current instance
    mask = filtered_instances.pred_masks[i].cpu().numpy()

    # Update the total area for the corresponding class
    mask_areas[class_label] += np.sum(mask)

# Calculate image dimensions in millimeters
dpi = 72  # default DPI value for OpenCV
pixels_per_mm = dpi / 25.4
image_width_mm = image.shape[1] / pixels_per_mm
image_height_mm = image.shape[0] / pixels_per_mm

# Calculate DPI and areas in millimeters
bin_mm = round(mask_areas['Bin'] / pixels_per_mm, 2)
nbd_mm = round(mask_areas['NBD'] / pixels_per_mm, 2)

# Calculate contamination percentage
contamination_percent = round((nbd_mm / bin_mm) * 100, 2)

print("Bin Area (mm^2):", bin_mm)
print("NBD Area (mm^2):", nbd_mm)
print("Contamination Percentage:", contamination_percent)
