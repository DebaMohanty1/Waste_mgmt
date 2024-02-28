from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import cv2
import numpy as np

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
class_colors = {
    0: (0.0, 1.0, 0.0),  # Green for class "Bin"
    1: (1.0, 0.0, 0.0),  # Red for class "NBD"
}
class_explanations = {
    "Bin": "Top view of the trash bin",
    "NBD": "Non - biodegradable wastes"
}

# Create a predictor for inference
predictor = DefaultPredictor(cfg)
confidence_threshold = 0.95

# Read the input image
image_path = "Client_test.jpeg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform inference
outputs = predictor(image_rgb)

# Visualize the predictions with labels and masks
v = Visualizer(image_rgb, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.5)

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

        # Draw bounding box and label on the image
        v.draw_box(box, edge_color="g", line_style="-")
        v.draw_text(f"{class_label} {score:.2f}", box[:2], font_size=12)

        # Visualize only the masks that meet the confidence threshold
        mask_color = class_colors.get(label, (0.0, 1.0, 0.0))
        v.draw_binary_mask(mask, color=mask_color, alpha=0.5)

        # Update the total area for the corresponding class
        mask_areas[class_label] += np.sum(mask)

# Get the output images
input_image = v.output.get_image()[:, :, ::-1]
output_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

# Resize images to have the same height
min_height = min(input_image.shape[0], output_image.shape[0])
input_image_resized = cv2.resize(input_image, (int(min_height * input_image.shape[1] / input_image.shape[0]), min_height))
output_image_resized = cv2.resize(output_image, (int(min_height * output_image.shape[1] / output_image.shape[0]), min_height))

# Concatenate images horizontally
result_image = np.concatenate((input_image_resized, output_image_resized), axis=1)

print("*"*40+f"\nThe Bin area is \t\t{mask_areas['Bin']}\nThe NBD area is \t\t{mask_areas['NBD']}\nThe contamination percent is \t{round((mask_areas['NBD'] / mask_areas['Bin']) * 100,0)} %\n"+"*"*40 )

# Display the result image
cv2.imshow("Input and Output", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()