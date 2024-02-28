from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import cv2
import streamlit as st
from PIL import Image
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
confidence_threshold = 0.67

# Streamlit UI
st.title("Waste Contamination Detection")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image from the file uploader
    image = Image.open(uploaded_file)
    # image = cv2.imread(r"E:\docs jai\1-waste_management\Code\Datasets\version1\val\test.jpg")
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Perform inference
    outputs = predictor(image_np)

    # Visualize the predictions with labels
    v = Visualizer(image_np[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.5)
    
    instances = outputs["instances"]

    # Dictionary to store mask areas for each class
    mask_areas = {"Bin": 0, "NBD": 0}
    i = 0
    for i in range(len(instances)):
        score = instances.scores[i].item()
        if score >= confidence_threshold:
            box = instances.pred_boxes.tensor.cpu().numpy()[i].astype(int)
            label = instances.pred_classes[i].item()
            class_label = class_names[label]
            score = instances.scores[i].item()
            
            # Get the mask for the current instance
            mask = instances.pred_masks[i].cpu().numpy()
            mask_area = np.sum(mask)  # Calculate the area of the mask

            # Update the total area for the corresponding class
            mask_areas[class_label] += mask_area

            # Draw bounding box and label on the image
            v.draw_box(instances.pred_boxes.tensor.cpu().numpy()[i].astype(int), edge_color="g", line_style="-")
            v.draw_text(f"{class_label} {score:.2f}", box[:2], font_size=12)

            # Visualize only the masks that meet the confidence threshold
            mask_color = class_colors.get(label, (0.0, 1.0, 0.0))
            if len(mask.shape) == 2:
               v.draw_binary_mask(mask, color=mask_color, alpha=0.5)

    # Calculate the percentage of NBD area relative to Bin area
    bin_area = mask_areas["Bin"]
    nbd_area = mask_areas["NBD"]
    contamination_percentage = (nbd_area / bin_area) * 100

    # Display the input image on the left side
    st.image(image, caption="Input Image", use_column_width=True, channels="RGB", width=100)

    st.image(v.output.get_image()[:, :, ::-1], caption="Result", use_column_width=True, channels="RGB",width=100)
    
    
    # Display the total areas and contamination status
    st.write(f"Total Bin Area: {bin_area}")
    st.write(f"Total NBD Area: {nbd_area}")
    st.write(f"Contaminated percentage: {contamination_percentage}")
    
    if contamination_percentage > 5:
      st.write("Contaminated Bin")
    else:
      st.write("Non-contaminated Bin")
    
    st.write("Class Explanations:")
    for class_name, explanation in class_explanations.items():
        st.write(f"- {class_name}: {explanation}")

# streamlit run "E:\docs jai\1-waste_management\Code\final-detectron2 code\waste_detection_app.py"