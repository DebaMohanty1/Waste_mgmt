from flask import Flask, render_template, request
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import base64

app = Flask(__name__)

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

def process_image(input_image):
    # Perform inference
    outputs = predictor(input_image)

    # Visualize the predictions with labels and masks
    v = Visualizer(input_image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.5)
    instances = outputs["instances"]

    # Filter instances with confidence scores above 95%
    filtered_instances = instances[instances.scores >= confidence_threshold]

    # Draw visualizations on the input image
    v = v.draw_instance_predictions(filtered_instances.to("cpu"))
    output_image = v.get_image()[:, :, ::-1]  # Convert BGR to RGB

    # Resize output_image to match the height of input_image
    output_image_resized = cv2.resize(output_image, (input_image.shape[1], input_image.shape[0]))

    # Concatenate input and output images horizontally
    concatenated_image = np.concatenate((input_image, output_image_resized), axis=1)

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
    image_width_mm = input_image.shape[1] / pixels_per_mm
    image_height_mm = input_image.shape[0] / pixels_per_mm

    # Calculate DPI and areas in millimeters
    bin_mm = round(mask_areas['Bin'] / pixels_per_mm, 2)
    nbd_mm = round(mask_areas['NBD'] / pixels_per_mm, 2)

    # Calculate contamination percentage
    contamination_percent = round((nbd_mm / bin_mm) * 100, 2)

    return concatenated_image, bin_mm, nbd_mm, contamination_percent

@app.route('/', methods=['GET', 'POST'])
def index():
    input_image = None
    concatenated_image_base64 = None
    bin_mm = None
    nbd_mm = None
    contamination_percent = None

    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        
        file = request.files['file']
        
        # If user does not select file, browser also submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        
        if file:
            # Read the input image
            image_data = file.read()
            nparr = np.frombuffer(image_data, np.uint8)
            input_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Process the input image
            concatenated_image, bin_mm, nbd_mm, contamination_percent = process_image(input_image)

            # Encode the concatenated image to base64 for displaying in HTML
            _, img_encoded = cv2.imencode('.png', concatenated_image)
            concatenated_image_base64 = base64.b64encode(img_encoded).decode('utf-8')

    return render_template('index.html', input_image=input_image, concatenated_image=concatenated_image_base64,
                           bin_mm=bin_mm, nbd_mm=nbd_mm, contamination_percent=contamination_percent)



if __name__ == '__main__':
    app.run(debug=True)
