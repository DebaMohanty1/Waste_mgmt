from flask import Flask, render_template, request, send_from_directory
import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

# Define paths for uploaded images and output images
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Function to perform inference and save the output image
def perform_inference(input_image_path, output_image_path):
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

     # Open the input image file and read its contents
    with open(input_image_path, 'rb') as f:
        image_data = f.read()

    # Decode the image data using OpenCV
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), 1)
    # Read the input image
    # image = cv2.imdecode(np.fromstring(input_image_path.read(), np.uint8), 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image_rgb = image
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

            # Update the total area for the corresponding class
            mask_areas[class_label] += np.sum(mask)

    # Calculate DPI and areas in millimeters
    dpi = Image.open(input_image_path).info['jfif_density']
    pixels_per_mm = (dpi[0] / 25.4)
    bin_mm = round(mask_areas['Bin'] / pixels_per_mm, 2)
    nbd_mm = round(mask_areas['NBD'] / pixels_per_mm, 2)

    # Calculate contamination percentage
    contamination_percent = round((nbd_mm / bin_mm) * 100, 2)

    # Draw visualizations on the input image
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Save the output image
    output_image = v.get_image()[:, :, ::-1]
    cv2.imwrite(output_image_path, output_image)

    return bin_mm, nbd_mm, contamination_percent

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        
        if file:
            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Perform inference and save the output image
            output_filename = 'output.jpg'
            output_image_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            bin_area, nbd_area, contamination_percent = perform_inference(file_path, output_image_path)

            # Pass base filenames and inference results to the template
            return render_template('result.html', input_image=file.filename, output_image=output_filename,
                                   bin_area=bin_area, nbd_area=nbd_area, contamination_percent=contamination_percent)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
