import streamlit as st
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import cv2
import numpy as np
from PIL import Image

def main():
    st.title("Inference with Detectron2 Streamlit App")
    
    # File uploader for the input image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png","webp"])
    print(uploaded_file)

    if uploaded_file is not None:
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
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        print(image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Specify the output path for the converted image (in JPG format)
        output_image_path = 'your_output_path.jpg'

        # Save the input image in JPG format
        cv2.imwrite(output_image_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

        # Open the saved image using PIL
        img_rgb = Image.open(output_image_path)

        print(img_rgb.info)


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
                v.draw_binary_mask(mask, color=(0.0, 1.0, 0.0), alpha=0.5)

                # Update the total area for the corresponding class
                mask_areas[class_label] += np.sum(mask)

        # Get the output images
        input_image = v.output.get_image()[:, :, ::-1]
        # output_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Resize images to have the same height
        min_height = min(img_rgb.size[1], input_image.shape[0])
        img_rgb_resized = np.array(img_rgb.resize((int(min_height * img_rgb.size[0] / img_rgb.size[1]), min_height)))
        input_image_resized = cv2.resize(input_image, (int(min_height * input_image.shape[1] / input_image.shape[0]), min_height))

        # Concatenate resized img_rgb and input_image horizontally
        concatenated_image = np.concatenate((img_rgb_resized, input_image_resized), axis=1)

        # Display the result image
        st.image(concatenated_image, caption="Input and Output Images", use_column_width=True)

        
        # Resize images to have the same height
        # min_height = min(input_image.shape[0], img_rgb.shape[0])
        # input_image_resized = cv2.resize(input_image, (int(min_height * input_image.shape[1] / input_image.shape[0]), min_height))
        # output_image_resized = cv2.resize(img_rgb, (int(min_height * img_rgb.shape[1] / img_rgb.shape[0]), min_height))

        # Concatenate images horizontally
        # result_image = np.concatenate((output_image_resized, input_image_resized), axis=1)

        # Display the result image
        # st.image(img_rgb, caption="Input Image", use_column_width=True)
        # st.image(input_image, caption="Output Image", use_column_width=True)
        # result_image = np.concatenate((img_rgb, input_image), axis=1)
        # st.image(result_image, caption="Input Image", use_column_width=True)

        def dpi_mm(uploaded_file,x):
            metadata=Image.open(output_image_path).info
            print(metadata)
            dpi = metadata['jfif_density']
            pixels_per_mm = (dpi[0] / 25.4)*x
            return pixels_per_mm
        bin_mm = round(dpi_mm(uploaded_file,mask_areas['Bin']),2)
        
        
        nbd_mm = round(dpi_mm(uploaded_file,mask_areas['NBD']),2)
        percent = round((nbd_mm / bin_mm) * 100, 0)
        # Check if bin_mm is zero
        if bin_mm == 0:
            bin_mm = " not found"
            percent = " not found as bin area is not found"


        st.write("*" * 40)
        st.write(f"The Bin area is \t\t{bin_mm} mm")
        st.write(f"The NBD area is \t\t{nbd_mm} mm")
        st.write(f"The contamination percent is \t{percent} %") 
        st.write("*" * 40)

if __name__ == "__main__":
    main()