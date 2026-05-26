# Waste Contamination Detection System

This project provides a waste bin contamination detection system using deep learning (Detectron2) and offers both web (Flask) and interactive (Streamlit) interfaces for image-based inference.

## Features
- Detects and segments waste bins and non-biodegradable (NBD) waste in images.
- Calculates contamination percentage based on detected areas.
- Provides both a Flask web app and Streamlit app for user interaction.
- Visualizes results with bounding boxes and masks.

## Project Structure
- `waste_detection_app.py`, `spp.py`: Streamlit apps for interactive inference.
- `flask-final.py`: Flask web application for uploading images and viewing results.
- `test.py`, `imgtest.py`, `pixel-test.py`: Scripts for testing and experimentation.
- `detectron2/`: (Empty or for custom Detectron2 code/models)
- `static/`: Static files for the web app (JS, CSS).
- `templates/`: HTML templates for the Flask app.
- `uploads/`: Stores uploaded images.
- `config.yaml`: Detectron2 model configuration.
- `metrics.json`: Training metrics and logs.
- `model_final.pth`: (Not included) Trained model weights (required for inference).

## Requirements
- Python 3.7+
- Detectron2
- Flask
- Streamlit
- OpenCV
- Pillow
- NumPy

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Streamlit App
Run the Streamlit app for interactive inference:
```bash
streamlit run waste_detection_app.py
```
Or
```bash
streamlit run spp.py
```

### Flask Web App
Run the Flask web server:
```bash
python flask-final.py
```
Then open your browser at `http://localhost:5000`.

### Testing
Run the test script:
```bash
python test.py
```

## How It Works
- Upload an image of a waste bin.
- The model detects and segments the bin and NBD waste.
- The app calculates the area of contamination and displays the result.
- If contamination percentage > 5%, the bin is considered contaminated.

## Notes
- You must provide a trained Detectron2 model weights file as `model_final.pth` in the project directory.
- Update `config.yaml` as needed for your dataset.

## Example Output
- Input image with detected regions and contamination status.

---

**Authors:**
- Your Name Here

**License:**
- Specify your license here.
