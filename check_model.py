import os

MODEL_PATH = "model_final.pth"
GOOGLE_DRIVE_LINK = "https://drive.google.com/drive/folders/1UojE6f6HHQ86VHqSxafFonFsXV0IGugK"

if not os.path.exists(MODEL_PATH):
    print("[ERROR] Required model weights file 'model_final.pth' not found.")
    print(f"Please download it from: {GOOGLE_DRIVE_LINK}")
    exit(1)
