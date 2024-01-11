from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO("C:/Users/abida/Desktop/garbage_class_detection/best.pt")

# Define path to the image file
source = 'C:/Users/abida/Desktop/61RqbJrQy7L._AC_UF894,1000_QL80_.jpg'

# Specify the result location path
save_path = 'C:/Users/abida/Desktop/results'

# Run inference on the source
results = model.predict(source, save=True, imgsz=320, conf=0.5, save_dir=save_path)
