from ultralytics import YOLO
import torch
import multiprocessing
import matplotlib.pyplot as plt

def train_model(model, device, data_yaml, num_epochs=1, gradient_accumulation_steps=2):
    # Define a list to store loss values during training
    train_losses = []

    # Use the model
    for epoch in range(num_epochs):
        for step in range(gradient_accumulation_steps):
            model.train(data=data_yaml, epochs=100)  # train the model for one epoch on the GPU
        print("validation\n\n\n\n\n")

    model.val()
    # Ensure that data is loaded onto the GPU as well (if not done automatically)
    model.hyp = model.hyp.to(device)

def predict_with_model(model, device, image_path):
    # Load an image for prediction (replace 'image_path' with the actual image path)
    img = model.load(image_path)

    # Perform inference on the image
    results = model.predict(img)

    # Display the results
    results.show()

def main():
    # Check if a GPU is available and set it as the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Load a model with the specified device
    model = YOLO("yolov8n.yaml").to(device)

    # Define the number of gradient accumulation steps
    gradient_accumulation_steps = 1

    # Update the data YAML path to match your dataset structure
    data_yaml = "data.yaml"  # Replace with the path to your data.yaml file

    # Train the model
    train_model(model, device, data_yaml, num_epochs=1, gradient_accumulation_steps=gradient_accumulation_steps)

    # Make predictions on an image (replace 'image_path' with the actual image path)
    predict_with_model(model, device, image_path="datasets/train/images/1_jpg.rf.1637bfe42fd0d0eaf1434d4ea224d54c.jpg")

if __name__ == '__main__':
    # Add freeze_support() to ensure proper initialization on Windows
    multiprocessing.freeze_support()
    main()
