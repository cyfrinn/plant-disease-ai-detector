import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

# Load the trained model
model = load_model(r"path_to_model")

# Class labels
class_labels = [
    "Pepper_bacterial_spot", "Pepper_healthy", "Potato_Early_blight", "Potato_Late_blight",
    "Potato_healthy", "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites", "Tomato_Target_Spot",
    "Tomato_Tomato_YellowLeaf_Curl_Virus", "Tomato_Tomato_mosaic_virus", "Tomato_healthy"
]

# Create the main window
root = tk.Tk()
root.title("Plant Bacterial Disease AI Detector")
root.geometry("600x500")

# Function to open the image file
def upload_image():
    global img_path
    img_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    
    if img_path:
        # Display the selected image
        img = Image.open(img_path)
        img.thumbnail((200, 200))
        img = ImageTk.PhotoImage(img)
        
        label_img.config(image=img)
        label_img.image = img

# Function to predict the disease
def predict_disease():
    if not img_path:
        messagebox.showerror("Error", "Please upload an image first!")
        return

    # Preprocess the image
    img_height, img_width = 128, 128
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]
    
    # Display the result
    result_label.config(text=f"Predicted Class: {predicted_class}")
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_class}")
    plt.show()

# UI elements
label_title = tk.Label(root, text="Plant Disease AI Detector", font=("Arial", 18))
label_title.pack(pady=20)

label_img = tk.Label(root)
label_img.pack(pady=20)

upload_btn = tk.Button(root, text="Upload Image", command=upload_image, font=("Arial", 14))
upload_btn.pack(pady=10)

predict_btn = tk.Button(root, text="Predict Disease", command=predict_disease, font=("Arial", 14))
predict_btn.pack(pady=10)

result_label = tk.Label(root, text="Prediction will appear here.", font=("Arial", 14), wraplength=500)
result_label.pack(pady=20)

# Initialize the image path as None
img_path = None

# Start the GUI event loop
root.mainloop()
