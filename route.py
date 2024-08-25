from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
import torch
from io import BytesIO

# Set the device
device = torch.device("cpu")

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Load SAM2 model
sam2_checkpoint = "sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

app = Flask(__name__)

# Global variable to store the image in memory
the_image = None

# Folder to store uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to process the image and generate a mask
def generate_mask(image, points_clicked):
    print('Mask creation called ')
    print(points_clicked)
    input_point = np.array(points_clicked)
    input_label = np.array([1]*len(points_clicked))
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    mask = masks.squeeze()  # Convert (1, 2524, 2524) to (2524, 2524)

    # Ensure the mask is broadcastable over the three color channels
    mask_3channel = np.stack([mask] * 3, axis=-1)

    # Create a white background (same shape as the image)
    white_background = np.ones_like(image) * 255  # RGB white is [255, 255, 255]

    # Combine the masked image with the white background
    masked_image_with_white_bg = np.where(mask_3channel == 1, image, white_background)

    return masked_image_with_white_bg

# Function to center and pad the image
def center_and_pad_image(image, padding=30):
    # Convert to PIL Image if it's a numpy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))

    # Get the bounding box of the non-white area
    bbox = image.getbbox()
    
    if bbox:
        # Crop the image to the bounding box
        cropped = image.crop(bbox)
        
        # Calculate new size with padding
        new_width = cropped.width + 2 * padding
        new_height = cropped.height + 2 * padding
        
        # Create a new white image with the new size
        centered = Image.new('RGB', (new_width, new_height), (255, 255, 255))
        
        # Paste the cropped image onto the center of the new image
        paste_x = (new_width - cropped.width) // 2
        paste_y = (new_height - cropped.height) // 2
        centered.paste(cropped, (paste_x, paste_y))
        
        # Make the image square by adding white padding
        max_dim = max(centered.width, centered.height)
        square_image = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
        paste_x = (max_dim - centered.width) // 2
        paste_y = (max_dim - centered.height) // 2
        square_image.paste(centered, (paste_x, paste_y))
        
        return square_image
    else:
        # If there's no non-white area, return a square white image
        return Image.new('RGB', (100, 100), (255, 255, 255))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    global the_image
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Load image and store it in memory
    the_image = Image.open(file_path)
    the_image = np.array(the_image.convert("RGB"))
    predictor.set_image(the_image)

    return jsonify({'image_path': file_path})

@app.route('/process', methods=['POST'])
def process_image():
    global the_image
    data = request.get_json()
    points = data['points']
    processed_image = generate_mask(the_image, points)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_image.png')
    Image.fromarray(processed_image.astype('uint8')).save(output_path)
    return jsonify({'processed_image_path': output_path})

@app.route('/finalize', methods=['POST'])
def finalize_image():
    global the_image
    data = request.get_json()
    points = data['points']
    processed_image = generate_mask(the_image, points)
    
    # Center and pad the image
    final_image = center_and_pad_image(processed_image, padding=30)
    
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'final_image.png')
    final_image.save(output_path)
    return jsonify({'final_image_path': output_path})

if __name__ == '__main__':
    app.run(debug=True)