import os
import sys
import requests
from PIL import Image
import io
import base64
import json

import time

def colorize_image(model, token, image_path, text_prompt):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # Open the image and convert it to JPEG format
    with open(image_path, 'rb') as image_file, io.BytesIO() as buffer:
        img = Image.open(image_file)
        img.save(buffer, format='JPEG')
        buffer.seek(0)
        jpeg_data = buffer.read()

    # Encode the JPEG image as a base64 string
    encoded_string = base64.b64encode(jpeg_data).decode('utf-8')

    # Prepare the data for the request
    data = {
        "inputs": encoded_string,
        "parameters":{
            "prompt":text_prompt
        }
    }

    # Initial attempt to post the request
    response = requests.post(f"https://api-inference.huggingface.co/models/{model}", 
                             headers=headers, 
                             data=json.dumps(data))

    # Check if the model is currently loading
    while response.status_code == 503:  # Service Unavailable
        try:
            response_data = response.json()
            if "estimated_time" in response_data:
                wait_time = response_data["estimated_time"]
                print(f"Model is loading. Waiting for {wait_time} seconds.")
                time.sleep(60)

                # Retry the request
                response = requests.post(f"https://api-inference.huggingface.co/models/{model}", 
                                         headers=headers, 
                                         data=json.dumps(data))
        except ValueError:
            print("Error parsing JSON response for wait time.")
            return None

    # Process the response after waiting
    if response.status_code == 200:
        try:
            return Image.open(io.BytesIO(response.content))
        except IOError:
            try:
                response_data = response.json()
                image_data = base64.b64decode(response_data['image'])
                return Image.open(io.BytesIO(image_data))
            except (KeyError, TypeError):
                print("Error processing the response as an image.")
                return None
    else:
        print(f"Error: {response.text}")
        return None

def process_folder(model, token, text_prompt, input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            file_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # Check if the output file already exists
            if os.path.exists(output_path):
                print(f'Output file {output_path} already exists, skipping.')
                continue
            
            print(f'Processing {filename} with prompt "{text_prompt}"...')
            colorized_image = colorize_image(model, token, file_path, text_prompt)
            if colorized_image:
                colorized_image.save(output_path)
                print(f'Saved colorized image to {output_path}')

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python edit_frames.py <model_name> <hf_token> <text_prompt> <input_folder> <output_folder>")
        sys.exit(1)

    model_name = sys.argv[1]
    hf_token = sys.argv[2]
    text_prompt = sys.argv[3]
    input_folder = sys.argv[4]
    output_folder = sys.argv[5]

    process_folder(model_name, hf_token, text_prompt, input_folder, output_folder)
