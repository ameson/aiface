import requests
import os

def download_file(url, local_path):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Download the file
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Save the file
    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

if __name__ == "__main__":
    url = "https://github.com/opencv/opencv_zoo/releases/download/0.1.0/face_detection_yunet_2023mar.onnx"
    local_path = "models/face_detection_yunet_2023mar.onnx"
    
    print(f"Downloading model to {local_path}...")
    download_file(url, local_path)
    print("Download completed!")
