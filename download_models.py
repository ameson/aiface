import os
import requests
from tqdm import tqdm

def download_file(url, filename):
    """下载文件并显示进度条"""
    # 禁用代理设置
    session = requests.Session()
    session.trust_env = False
    
    response = session.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=f"Downloading {filename}",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

def download_models():
    # 创建models目录
    os.makedirs('models', exist_ok=True)
    
    # 模型URLs
    models = {
        'face_detection_yunet_2023mar.onnx': 'https://huggingface.co/spaces/opencv-zoo/face_detection_yunet/resolve/main/face_detection_yunet.onnx',
        'face_recognition_sface_2021dec.onnx': 'https://huggingface.co/spaces/opencv-zoo/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx'
    }
    
    # 下载每个模型
    for model_name, url in models.items():
        model_path = os.path.join('models', model_name)
        if os.path.exists(model_path):
            print(f"{model_name} already exists, skipping...")
            continue
            
        print(f"\nDownloading {model_name}...")
        try:
            download_file(url, model_path)
            print(f"Successfully downloaded {model_name}")
        except Exception as e:
            print(f"Error downloading {model_name}: {str(e)}")

if __name__ == "__main__":
    download_models()
