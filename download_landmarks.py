import os
import bz2
import requests
from tqdm import tqdm

def download_landmarks_model():
    # 创建models目录（如果不存在）
    os.makedirs('models', exist_ok=True)
    
    # 模型文件路径
    model_path = 'models/shape_predictor_68_face_landmarks.dat'
    
    # 如果模型已存在，跳过下载
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}")
        return
    
    # 下载地址
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    
    print("Downloading dlib facial landmarks model...")
    
    # 下载压缩文件
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # 下载进度条
    with open(model_path + '.bz2', 'wb') as file, tqdm(
        desc="Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)
    
    print("Extracting model file...")
    
    # 解压文件
    with open(model_path, 'wb') as new_file:
        with open(model_path + '.bz2', 'rb') as file:
            decompressor = bz2.BZ2Decompressor()
            for data in iter(lambda: file.read(100 * 1024), b''):
                new_file.write(decompressor.decompress(data))
    
    # 删除压缩文件
    os.remove(model_path + '.bz2')
    
    print(f"Model successfully downloaded and extracted to {model_path}")

if __name__ == "__main__":
    download_landmarks_model()
