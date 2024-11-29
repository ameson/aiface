import cv2
import numpy as np
import logging
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import time
from typing import List, Dict, Any
import json
from sklearn.preprocessing import StandardScaler
import joblib

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 配置静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化人脸检测器
face_detector = cv2.FaceDetectorYN.create(
    "models/face_detection_yunet_2023mar.onnx",
    "",
    (320, 320),
    0.9,  # score threshold
    0.3,  # nms threshold
    5000  # top_k
)

# 初始化人脸关键点检测器
face_recognizer = cv2.FaceRecognizerSF.create(
    "models/face_recognition_sface_2021dec.onnx",
    ""
)

def get_face_features(face_info, img):
    """从面部区域提取关键特征，基于专业美学标准"""
    x, y, w, h = map(int, face_info[0:4])
    
    # 确保坐标不越界
    x = max(0, x)
    y = max(0, y)
    w = min(w, img.shape[1] - x)
    h = min(h, img.shape[0] - y)
    
    # 提取人脸区域
    face_roi = img[y:y+h, x:x+w]
    if face_roi.size == 0:
        return None
        
    try:
        # 获取人脸关键点
        landmarks = face_info[4:len(face_info)-1].reshape(-1, 2)
        
        # 提取关键特征点组
        # 眼睛关键点
        left_eye = landmarks[0:2]  # 左眼角点
        right_eye = landmarks[2:4]  # 右眼角点
        
        # 鼻子关键点
        nose = landmarks[4:6]
        
        # 嘴巴关键点
        mouth_left = landmarks[6:8]
        mouth_right = landmarks[8:10]
        
        # 下巴关键点（使用嘴巴底部点估计）
        chin = landmarks[10:12]
        
        # 1. 计算面部对称性
        # 使用关键点的水平对称性
        left_points = np.vstack([left_eye, mouth_left])
        right_points = np.vstack([right_eye, mouth_right])
        symmetry_score = calculate_symmetry_from_points(left_points, right_points)
        
        # 2. 计算黄金比例
        # 眼睛间距与脸宽的比例
        eye_distance = np.linalg.norm(np.mean(left_eye, axis=0) - np.mean(right_eye, axis=0))
        face_width = w
        eye_ratio = eye_distance / face_width
        golden_ratio_score = 1 - abs(eye_ratio - 0.46)  # 0.46是理想的眼睛间距比例
        
        # 3. 计算面部轮廓评分
        # 使用下巴点和脸部轮廓评估
        contour_score = calculate_contour_score(chin, w, h)
        
        # 4. 计算五官精致度
        feature_details = {
            'eyes': calculate_feature_details_from_points(np.vstack([left_eye, right_eye])),
            'nose': calculate_feature_details_from_points(nose),
            'mouth': calculate_feature_details_from_points(np.vstack([mouth_left, mouth_right]))
        }
        
        # 5. 计算五官比例
        proportions = calculate_proportions(landmarks, w, h)
        
        # 提取面部特征向量
        face_feature = face_recognizer.feature(img)
        
        return {
            'symmetry_score': symmetry_score,
            'golden_ratio_score': golden_ratio_score,
            'contour_score': contour_score,
            'feature_details': feature_details,
            'proportions': proportions,
            'face_feature': face_feature.flatten()
        }
        
    except Exception as e:
        logger.error(f"Error in face feature extraction: {str(e)}")
        return None

def calculate_symmetry_from_points(left_points, right_points):
    """计算面部对称性"""
    try:
        # 将右侧点映射到左侧
        right_points_flipped = right_points.copy()
        right_points_flipped[:, 0] = -right_points_flipped[:, 0]
        
        # 定义不同部位的权重
        weights = np.array([
            1.5,  # 眼睛
            1.5,  # 眼睛
            1.2,  # 眉毛
            1.2,  # 眉毛
            1.3,  # 嘴巴
            1.0,  # 其他点
        ])
        
        # 计算加权距离
        distances = np.linalg.norm(left_points - right_points_flipped, axis=1)
        weighted_distances = distances * weights
        
        # 归一化距离并转换为对称性分数
        symmetry = 1 - np.average(weighted_distances) / np.max(weighted_distances)
        # 提高基准分数
        symmetry = 0.7 + (symmetry - 0.5) * 0.6
        return max(0, min(1, symmetry))
    except Exception as e:
        logger.error(f"Symmetry calculation error: {str(e)}")
        return 0.7

def calculate_contour_score(chin_point, face_width, face_height):
    """计算面部轮廓评分"""
    try:
        # 理想下巴位置（黄金分割比例）
        ideal_chin_y = face_height * 0.618
        chin_y = chin_point[1]
        
        # 计算与理想位置的偏差
        deviation = abs(chin_y - ideal_chin_y) / face_height
        
        # 脸型比例评分（宽高比）
        ideal_ratio = 1.618  # 黄金分割比
        actual_ratio = face_height / face_width
        ratio_score = 1 - abs(actual_ratio - ideal_ratio) / ideal_ratio
        
        # 综合评分
        score = (ratio_score * 0.6 + (1 - deviation) * 0.4)
        # 提高基准分数
        score = 0.7 + (score - 0.5) * 0.6
        return max(0, min(1, score))
    except Exception as e:
        logger.error(f"Contour calculation error: {str(e)}")
        return 0.7

def calculate_feature_details_from_points(points):
    """计算特征精细度"""
    try:
        # 计算特征点之间的距离
        dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
        
        # 计算平滑度（使用移动平均）
        window = 3
        smoothness = np.convolve(dists, np.ones(window)/window, mode='valid')
        variation = np.std(smoothness) / np.mean(smoothness)
        
        # 转换为分数，使用sigmoid函数使得分数分布更合理
        score = 1 / (1 + np.exp(-2 * (1 - variation)))
        # 提高基准分数
        score = 0.7 + (score - 0.5) * 0.6
        return max(0, min(1, score))
    except Exception as e:
        logger.error(f"Feature details calculation error: {str(e)}")
        return 0.7

def calculate_proportions(landmarks, face_width, face_height):
    """计算面部比例"""
    try:
        # 计算关键特征之间的比例
        eye_distance = np.linalg.norm(landmarks[0] - landmarks[2])
        nose_to_chin = np.linalg.norm(landmarks[4] - landmarks[10])
        mouth_width = np.linalg.norm(landmarks[6] - landmarks[8])
        
        # 调整后的理想比例（基于美学研究）
        ideal_eye_ratio = 0.45    # 眼距/脸宽
        ideal_nose_ratio = 0.36   # 鼻子到下巴/脸高
        ideal_mouth_ratio = 0.42  # 嘴宽/脸宽
        
        # 计算实际比例
        eye_ratio = eye_distance / face_width
        nose_ratio = nose_to_chin / face_height
        mouth_ratio = mouth_width / face_width
        
        # 计算与理想比例的接近程度，使用高斯函数使得分数分布更合理
        def gaussian_score(actual, ideal, sigma=0.1):
            return np.exp(-((actual - ideal) ** 2) / (2 * sigma ** 2))
        
        eye_score = gaussian_score(eye_ratio, ideal_eye_ratio)
        nose_score = gaussian_score(nose_ratio, ideal_nose_ratio)
        mouth_score = gaussian_score(mouth_ratio, ideal_mouth_ratio)
        
        # 加权平均
        weights = [0.4, 0.3, 0.3]  # 眼睛权重更高
        score = (eye_score * weights[0] + nose_score * weights[1] + mouth_score * weights[2])
        # 提高基准分数
        score = 0.7 + (score - 0.5) * 0.6
        return max(0, min(1, score))
    except Exception as e:
        logger.error(f"Proportions calculation error: {str(e)}")
        return 0.7

def calculate_beauty_score(features):
    """计算综合美丽分数"""
    if features is None:
        return 75  # 提高默认分数
        
    try:
        # 调整各维度权重
        weights = {
            'symmetry_score': 0.2,      # 降低对称性权重
            'golden_ratio_score': 0.2,  # 增加黄金比例权重
            'contour_score': 0.2,       # 增加轮廓分数权重
            'feature_details': 0.2,     # 保持特征细节权重
            'proportions': 0.2          # 保持比例权重
        }
        
        # 计算特征详情得分，添加基准分数
        feature_details_score = np.mean([
            features['feature_details']['eyes'],
            features['feature_details']['nose'],
            features['feature_details']['mouth']
        ])
        
        # 综合评分，为每个维度添加基准分数
        base_score = 70  # 基准分数
        score = base_score + (
            weights['symmetry_score'] * (features['symmetry_score'] - 0.5) * 60 +
            weights['golden_ratio_score'] * (features['golden_ratio_score'] - 0.5) * 60 +
            weights['contour_score'] * (features['contour_score'] - 0.5) * 60 +
            weights['feature_details'] * (feature_details_score - 0.5) * 60 +
            weights['proportions'] * (features['proportions'] - 0.5) * 60
        )
        
        # 调整分数范围
        score = max(60, min(98, score))  # 提高最低分和最高分
        return score
    except Exception as e:
        logger.error(f"评分计算错误: {str(e)}")
        return 75  # 提高错误情况下的默认分数

def process_image(image_bytes: bytes) -> Dict[str, Any]:
    """处理图像并返回颜值评分"""
    try:
        # 将图像字节转换为numpy数组
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("无法解码图像")
        
        # 设置输入图像大小
        height, width, _ = img.shape
        face_detector.setInputSize((width, height))
        
        # 检测人脸
        _, faces = face_detector.detect(img)
        
        if faces is None or len(faces) == 0:
            return {"error": "未检测到人脸"}
        
        # 获取置信度最高的人脸
        face = faces[0]
        
        # 提取特征
        features = get_face_features(face, img)
        if features is None:
            return {"error": "无法提取面部特征"}
            
        # 计算美丽分数
        beauty_score = calculate_beauty_score(features)
        
        # 生成评分维度
        dimension_scores = {
            "symmetry": {
                "name": "面部对称性",
                "score": round(float(features['symmetry_score'] * 100), 1),
                "weight": 25
            },
            "proportion": {
                "name": "五官比例",
                "score": round(float(features['proportions'] * 100), 1),
                "weight": 20
            },
            "details": {
                "name": "五官精致度",
                "score": round(float(np.mean([
                    features['feature_details']['eyes'],
                    features['feature_details']['nose'],
                    features['feature_details']['mouth']
                ]) * 100), 1),
                "weight": 25
            },
            "contour": {
                "name": "轮廓协调",
                "score": round(float(features['contour_score'] * 100), 1),
                "weight": 15
            },
            "harmony": {
                "name": "整体和谐度",
                "score": round(float(features['golden_ratio_score'] * 100), 1),
                "weight": 15
            }
        }
        
        # 构建返回结果
        x, y, w, h = map(int, face[0:4])
        results = [{
            "score": round(float(beauty_score), 1),
            "bbox": [
                int(x / img.shape[1] * 100),
                int(y / img.shape[0] * 100),
                int(w / img.shape[1] * 100),
                int(h / img.shape[0] * 100)
            ],
            "confidence": float(face[14]),
            "dimensions": dimension_scores
        }]
        
        logger.info(f"处理结果: {results}")
        return {"results": results}
        
    except Exception as e:
        logger.error(f"处理请求时出错: {str(e)}")
        return {"error": str(e)}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """分析上传的图像并返回颜值评分"""
    try:
        start_time = time.time()
        image_bytes = await file.read()
        
        # 记录接收到的文件
        logger.info(f"接收到文件: {file.filename}")
        
        # 处理图像
        result = process_image(image_bytes)
        
        # 记录处理时间
        process_time = time.time() - start_time
        logger.info(f"处理时间: {process_time:.2f}秒")
        
        return result
    except Exception as e:
        logger.error(f"处理请求时出错: {str(e)}")
        return {"error": str(e)}

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")
