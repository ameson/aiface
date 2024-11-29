# AI颜值评分API

这是一个基于深度学习的人脸颜值评分API服务。使用了先进的人脸识别和分析模型，可以对上传的图片中的人脸进行分析并给出颜值评分。

## 功能特点

- 支持多人脸检测和评分
- 提供性别和年龄识别
- 返回人脸位置信息
- RESTful API接口
- 跨域支持

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行服务

```bash
python main.py
```

服务将在 http://localhost:8000 启动

## API使用

### 1. 颜值评分接口

**POST** `/analyze`

上传图片文件，返回颜值评分结果。

请求示例：
```bash
curl -X POST "http://localhost:8000/analyze" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@your_image.jpg"
```

返回示例：
```json
{
  "results": [
    {
      "score": 85.6,
      "bbox": [100, 200, 300, 400],
      "gender": "female",
      "age": 25.5
    }
  ]
}
```

## 注意事项

- 请确保上传清晰的正面人像照片
- 支持常见的图片格式（JPG、PNG等）
- 建议图片分辨率不低于640x640
