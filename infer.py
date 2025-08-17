import os
import torch
import joblib
import numpy as np
from PIL import Image
import pandas as pd
from torchvision import transforms
# 导入训练代码中的模型和配置
from train import FusionModel, Config

class SkinInferencer:
    def __init__(self, model_path, meta_preprocessor_path, label_encoder_path):
        """初始化推理器：加载模型、预处理组件和标签编码器"""
        self.device = Config.device
        self.img_size = Config.img_size
        
        # 1. 加载元数据预处理管道和标签编码器
        self.meta_preprocessor = joblib.load(meta_preprocessor_path)
        self.label_encoder = joblib.load(label_encoder_path)
        self.num_classes = len(self.label_encoder.classes_)
        
        # 2. 初始化模型并加载权重
        sample_meta_df = pd.DataFrame([[30, "head"]], columns=['age', 'region'])  # 与训练时列名一致
        meta_feature_dim = self.meta_preprocessor.transform(sample_meta_df).shape[1]

        self.model = FusionModel(self.num_classes, meta_feature_dim).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()  # 切换到评估模式
        
        # 3. 定义图像预处理（与训练时的验证集预处理一致）
        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_input(self, img_path, age, region):
        """预处理输入：图像和元数据"""
        # 1. 图像预处理
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"加载图像失败：{e}")
        image = self.img_transform(image).unsqueeze(0)  # 增加batch维度 (1, 3, 224, 224)
        
        # 2. 元数据预处理（与训练时一致）
        meta_data = pd.DataFrame([[age, region]], columns=['age', 'region'])  # 形状 (1, 2)
        meta_processed = self.meta_preprocessor.transform(meta_data)  # 应用训练时的预处理
        meta_tensor = torch.FloatTensor(meta_processed).to(self.device)
        
        return image.to(self.device), meta_tensor
    
    def predict(self, img_path, age, region):
        """预测：输入图像路径、年龄、病变区域，返回诊断结果和概率"""
        # 预处理输入
        image, meta_tensor = self.preprocess_input(img_path, age, region)
        
        # 推理（关闭梯度计算）
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                outputs = self.model(image, meta_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]  # 转换为概率
        
        # 转换为诊断标签
        pred_label_idx = np.argmax(probs)
        pred_label = self.label_encoder.inverse_transform([pred_label_idx])[0]
        
        # 整理结果：字典形式包含标签和各类别概率
        result = {
            "predicted_diagnostic": pred_label,
            "probabilities": {
                cls: float(probs[i]) for i, cls in enumerate(self.label_encoder.classes_)
            }
        }
        return result


if __name__ == "__main__":
    #配置文件路径
    MODEL_PATH = "config/best_fusion_model_skin.pth"  # 训练好的模型权重
    META_PREPROCESSOR_PATH = "config/meta_preprocessor.pkl"  # 元数据预处理管道
    LABEL_ENCODER_PATH = "config/label_encoder.pkl"  # 标签编码器

    # 初始化推理器
    inferencer = SkinInferencer(MODEL_PATH, META_PREPROCESSOR_PATH, LABEL_ENCODER_PATH)
    
    # 示例输入（根据实际情况修改）
    test_img_path = "data/images/PAT_8_15_820.png"  # 测试图像路径
    test_age = 53  # 患者年龄
    test_region = "CHEST"  # 病变区域（需与训练时的类别一致，如"head"、"trunk"等）
    
    # 预测并打印结果
    try:
        result = inferencer.predict(test_img_path, test_age, test_region)
        print("推理结果：")
        print(f"预测诊断：{result['predicted_diagnostic']}")
        print("类别概率：")
        for cls, prob in result["probabilities"].items():
            print(f"  {cls}: {prob:.4f}")
    except Exception as e:
        print(f"推理失败：{e}")