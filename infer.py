import os
import torch
import joblib
import numpy as np
from PIL import Image
import pandas as pd
from torchvision import transforms
from train import FusionModel, Config

class SkinInferencer:
    def __init__(self, model_path, meta_preprocessor_path, label_encoder_path):
        """初始化推理器：自动适配GPU/CPU环境"""
        # 1. 自动检测设备（优先GPU，无则CPU）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_size = Config.img_size
        self.use_amp = torch.cuda.is_available()  # 仅GPU支持混合精度
        
        # 2. 加载预处理组件和标签编码器
        self.meta_preprocessor = joblib.load(meta_preprocessor_path)
        self.label_encoder = joblib.load(label_encoder_path)
        self.num_classes = len(self.label_encoder.classes_)
        
        # 3. 初始化模型并加载权重
        sample_meta_df = pd.DataFrame([[30, "head"]], columns=['age', 'region'])
        meta_feature_dim = self.meta_preprocessor.transform(sample_meta_df).shape[1]

        self.model = FusionModel(self.num_classes, meta_feature_dim).to(self.device)
        
        # 加载模型并处理类型兼容性
        state_dict = torch.load(
            model_path, 
            map_location=self.device,
            weights_only=True
        )
        # 转换不支持的类型（CPU不支持BFloat16）
        target_dtype = torch.float32 if self.device.type == 'cpu' else None
        for key in state_dict:
            if target_dtype and state_dict[key].dtype == torch.bfloat16:
                state_dict[key] = state_dict[key].to(dtype=target_dtype)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # 4. 图像预处理处理
        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_input(self, img_path, age, region):
        """预处理输入，自动适配设备类型"""
        # 图像处理
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"加载图像失败：{e}")
        image = self.img_transform(image).unsqueeze(0)
        image = image.to(self.device)
        
        # 元数据处理
        meta_data = pd.DataFrame([[age, region]], columns=['age', 'region'])
        meta_processed = self.meta_preprocessor.transform(meta_data)
        meta_tensor = torch.FloatTensor(meta_processed).to(self.device)
        
        return image, meta_tensor
    
    def predict(self, img_path, age, region):
        """预测：根据设备自动选择推理模式"""
        image, meta_tensor = self.preprocess_input(img_path, age, region)
        
        # 推理（根据设备选择是否使用混合精度）
        with torch.no_grad():
            # 仅GPU启用混合精度，CPU使用常规模式
            if self.use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    outputs = self.model(image, meta_tensor)
            else:
                outputs = self.model(image, meta_tensor)
            
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
        # 结果处理
        pred_label_idx = np.argmax(probs)
        pred_label = self.label_encoder.inverse_transform([pred_label_idx])[0]
        
        return {
            "predicted_diagnostic": pred_label,
            "probabilities": np.max(probs),
            "device_used": self.device.type  # 记录使用的设备
        }


if __name__ == "__main__":
    # 配置路径
    MODEL_PATH = "config/best_fusion_model_skin.pth"
    META_PREPROCESSOR_PATH = "config/meta_preprocessor.pkl"
    LABEL_ENCODER_PATH = "config/label_encoder.pkl"

    # 初始化推理器
    inferencer = SkinInferencer(MODEL_PATH, META_PREPROCESSOR_PATH, LABEL_ENCODER_PATH)
    print(f"使用设备: {inferencer.device}")
    
    # 示例预测
    test_img_path = "data/images/PAT_8_15_820.png"
    test_age = 53
    test_region = "CHEST"
    
    try:
        result = inferencer.predict(test_img_path, test_age, test_region)
        print(f"预测诊断：{result['predicted_diagnostic']}")
        print(f"类别概率：{result['probabilities']:.2f}")

    except Exception as e:
        print(f"推理失败：{e}")
    