import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import pickle
from torch.utils.data import Dataset, DataLoader

class SkinDiseaseInference:
    def __init__(self, model_path, config_dir):
        """
        初始化推理器
        :param model_path: 模型权重文件路径 (e.g., "config2/skin_model2.pth")
        :param config_dir: 配置文件目录 (e.g., "config2")
        """
        # 加载配置文件
        self.config = self._load_config(config_dir)
        self.label_encoder = self._load_label_encoder(config_dir)
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()  # 设置为评估模式
        
        # 图像预处理
        self.transform = self._get_transforms()
        
    def _load_config(self, config_dir):
        """加载预处理配置"""
        config_path = os.path.join(config_dir, "preprocess_config.json")
        with open(config_path, "r") as f:
            return json.load(f)
    
    def _load_label_encoder(self, config_dir):
        """加载标签编码器"""
        le_path = os.path.join(config_dir, "label_encoder.pkl")
        with open(le_path, "rb") as f:
            return pickle.load(f)
    
    def _get_transforms(self):
        """获取图像预处理变换"""
        return transforms.Compose([
            transforms.Resize((self.config["image_size"], self.config["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config["normalize_mean"],
                std=self.config["normalize_std"]
            )
        ])
    
    def _load_model(self, model_path):
        """加载模型结构和权重"""
        # 加载模型权重和类别数
        checkpoint = torch.load(model_path, map_location=self.device)
        num_classes = checkpoint["num_classes"]
        
        # 构建模型结构
        from torchvision import models
        model = models.efficientnet_b4(weights=None)  # 不加载预训练权重
        
        # 替换分类头
        in_features = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.3, inplace=True),
            torch.nn.Linear(in_features, num_classes)
        )
        
        # 加载训练好的权重
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(self.device)
    
    def predict_single_image(self, image_path):
        """
        预测单张图像
        :param image_path: 图像文件路径
        :return: 预测结果字典，包含原始标签、预测概率和类别索引
        """
        # 加载和预处理图像
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0)  # 添加批次维度
        image = image.to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            predicted_prob = probabilities[0][predicted_idx].item()
        
        # 转换为原始标签
        predicted_label = self.label_encoder.inverse_transform([predicted_idx])[0]
        
        return {
            "label": predicted_label,
            "confidence": predicted_prob,
            "class_index": predicted_idx,
            "all_probabilities": probabilities.cpu().numpy()[0].tolist()
        }
    
    def predict_batch(self, image_paths, batch_size=32):
        """
        批量预测图像
        :param image_paths: 图像路径列表
        :param batch_size: 批次大小
        :return: 预测结果列表，每个元素是单张图像的预测结果字典
        """
        # 创建自定义数据集
        class BatchDataset(Dataset):
            def __init__(self, paths, transform):
                self.paths = paths
                self.transform = transform
                
            def __len__(self):
                return len(self.paths)
                
            def __getitem__(self, idx):
                image = Image.open(self.paths[idx]).convert("RGB")
                return self.transform(image)
        
        # 创建数据加载器
        dataset = BatchDataset(image_paths, self.transform)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        # 批量预测
        results = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_indices = torch.argmax(probabilities, dim=1).cpu().numpy()
                predicted_probs = probabilities[torch.arange(len(probabilities)), predicted_indices].cpu().numpy()
                
                # 转换为原始标签
                predicted_labels = self.label_encoder.inverse_transform(predicted_indices)
                
                # 收集结果
                for label, prob, idx, probs in zip(
                    predicted_labels, 
                    predicted_probs, 
                    predicted_indices,
                    probabilities.cpu().numpy()
                ):
                    results.append({
                        "label": label,
                        "confidence": float(prob),
                        "class_index": int(idx),
                        "all_probabilities": probs.tolist()
                    })
        
        return results

diagnose_map3={"genodermatoses": "遗传性皮肤病", "inflammatory": "炎症性皮肤病", 
               "benign dermal": "良性真皮肿瘤", "malignant melanoma": "恶性黑色素瘤", 
               "malignant epidermal": "恶性表皮肿瘤", "malignant cutaneous lymphoma": "恶性皮肤淋巴瘤",
                 "benign epidermal": "良性表皮肿瘤", "malignant dermal": "恶性真皮肿瘤", 
                 "benign melanocyte": "良性黑素细胞肿瘤"}
# 示例用法
if __name__ == "__main__":
    # 配置路径
    MODEL_PATH = "config3/skin_model2.pth"
    CONFIG_DIR = "config3"
    
    # 初始化推理器
    inferencer = SkinDiseaseInference(MODEL_PATH, CONFIG_DIR)
    
    # 单张图像预测示例
    test_image = "data/测试3.jpg"  # 替换为实际图像路径
    if os.path.exists(test_image):
        result = inferencer.predict_single_image(test_image)
        print(f"单张图像预测结果:")
        print(f"类别: {diagnose_map3[result['label']]}")
        print(f"置信度: {result['confidence']:.4f}")
        print(f"类别索引: {result['class_index']}")
    
    # 批量预测示例
    # test_images = ["image1.jpg", "image2.jpg", "image3.jpg"]  # 替换为实际图像路径列表
    # if test_images and all(os.path.exists(img) for img in test_images):
    #     batch_results = inferencer.predict_batch(test_images)
    #     print("\n批量预测结果:")
    #     for i, res in enumerate(batch_results):
    #         print(f"图像 {i+1}: {res['label']} (置信度: {res['confidence']:.4f})")
    