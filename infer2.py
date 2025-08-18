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

diagnose_map2={
                "hailey hailey disease": "家族性良性天疱疮",
                "papilomatosis confluentes and reticulate": "融合性网状乳头瘤病",
                "scabies": "疥疮",
                "tuberous sclerosis": "结节性硬化症",
                "keloid": "瘢痕疙瘩",
                "tungiasis": "潜蚤病",
                "dermatofibroma": "皮肤纤维瘤",
                "lupus erythematosus": "红斑狼疮",
                "dariers disease": "达里埃病（毛囊角化病）",
                "neutrophilic dermatoses": "嗜中性皮病",
                "seborrheic dermatitis": "脂溢性皮炎",
                "photodermatoses": "光敏感性皮肤病",
                "cheilitis": "唇炎",
                "mucinosis": "黏蛋白沉积症",
                "myiasis": "蝇蛆病",
                "sarcoidosis": "结节病",
                "nematode infection": "线虫感染",
                "erythema nodosum": "结节性红斑",
                "keratosis pilaris": "毛发角化病",
                "lichen simplex": "单纯性苔藓",
                "scleromyxedema": "硬化性黏液水肿",
                "pityriasis rosea": "玫瑰糠疹",
                "pityriasis rubra pilaris": "毛发红糠疹",
                "melanoma": "黑色素瘤",
                "squamous cell carcinoma": "鳞状细胞癌",
                "lichen amyloidosis": "苔藓样淀粉样变",
                "mycosis fungoides": "蕈样肉芽肿",
                "scleroderma": "硬皮病",
                "porokeratosis actinic": "光化性汗孔角化症",
                "allergic contact dermatitis": "过敏性接触性皮炎",
                "naevus comedonicus": "黑头粉刺痣",
                "folliculitis": "毛囊炎",
                "vitiligo": "白癜风",
                "erythema annulare centrifigum": "离心性环状红斑",
                "dyshidrotic eczema": "汗疱疹",
                "neurofibromatosis": "神经纤维瘤病",
                "lichen planus": "扁平苔藓",
                "calcinosis cutis": "皮肤钙质沉着症",
                "pediculosis lids": "眼睑虱病",
                "basal cell carcinoma": "基底细胞癌",
                "granuloma annulare": "环状肉芽肿",
                "fixed eruptions": "固定性药疹",
                "psoriasis": "银屑病（牛皮癣）",
                "fordyce spots": "福代斯斑点（异位皮脂腺）",
                "porokeratosis of mibelli": "米贝利汗孔角化症",
                "syringoma": "汗管瘤",
                "granuloma pyogenic": "化脓性肉芽肿",
                "telangiectases": "毛细血管扩张",
                "erythema multiforme": "多形红斑",
                "pityriasis lichenoides chronica": "慢性苔藓样糠疹",
                "factitial dermatitis": "人工性皮炎",
                "kaposi sarcoma": "卡波西肉瘤",
                "ehlers danlos syndrome": "埃勒斯 - 当洛斯综合征",
                "prurigo nodularis": "结节性痒疹",
                "neurotic excoriations": "神经官能性表皮剥脱",
                "actinic keratosis": "光化性角化病",
                "acrodermatitis enteropathica": "肠病性肢端皮炎",
                "juvenile xanthogranuloma": "幼年性黄色肉芽肿",
                "stasis edema": "淤滞性水肿",
                "acanthosis nigricans": "黑棘皮病",
                "urticaria": "荨麻疹",
                "aplasia cutis": "皮肤发育不全",
                "lymphangioma": "淋巴管瘤",
                "acne": "痤疮",
                "langerhans cell histiocytosis": "朗格汉斯细胞组织细胞增生症",
                "ichthyosis vulgaris": "寻常型鱼鳞病",
                "behcets disease": "贝赫切特病",
                "drug induced pigmentary changes": "药物性色素异常",
                "necrobiosis lipoidica": "类脂质渐进性坏死",
                "pilar cyst": "毛发囊肿",
                "mucous cyst": "黏液囊肿",
                "porphyria": "卟啉病",
                "erythema elevatum diutinum": "持久性隆起性红斑",
                "milia": "粟丘疹",
                "hidradenitis": "汗腺炎",
                "xeroderma pigmentosum": "着色性干皮病",
                "rhinophyma": "鼻赘",
                "urticaria pigmentosa": "色素性荨麻疹",
                "dermatomyositis": "皮肌炎",
                "striae": "皮肤条纹（萎缩纹）",
                "eczema": "湿疹",
                "tick bite": "蜱虫叮咬",
                "pilomatricoma": "毛母质瘤",
                "lupus subacute": "亚急性红斑狼疮",
                "paronychia": "甲沟炎",
                "congenital nevus": "先天性色素痣",
                "lyme disease": "莱姆病",
                "epidermal nevus": "表皮痣",
                "perioral dermatitis": "口周皮炎",
                "rosacea": "玫瑰痤疮"
                }
# 示例用法
if __name__ == "__main__":
    # 配置路径
    MODEL_PATH = "config2/skin_model2.pth"
    CONFIG_DIR = "config2"
    
    # 初始化推理器
    inferencer = SkinDiseaseInference(MODEL_PATH, CONFIG_DIR)
    
    # 单张图像预测示例
    test_image = "data/测试3.jpg"  # 替换为实际图像路径
    if os.path.exists(test_image):
        result = inferencer.predict_single_image(test_image)
        print(f"单张图像预测结果:")
        print(f"类别: {diagnose_map2[result['label']]}")
        print(f"置信度: {result['confidence']:.4f}")
        print(f"类别索引: {result['class_index']}")
    
    # 批量预测示例
    # test_images = ["image1.jpg", "image2.jpg", "image3.jpg"]  # 替换为实际图像路径列表
    # if test_images and all(os.path.exists(img) for img in test_images):
    #     batch_results = inferencer.predict_batch(test_images)
    #     print("\n批量预测结果:")
    #     for i, res in enumerate(batch_results):
    #         print(f"图像 {i+1}: {res['label']} (置信度: {res['confidence']:.4f})")
    