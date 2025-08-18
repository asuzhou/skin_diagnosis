import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
import pickle

# -------------------------- 配置参数 --------------------------
IMAGE_DIR = "data/images2"  # 图片文件夹路径（id.jpg）
METADATA_PATH = "data/metadata2.csv"     # 包含id和label的CSV路径
OUTPUT_DIR = "config3"        # 模型和配置文件保存目录
BATCH_SIZE = 32                    # 批次大小
EPOCHS = 40                        # 训练轮次
LR = 6e-4                          # 初始学习率
Weight_decay=2e-4
IMAGE_SIZE = 380                   # 输入图像尺寸（EfficientNetB4推荐380）
# 标准化参数（ImageNet均值和标准差）
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --------------------------------------------------------------

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 数据处理类与预处理配置（与推理共享）
class SkinDiseaseDataset(Dataset):
    def __init__(self, image_ids, labels, image_dir, transform=None):
        self.image_ids = image_ids
        self.labels = labels
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")  # 统一转为RGB
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, torch.tensor(label, dtype=torch.long)

# 保存预处理配置（供推理使用）
preprocess_config = {
    "image_size": IMAGE_SIZE,
    "normalize_mean": NORMALIZE_MEAN,
    "normalize_std": NORMALIZE_STD
}
with open(os.path.join(OUTPUT_DIR, "preprocess_config.json"), "w") as f:
    json.dump(preprocess_config, f, indent=4)

# 构建数据变换（增强训练集的数据多样性）
def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            # 先放大再裁剪，增加局部特征学习
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            #transforms.RandomCrop(IMAGE_SIZE),  # 随机裁剪到目标尺寸
            transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转（50%概率）
            transforms.RandomVerticalFlip(p=0.2),  # 垂直翻转（降低概率，避免破坏特征）
            transforms.RandomRotation(10),  # 旋转角度
            # 更激进的随机缩放裁剪
            transforms.RandomResizedCrop(
                IMAGE_SIZE, 
                scale=(0.8, 1.0),  # 缩放范围
                ratio=(0.95, 1.05)   # 宽高比轻微变化
            ),
            # 增强色彩抖动
            transforms.ColorJitter(
                brightness=0.2,    # 亮度变化范围增大
                contrast=0.2,      # 对比度变化范围增大
                saturation=0.2,    # 饱和度变化范围增大
                hue=0.05            # 色调轻微变化（避免改变疾病特征）
            ),
            # 新增：随机高斯模糊（模拟模糊图片）
            transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 1.0)),
            # 新增：随机调整锐度
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),

            # 新增：随机擦除（模拟遮挡，如毛发、污渍）
            transforms.ToTensor(),
            # transforms.RandomErasing(
            #     p=0.3,               # 30%概率应用
            #     scale=(0.02, 0.25),  # 擦除区域大小
            #     ratio=(0.3, 3.3)     # 擦除区域比例
            # ),
            
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])
    else:
        # 验证集保持简单变换（仅 resize 和标准化）
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])

# 2. 数据加载与标签编码（动态处理类别）
def load_data():
    # 读取元数据
    df = pd.read_csv(METADATA_PATH)
    df["id"] = df["id"].astype(str)  # 确保id为字符串
    original_count = len(df)
    original_labels = df["nine_partition_label"].nunique()
    print(f"原始数据：{original_count}条样本，{original_labels}个类别")
    
    # 统计每个类别的样本数
    label_counts = df["nine_partition_label"].value_counts()
    # 筛选出样本数
    valid_labels = label_counts[label_counts >= 50].index
    df = df[df["nine_partition_label"].isin(valid_labels)]
    
    # 输出过滤后的统计信息
    filtered_count = len(df)
    filtered_labels = df["nine_partition_label"].nunique()
    print(f"过滤后数据：{filtered_count}条样本，{filtered_labels}个类别")
    if original_labels != filtered_labels:
        removed = original_labels - filtered_labels
        print(f"共移除{removed}个类别")
    
    # 标签编码并保存编码器
    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["nine_partition_label"])
    
    # 保存标签编码器（推理时需要将预测结果转回原标签）
    with open(os.path.join(OUTPUT_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    
    # 划分训练集和验证集（此时每个类别至少有2个样本，可安全使用分层抽样）
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df["label_encoded"]
    )
    
    # 构建数据集和数据加载器
    train_dataset = SkinDiseaseDataset(
        image_ids=train_df["id"].values,
        labels=train_df["label_encoded"].values,
        image_dir=IMAGE_DIR,
        transform=get_transforms(is_train=True)
    )
    val_dataset = SkinDiseaseDataset(
        image_ids=val_df["id"].values,
        labels=val_df["label_encoded"].values,
        image_dir=IMAGE_DIR,
        transform=get_transforms(is_train=False)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4
    )
    
    # 返回加载器和实际类别数
    return train_loader, val_loader, filtered_labels

# 3. 模型定义（接收动态类别数）
def build_model(num_classes):
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    
    # 冻结部分预训练层
    for param in list(model.parameters())[:-30]:
        param.requires_grad = False
    
    # 替换分类头（增强正则化）
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    
    return model.to(DEVICE)

# 4. 训练与验证（无进度条）
def train_model():
    # 加载数据并获取实际类别数
    train_loader, val_loader, num_classes = load_data()
    
    # 若过滤后无有效数据，退出
    if num_classes == 0:
        print("❌ 过滤后无有效类别，无法训练")
        return
    
    model = build_model(num_classes)  # 使用动态类别数
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=LR,
        weight_decay=Weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3
    )
    
    best_val_acc = 0.0
    patience = 5  # 早停策略：连续5轮无提升则停止
    no_improve_epochs = 0
    print(f"开始训练（设备：{DEVICE}，类别数：{num_classes}）")
    
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        train_loss /= train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total
        val_loss /= val_total
        
        # 调整学习率
        scheduler.step(val_loss)
        
        # 打印当前 epoch 结果
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"训练集 - 损失: {train_loss:.4f}, 准确率: {train_acc:.4f}")
        print(f"验证集 - 损失: {val_loss:.4f}, 准确率: {val_acc:.4f}")
        print("-" * 50)
        
        # 保存最佳模型（包含动态类别数）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_epochs = 0  # 重置无提升计数器
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_classes': num_classes
            }, os.path.join(OUTPUT_DIR, "skin_model2.pth"))
            print(f"保存最佳模型（验证准确率：{best_val_acc:.4f}）")
            print("-" * 50)
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"早停,连续{patience}轮验证准确率未提升，最佳准确率：{best_val_acc:.4f}")
                break
    
    print(f"训练完成！最佳验证准确率：{best_val_acc:.4f}")
    print(f"模型和配置文件已保存至：{OUTPUT_DIR}")

if __name__ == "__main__":
    train_model()
