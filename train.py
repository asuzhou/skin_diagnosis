import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# 1. 配置参数
class Config:
    data_dir = "data/images"  # 图像文件夹
    csv_path = "data/metadata.csv"  
    
    num_classes = 6  # PAD-UFES-20的6个诊断类别
    img_size = 224  
    batch_size = 24  # 融合元数据后适当降低batch size
    epochs = 30
    lr = 1e-4
    weight_decay = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_split = 0.2  # 训练集:验证集=8:2
    patience = 5  # 早停机制

# 2. 数据集定义（包含图像+元数据）
class SkinDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, meta_preprocessor=None):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.meta_preprocessor = meta_preprocessor
        
        # 编码诊断标签（字符串→数字）
        self.label_encoder = LabelEncoder()
        self.df['label'] = self.label_encoder.fit_transform(self.df['diagnostic'])
        
        # 提取需要的元数据（年龄和病变区域）
        self.meta_data = self.df[['age', 'region']].copy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 加载图像（代码不变）
        img_name = self.df.iloc[idx]['img_id']
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (Config.img_size, Config.img_size))
        if self.transform:
            image = self.transform(image)
        
        # 处理元数据（代码不变）
        meta_row = self.meta_data.iloc[idx:idx+1]
        if self.meta_preprocessor:
            meta_processed = self.meta_preprocessor.transform(meta_row)
        else:
            meta_processed = meta_row.values
        meta_tensor = torch.FloatTensor(meta_processed.ravel())
        
        # 处理标签（关键修改：转换为long类型）
        label = self.df.iloc[idx]['label']
        label_tensor = torch.tensor(label, dtype=torch.long)  # 显式指定dtype=torch.long
        
        return image, meta_tensor, label_tensor

# 3. 图像增强策略
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.RandomCrop(Config.img_size - 32),  # 聚焦病变区域
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 模拟光照变化
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

# 4. 元数据预处理管道（处理年龄和病变区域）
def get_meta_preprocessor(train_df):
    # 定义数值型和类别型特征
    numeric_features = ['age']  # 年龄是数值型
    categorical_features = ['region']  # 病变区域是类别型
    
    # 数值型预处理：填充缺失值（用中位数）
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))  # 处理缺失的年龄数据
    ])
    
    # 类别型预处理：填充缺失值（用最频繁值）+ 独热编码
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # 处理缺失的病变区域
        ('onehot', OneHotEncoder(handle_unknown='ignore',sparse_output=False))  # 转为独热向量（如头部→[1,0,0,...]）
    ])
    
    # 组合预处理流程
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # 用训练数据拟合预处理管道（确保验证集使用相同的处理方式）
    preprocessor.fit(train_df[['age', 'region']])
    
    return preprocessor

# 5. 融合模型（图像特征 + 元数据特征）
class FusionModel(nn.Module):
    def __init__(self, num_classes, meta_feature_dim):
        super().__init__()
        # 图像分支（MobileNetV3-Large）
        self.image_backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        
        # 主干特征提取（输出特征图）
        self.backbone_features = self.image_backbone.features  # 关键：获取主干网络（输出三维特征图）
        

        # 全局平均池化（将特征图压缩为向量）
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # 输出形状：(batch_size, 960, 1, 1)
        
        # 分类器前几层（处理池化后的向量）
        self.classifier_layers = nn.Sequential(
            *list(self.image_backbone.classifier.children())[:-1]  # 移除最后一层分类头
        )
        self.image_feature_dim = 1280  # MobileNetV3分类器前几层的输出维度固定为1280
        
        # 元数据分支（保持不变）
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32)
        )
        
        # 融合后分类头
        self.fusion_head = nn.Linear(self.image_feature_dim + 32, num_classes)
        
        # 全量微调
        for param in self.parameters():
            param.requires_grad = True
    
    def forward(self, images, metadata):
        # 1. 提取图像特征（补全流程）
        x = self.backbone_features(images)  # 主干网络输出三维特征图：(batch_size, 960, 7, 7)
        x = self.global_pool(x)  # 全局池化：(batch_size, 960, 1, 1)
        x = torch.flatten(x, 1)  # 展平为二维向量：(batch_size, 960)
        image_feat = self.classifier_layers(x)  # 经过分类器前几层：(batch_size, 1280)
        
        # 2. 提取元数据特征
        meta_feat = self.meta_mlp(metadata)  # (batch_size, 32)
        
        # 3. 融合特征并分类
        fused_feat = torch.cat([image_feat, meta_feat], dim=1)  # (batch_size, 1280+32=1312)
        output = self.fusion_head(fused_feat)
        return output
    

# 6. 训练和验证函数
def train_epoch(model, train_loader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(train_loader)
    
    for batch_idx, (images, metadata, labels) in enumerate(train_loader):
        # if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
        #     print(f"  Batch {batch_idx + 1}/{total_batches}")
        
        # 转移到设备
        images = images.to(Config.device)
        metadata = metadata.to(Config.device)
        labels = labels.to(Config.device)
        
        optimizer.zero_grad()
        
        # 混合精度训练
        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            outputs = model(images, metadata)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 统计指标
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return correct/total, total_loss/total_batches

def val_epoch(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(val_loader)
    
    with torch.no_grad():
        for batch_idx, (images, metadata, labels) in enumerate(val_loader):
            # if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            #     print(f"  Batch {batch_idx + 1}/{total_batches}")
            
            images = images.to(Config.device)
            metadata = metadata.to(Config.device)
            labels = labels.to(Config.device)
            
            with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                outputs = model(images, metadata)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct/total, total_loss/total_batches

# 7. 主训练流程
def main():
    # 加载数据
    df = pd.read_csv(Config.csv_path)
    
    # 过滤存在的图像（避免路径错误）
    df = df[df['img_id'].apply(
        lambda x: os.path.exists(os.path.join(Config.data_dir, x))
    )]
    
    # 分割训练集和验证集（保持类别比例）
    train_df, val_df = train_test_split(
        df, 
        test_size=Config.val_split, 
        random_state=42, 
        stratify=df['diagnostic']
    )
    
    # 创建元数据预处理管道（用训练数据拟合）
    meta_preprocessor = get_meta_preprocessor(train_df)
    
    # 计算元数据特征维度（用于模型构建）
    sample_meta = train_df[['age', 'region']].iloc[:1]
    meta_feature_dim = meta_preprocessor.transform(sample_meta).shape[1]
    print(f"元数据特征维度: {meta_feature_dim}（年龄1维 + 病变区域独热编码维度）")
    
    # 获取图像增强策略
    train_transform, val_transform = get_transforms()
    
    # 创建数据集
    train_dataset = SkinDataset(
        train_df, 
        Config.data_dir, 
        train_transform, 
        meta_preprocessor
    )
    val_dataset = SkinDataset(
        val_df, 
        Config.data_dir, 
        val_transform, 
        meta_preprocessor
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.batch_size, 
        shuffle=True, 
        num_workers=2  # 减少线程数降低内存占用
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    # 构建融合模型
    model = FusionModel(Config.num_classes, meta_feature_dim).to(Config.device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑防过拟合
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=Config.lr, 
        weight_decay=Config.weight_decay
    )
    scaler = torch.amp.GradScaler()  # 混合精度训练
    
    # 早停机制变量
    best_val_acc = 0.0
    counter = 0
    #保存，用于推理
    joblib.dump(meta_preprocessor, "meta_preprocessor.pkl")  # 保存预处理管道
    joblib.dump(train_dataset.label_encoder, "label_encoder.pkl")  # 保存标签编码器
    # 训练循环
    for epoch in range(Config.epochs):
        print(f"\nEpoch {epoch+1}/{Config.epochs}")
        print("Training...")
        train_acc, train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler)
        
        print("Validation...")
        val_acc, val_loss = val_epoch(model, val_loader, criterion)
        
        # 打印本轮指标
        print(f"Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}")
        print(f"Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_fusion_model_skin.pth")
            print(f"Saved best model (Val Acc: {best_val_acc:.4f})")
            counter = 0  # 重置早停计数器
        else:
            counter += 1
            if counter >= Config.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"Training complete. Best Val Acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
    