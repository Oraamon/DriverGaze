import os
import shutil
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import yaml
from pathlib import Path
import random
import glob

def prepare_yolo_dataset(source_dir, output_dir, train_ratio=0.8, max_images_per_class=1000):
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    class_names = sorted([d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))])
    
    for class_name in class_names:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        
        class_dir = os.path.join(source_dir, class_name)
        all_images = []
        
        print(f"Coletando imagens da classe {class_name}...")
        for subdir in os.listdir(class_dir):
            subdir_path = os.path.join(class_dir, subdir)
            if os.path.isdir(subdir_path):
                all_images.extend(glob.glob(os.path.join(subdir_path, "*.jpg")))
        
        print(f"Total de imagens encontradas: {len(all_images)}")
        
        random.shuffle(all_images)
        selected_images = all_images[:max_images_per_class]
        train_count = int(len(selected_images) * train_ratio)
        
        train_images = selected_images[:train_count]
        val_images = selected_images[train_count:]
        
        print(f"Usando {len(train_images)} imagens para treino e {len(val_images)} para validação")
        
        for i, img_path in enumerate(train_images):
            filename = f"{class_name}_{i:05d}.jpg"
            shutil.copy2(img_path, os.path.join(train_dir, class_name, filename))
        
        for i, img_path in enumerate(val_images):
            filename = f"{class_name}_{i:05d}.jpg"
            shutil.copy2(img_path, os.path.join(val_dir, class_name, filename))
    
    return class_names

def create_dataset_yaml(dataset_path, class_names):
    yaml_content = {
        'path': dataset_path,
        'train': 'train',
        'val': 'val',
        'nc': len(class_names),
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    yaml_path = os.path.join(dataset_path, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    return yaml_path

def evaluate_model(model, val_dir, class_names):
    y_true = []
    y_pred = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(val_dir, class_name)
        if os.path.exists(class_dir):
            for img_file in os.listdir(class_dir):
                if img_file.endswith('.jpg'):
                    img_path = os.path.join(class_dir, img_file)
                    results = model(img_path)
                    predicted_class = results[0].probs.top1
                    
                    y_true.append(class_idx)
                    y_pred.append(predicted_class)
    
    return np.array(y_true), np.array(y_pred)

print("Preparando dataset para YOLO...")
dataset_dir = 'yolo_dataset'
class_names = prepare_yolo_dataset('frame', dataset_dir)
print(f"Classes encontradas: {class_names}")

print("Iniciando treinamento com YOLOv8...")
model = YOLO('yolov8n-cls.pt')

results = model.train(
    data=dataset_dir,
    epochs=50,
    imgsz=224,
    batch=16,
    device='cpu',
    patience=10,
    save=True,
    plots=True,
    verbose=True
)

print("Treinamento concluído!")

print("Avaliando modelo...")
val_dir = os.path.join(dataset_dir, 'val')
y_true, y_pred = evaluate_model(model, val_dir, class_names)

print("\nRelatório de Classificação:")
print(classification_report(y_true, y_pred, target_names=class_names))

print("\nGerando matriz de confusão...")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusão - YOLO DriveGaze')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('yolo_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("Matriz de confusão salva como 'yolo_confusion_matrix.png'")

print("\nSalvando modelo final...")
model.export(format='onnx')
print("Modelo exportado para formato ONNX")
print(f"Modelo YOLO salvo em: runs/classify/train/weights/best.pt") 