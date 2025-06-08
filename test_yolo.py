import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
import glob

def test_single_image(model, image_path, class_names):
    results = model(image_path)
    result = results[0]
    
    predicted_class_idx = result.probs.top1
    confidence = result.probs.top1conf.item()
    predicted_class = class_names[predicted_class_idx]
    
    img = Image.open(image_path)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f'Predição: {predicted_class} (Confiança: {confidence:.2f})')
    plt.axis('off')
    plt.show()
    
    print(f"Imagem: {os.path.basename(image_path)}")
    print(f"Classe predita: {predicted_class}")
    print(f"Confiança: {confidence:.4f}")
    print("Top 3 predições:")
    
    top3_indices = result.probs.top5[:3]
    top3_confidences = result.probs.top5conf[:3]
    
    for i, (idx, conf) in enumerate(zip(top3_indices, top3_confidences)):
        print(f"  {i+1}. {class_names[idx]}: {conf:.4f}")
    print("-" * 50)
    
    return predicted_class, result.probs.data.tolist()

def test_random_images(model, test_dir, class_names, num_images=5):
    print(f"Testando {num_images} imagens aleatórias do conjunto de validação...")
    
    all_test_images = []
    for class_name in class_names:
        class_dir = os.path.join(test_dir, class_name)
        if os.path.exists(class_dir):
            images = glob.glob(os.path.join(class_dir, "*.jpg"))
            for img_path in images:
                all_test_images.append((img_path, class_name))
    
    if not all_test_images:
        print("Nenhuma imagem de teste encontrada!")
        return
    
    random_images = random.sample(all_test_images, min(num_images, len(all_test_images)))
    
    for img_path, true_class in random_images:
        print(f"\nClasse verdadeira: {true_class}")
        test_single_image(model, img_path, class_names)

def evaluate_on_validation_set(model, val_dir, class_names):
    print("Avaliando no conjunto de validação (20% das imagens)...")
    
    correct = 0
    total = 0
    class_correct = {class_name: 0 for class_name in class_names}
    class_total = {class_name: 0 for class_name in class_names}
    
    # Criar diretório para resultados se não existir
    os.makedirs('test_results', exist_ok=True)
    
    # Abrir arquivo para salvar predições
    with open('test_results/predictions.txt', 'w', encoding='utf-8') as f:
        f.write("Resultados das Predições\n")
        f.write("======================\n\n")
        
        print("\nDistribuição de imagens por classe no conjunto de validação:")
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(val_dir, class_name)
            if os.path.exists(class_dir):
                images = glob.glob(os.path.join(class_dir, "*.jpg"))
                class_total[class_name] = len(images)
                print(f"  {class_name}: {len(images)} imagens")
                
                f.write(f"\nClasse Real: {class_name}\n")
                f.write("-" * 50 + "\n")
                
                for img_path in images:
                    results = model(img_path)
                    predicted_class_idx = results[0].probs.top1
                    predicted_class = class_names[predicted_class_idx]
                    probs = results[0].probs.data.tolist()
                    
                    total += 1
                    
                    if predicted_class_idx == class_idx:
                        correct += 1
                        class_correct[class_name] += 1
                    
                    # Salvar predição no arquivo
                    f.write(f"\nImagem: {os.path.basename(img_path)}\n")
                    f.write(f"Predição: {predicted_class}\n")
                    f.write("Probabilidades:\n")
                    for i, (prob, cls) in enumerate(zip(probs, class_names)):
                        f.write(f"- {cls}: {prob*100:.2f}%\n")
                    f.write("-" * 30 + "\n")
    
    overall_accuracy = correct / total if total > 0 else 0
    print(f"\nAcurácia geral: {overall_accuracy:.4f} ({correct}/{total})")
    
    print("\nAcurácia por classe:")
    for class_name in class_names:
        if class_total[class_name] > 0:
            class_accuracy = class_correct[class_name] / class_total[class_name]
            print(f"  {class_name}: {class_accuracy:.4f} ({class_correct[class_name]}/{class_total[class_name]})")

def main():
    class_names = ['angry', 'brake', 'distracted', 'excited', 'focus', 'mistake', 'tired']
    
    model_path = 'runs/classify/train/weights/best.pt'
    
    if not os.path.exists(model_path):
        print("Modelo não encontrado! Procurando em outras pastas...")
        
        possible_paths = [
            'runs/classify/train7/weights/best.pt',
            'runs/classify/train6/weights/best.pt',
            'runs/classify/train5/weights/best.pt',
            'runs/classify/train4/weights/best.pt',
            'runs/classify/train3/weights/best.pt',
            'runs/classify/train2/weights/best.pt'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                print(f"Modelo encontrado em: {model_path}")
                break
        else:
            print("Nenhum modelo treinado encontrado!")
            return
    
    print(f"Carregando modelo: {model_path}")
    model = YOLO(model_path)
    
    val_dir = 'yolo_dataset/val'
    if os.path.exists(val_dir):
        print("1. Avaliando no conjunto de validação")
        evaluate_on_validation_set(model, val_dir, class_names)
        
        print("\n2. Testando imagens aleatórias do conjunto de validação")
        test_random_images(model, val_dir, class_names, num_images=3)
        
        print("\n3. Gerando visualizações dos resultados")
        import visualize_results
        visualize_results.main()
    else:
        print("Conjunto de validação não encontrado!")
        print("Por favor, execute primeiro o script de treinamento (train.py) para criar o conjunto de validação.")
        return

if __name__ == "__main__":
    main() 