import sys
from ultralytics import YOLO
import os

def test_image(image_path, model_path=None):
    class_names = ['angry', 'brake', 'distracted', 'excited', 'focus', 'mistake', 'tired']
    
    if model_path is None:
        possible_paths = [
            'runs/classify/train/weights/best.pt',
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
                break
        else:
            print("Nenhum modelo encontrado!")
            return
    
    print(f"Carregando modelo: {model_path}")
    model = YOLO(model_path)
    
    print(f"Testando imagem: {image_path}")
    results = model(image_path)
    result = results[0]
    
    predicted_class_idx = result.probs.top1
    confidence = result.probs.top1conf.item()
    predicted_class = class_names[predicted_class_idx]
    
    print(f"\nResultado:")
    print(f"Classe predita: {predicted_class}")
    print(f"Confiança: {confidence:.4f}")
    
    print(f"\nTodas as predições:")
    for i, (prob, class_name) in enumerate(zip(result.probs.data, class_names)):
        print(f"  {class_name}: {prob:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python test_single.py <caminho_da_imagem>")
        print("Exemplo: python test_single.py frame/focus/alguns_subdir/imagem.jpg")
    else:
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"Imagem não encontrada: {image_path}")
        else:
            test_image(image_path) 