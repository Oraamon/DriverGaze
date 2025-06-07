import os
import shutil
import joblib
import numpy as np
from PIL import Image
import glob
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_image(image_path, target_size=(32, 32)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    return np.array(img).flatten()

def create_results_folder():
    results_dir = 'test_results'
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)
    
    class_names = sorted([d for d in os.listdir('frame') if os.path.isdir(os.path.join('frame', d))])
    for class_name in class_names:
        os.makedirs(os.path.join(results_dir, class_name))
    
    return results_dir, class_names

def copy_test_images_and_predict(results_dir, class_names, max_images_per_class=5):
    try:
        model = joblib.load('drivegaze_model_all_classes.joblib')
        scaler = joblib.load('drivegaze_scaler_all_classes.joblib')
    except:
        print("Erro: Modelo não encontrado. Execute primeiro o script de treinamento (train.py)")
        return

    results_file = open(os.path.join(results_dir, 'predictions.txt'), 'w', encoding='utf-8')
    results_file.write("Resultados das Predições\n")
    results_file.write("======================\n\n")

    for class_name in class_names:
        class_dir = os.path.join('frame', class_name)
        image_paths = []
        
        for subdir in os.listdir(class_dir):
            subdir_path = os.path.join(class_dir, subdir)
            if os.path.isdir(subdir_path):
                image_paths.extend(glob.glob(os.path.join(subdir_path, "*.jpg")))
        
        selected_images = np.random.choice(image_paths, min(max_images_per_class, len(image_paths)), replace=False)
        
        results_file.write(f"\nClasse Real: {class_name}\n")
        results_file.write("-" * 50 + "\n")
        
        for i, img_path in enumerate(selected_images):
            dest_path = os.path.join(results_dir, class_name, f'test_image_{i+1}.jpg')
            shutil.copy2(img_path, dest_path)
            
            features = load_and_preprocess_image(img_path)
            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
            
            results_file.write(f"\nImagem: test_image_{i+1}.jpg\n")
            results_file.write(f"Predição: {class_names[prediction]}\n")
            results_file.write("Probabilidades:\n")
            for j, prob in enumerate(probabilities):
                results_file.write(f"- {class_names[j]}: {prob:.2%}\n")
            results_file.write("-" * 30 + "\n")
    
    results_file.close()

def create_results_file(results_dir, class_names):
    with open(os.path.join(results_dir, 'test_info.txt'), 'w', encoding='utf-8') as f:
        f.write("Informações do Teste de Classificação de Emoções\n")
        f.write("=============================================\n\n")
        
        f.write("Classes utilizadas:\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{i+1}. {class_name}\n")
        
        f.write("\nEstrutura do Dataset:\n")
        for class_name in class_names:
            class_dir = os.path.join('frame', class_name)
            num_subdirs = len([d for d in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, d))])
            f.write(f"- {class_name}: {num_subdirs} subdiretórios\n")
        
        f.write("\nModelo utilizado: SVM com kernel RBF\n")
        f.write("Pré-processamento: Redimensionamento para 32x32 pixels\n")
        f.write("Normalização: StandardScaler\n")

def main():
    print("Criando estrutura de pastas para resultados...")
    results_dir, class_names = create_results_folder()
    
    print("Copiando imagens de teste e fazendo predições...")
    copy_test_images_and_predict(results_dir, class_names)
    
    print("Criando arquivo de informações...")
    create_results_file(results_dir, class_names)
    
    print(f"\nResultados organizados na pasta: {results_dir}")
    print("Cada subpasta contém imagens de teste da respectiva classe")
    print("O arquivo test_info.txt contém informações detalhadas sobre o teste")
    print("O arquivo predictions.txt contém as predições do modelo para cada imagem")

if __name__ == "__main__":
    main() 