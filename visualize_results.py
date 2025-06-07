import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import re
import pandas as pd
from collections import defaultdict
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

def parse_predictions_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    sections = content.split('Classe Real: ')[1:]
    
    y_true = []
    y_pred = []
    confidences = []
    class_probs = defaultdict(list)
    
    for section in sections:
        lines = section.strip().split('\n')
        true_class = lines[0]
        
        current_predictions = []
        i = 1
        while i < len(lines):
            if lines[i].startswith('Imagem:'):
                pred_line = lines[i+1]
                pred_class = pred_line.split(': ')[1]
                
                y_true.append(true_class)
                y_pred.append(pred_class)
                
                prob_section = []
                j = i + 3
                while j < len(lines) and lines[j].startswith('- '):
                    prob_line = lines[j]
                    class_name = prob_line.split(': ')[0][2:]
                    prob_value = float(prob_line.split(': ')[1].replace('%', '')) / 100
                    prob_section.append((class_name, prob_value))
                    j += 1
                
                class_probs[true_class].append(dict(prob_section))
                
                max_prob = max([prob for _, prob in prob_section])
                confidences.append(max_prob)
                
                i = j
            else:
                i += 1
    
    return y_true, y_pred, confidences, class_probs

def create_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusão', fontsize=16, fontweight='bold')
    plt.xlabel('Predição', fontsize=12)
    plt.ylabel('Classe Real', fontsize=12)
    plt.tight_layout()
    plt.savefig('test_results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_accuracy_by_class(y_true, y_pred, classes):
    accuracies = []
    for cls in classes:
        true_positives = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        total_class = sum(1 for t in y_true if t == cls)
        accuracy = true_positives / total_class if total_class > 0 else 0
        accuracies.append(accuracy * 100)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF'])
    plt.title('Acurácia por Classe', fontsize=16, fontweight='bold')
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Acurácia (%)', fontsize=12)
    plt.ylim(0, 100)
    
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('test_results/accuracy_by_class.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_confidence_distribution(confidences, y_true, y_pred):
    correct = [conf for conf, t, p in zip(confidences, y_true, y_pred) if t == p]
    incorrect = [conf for conf, t, p in zip(confidences, y_true, y_pred) if t != p]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(correct, bins=20, alpha=0.7, color='green', label=f'Corretas ({len(correct)})')
    plt.hist(incorrect, bins=20, alpha=0.7, color='red', label=f'Incorretas ({len(incorrect)})')
    plt.xlabel('Confiança da Predição')
    plt.ylabel('Frequência')
    plt.title('Distribuição de Confiança')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.boxplot([correct, incorrect], labels=['Corretas', 'Incorretas'])
    plt.ylabel('Confiança')
    plt.title('Comparação de Confiança')
    
    plt.tight_layout()
    plt.savefig('test_results/confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_performance_metrics(y_true, y_pred, classes):
    report = classification_report(y_true, y_pred, labels=classes, output_dict=True)
    
    metrics_df = pd.DataFrame(report).transpose()
    metrics_df = metrics_df.iloc[:-3]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics_df['precision'].plot(kind='bar', ax=axes[0,0], color='skyblue')
    axes[0,0].set_title('Precisão por Classe')
    axes[0,0].set_ylabel('Precisão')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    metrics_df['recall'].plot(kind='bar', ax=axes[0,1], color='lightcoral')
    axes[0,1].set_title('Recall por Classe')
    axes[0,1].set_ylabel('Recall')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    metrics_df['f1-score'].plot(kind='bar', ax=axes[1,0], color='lightgreen')
    axes[1,0].set_title('F1-Score por Classe')
    axes[1,0].set_ylabel('F1-Score')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    support_data = metrics_df['support'].astype(int)
    axes[1,1].bar(support_data.index, support_data.values, color='orange')
    axes[1,1].set_title('Suporte por Classe')
    axes[1,1].set_ylabel('Número de Amostras')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('test_results/performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_probability_heatmap(class_probs, classes):
    avg_probs = np.zeros((len(classes), len(classes)))
    
    for i, true_class in enumerate(classes):
        if true_class in class_probs:
            probs_list = class_probs[true_class]
            for j, pred_class in enumerate(classes):
                avg_prob = np.mean([probs.get(pred_class, 0) for probs in probs_list])
                avg_probs[i][j] = avg_prob
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_probs, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=classes, yticklabels=classes)
    plt.title('Mapa de Calor - Probabilidades Médias por Classe', fontsize=14)
    plt.xlabel('Classe Predita')
    plt.ylabel('Classe Real')
    plt.tight_layout()
    plt.savefig('test_results/probability_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_report(y_true, y_pred, classes, confidences):
    total_accuracy = accuracy_score(y_true, y_pred) * 100
    avg_confidence = np.mean(confidences) * 100
    
    correct_predictions = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    total_predictions = len(y_true)
    
    report_text = f"""
RELATÓRIO RESUMO DE PERFORMANCE
================================

📊 MÉTRICAS GERAIS:
• Acurácia Total: {total_accuracy:.2f}%
• Confiança Média: {avg_confidence:.2f}%
• Predições Corretas: {correct_predictions}/{total_predictions}

📈 PERFORMANCE POR CLASSE:
"""
    
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        total_cls = sum(1 for t in y_true if t == cls)
        accuracy_cls = (tp / total_cls * 100) if total_cls > 0 else 0
        report_text += f"• {cls.capitalize()}: {accuracy_cls:.1f}% ({tp}/{total_cls})\n"
    
    report_text += f"""
🎯 ANÁLISE:
• Melhor classe: {classes[np.argmax([sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)/sum(1 for t in y_true if t == cls) for cls in classes])]}
• Classe com mais dificuldade: {classes[np.argmin([sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)/sum(1 for t in y_true if t == cls) for cls in classes])]}

💡 MODELO: SVM com kernel RBF
📸 PRÉ-PROCESSAMENTO: Imagens 32x32 pixels
⚖️ NORMALIZAÇÃO: StandardScaler
"""

    with open('test_results/summary_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)

def main():
    print("🔍 Analisando resultados das predições...")
    
    y_true, y_pred, confidences, class_probs = parse_predictions_file('test_results/predictions.txt')
    classes = sorted(list(set(y_true)))
    
    print("📊 Gerando visualizações...")
    
    create_confusion_matrix(y_true, y_pred, classes)
    create_accuracy_by_class(y_true, y_pred, classes)
    create_confidence_distribution(confidences, y_true, y_pred)
    create_performance_metrics(y_true, y_pred, classes)
    create_probability_heatmap(class_probs, classes)
    create_summary_report(y_true, y_pred, classes, confidences)
    
    print("✅ Visualizações salvas na pasta test_results/")
    print("📁 Arquivos gerados:")
    print("  • confusion_matrix.png")
    print("  • accuracy_by_class.png") 
    print("  • confidence_distribution.png")
    print("  • performance_metrics.png")
    print("  • probability_heatmap.png")
    print("  • summary_report.txt")

if __name__ == "__main__":
    main() 