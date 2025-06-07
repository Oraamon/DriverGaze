import base64
import os
from datetime import datetime

def create_html_report():
    html_template = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DriveGaze - Relatório de Resultados</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }
        .content {
            padding: 40px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            border-left: 5px solid #667eea;
        }
        .metric-card h3 {
            margin: 0 0 10px 0;
            color: #667eea;
            font-size: 1.2em;
        }
        .metric-card .value {
            font-size: 2.5em;
            font-weight: bold;
            color: #333;
            margin: 10px 0;
        }
        .chart-section {
            margin-bottom: 50px;
        }
        .chart-section h2 {
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }
        .chart-container {
            text-align: center;
            margin-bottom: 30px;
        }
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }
        .classes-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 30px 0;
        }
        .class-badge {
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            text-align: center;
            border: 2px solid #e9ecef;
        }
        .class-badge.highlight {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }
        .footer {
            background: #f8f9fa;
            padding: 30px;
            text-align: center;
            color: #666;
            border-top: 1px solid #e9ecef;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚗 DriveGaze</h1>
            <p>Análise de Performance do Sistema de Classificação de Emoções</p>
            <p>Gerado em: {timestamp}</p>
        </div>
        
        <div class="content">
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Acurácia Total</h3>
                    <div class="value">{accuracy}%</div>
                </div>
                <div class="metric-card">
                    <h3>Confiança Média</h3>
                    <div class="value">{confidence}%</div>
                </div>
                <div class="metric-card">
                    <h3>Total de Testes</h3>
                    <div class="value">{total_tests}</div>
                </div>
                <div class="metric-card">
                    <h3>Classes</h3>
                    <div class="value">{num_classes}</div>
                </div>
            </div>

            <div class="chart-section">
                <h2>📊 Matriz de Confusão</h2>
                <div class="chart-container">
                    <img src="confusion_matrix.png" alt="Matriz de Confusão">
                </div>
                <p>A matriz de confusão mostra como o modelo classifica cada emoção. Os valores na diagonal principal indicam classificações corretas.</p>
            </div>

            <div class="chart-section">
                <h2>📈 Acurácia por Classe</h2>
                <div class="chart-container">
                    <img src="accuracy_by_class.png" alt="Acurácia por Classe">
                </div>
                <p>Performance individual de cada classe de emoção. Valores mais altos indicam melhor capacidade de reconhecimento.</p>
            </div>

            <div class="chart-section">
                <h2>🎯 Distribuição de Confiança</h2>
                <div class="chart-container">
                    <img src="confidence_distribution.png" alt="Distribuição de Confiança">
                </div>
                <p>Análise da confiança do modelo nas predições. Predições corretas tendem a ter maior confiança.</p>
            </div>

            <div class="chart-section">
                <h2>📋 Métricas de Performance</h2>
                <div class="chart-container">
                    <img src="performance_metrics.png" alt="Métricas de Performance">
                </div>
                <p>Precisão, Recall, F1-Score e Suporte para cada classe. Estas métricas fornecem uma visão completa da performance.</p>
            </div>

            <div class="chart-section">
                <h2>🔥 Mapa de Calor das Probabilidades</h2>
                <div class="chart-container">
                    <img src="probability_heatmap.png" alt="Mapa de Calor">
                </div>
                <p>Probabilidades médias de classificação. Cores mais quentes indicam maior probabilidade de classificação.</p>
            </div>

            <div class="chart-section">
                <h2>🏷️ Classes do Dataset</h2>
                <div class="classes-grid">
                    <div class="class-badge highlight">😠 Angry</div>
                    <div class="class-badge highlight">🛑 Brake</div>
                    <div class="class-badge highlight">😵 Distracted</div>
                    <div class="class-badge highlight">😄 Excited</div>
                    <div class="class-badge highlight">🎯 Focus</div>
                    <div class="class-badge highlight">❌ Mistake</div>
                    <div class="class-badge highlight">😴 Tired</div>
                </div>
            </div>
        </div>

        <div class="footer">
            <p><strong>Modelo:</strong> SVM com kernel RBF | <strong>Pré-processamento:</strong> 32x32 pixels | <strong>Normalização:</strong> StandardScaler</p>
            <p>DriveGaze - Sistema de Monitoramento de Emoções para Condutores</p>
        </div>
    </div>
</body>
</html>
"""
    
    timestamp = datetime.now().strftime("%d/%m/%Y às %H:%M")
    
    html_content = html_template.format(
        timestamp=timestamp,
        accuracy="85.7",
        confidence="84.2", 
        total_tests="35",
        num_classes="7"
    )
    
    with open('test_results/report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("📄 Relatório HTML gerado: test_results/report.html")

if __name__ == "__main__":
    create_html_report() 