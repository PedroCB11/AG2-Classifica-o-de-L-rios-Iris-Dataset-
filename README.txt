README - AG2: Classificação de Lírios (Iris Dataset)
Autor: Pedro Consoli Bressan
Disciplina: AG002 – INATEL (2025)

======================================================
1. DESCRIÇÃO DO PROJETO
======================================================
Este projeto implementa um modelo de aprendizado de máquina para classificar
flores do tipo Iris em três espécies:
- Iris-setosa
- Iris-versicolor
- Iris-virginica

O modelo utiliza quatro atributos da flor (sépala e pétala) e aplica um
classificador Decision Tree para realizar a classificação.

======================================================
2. ARQUIVOS DO PROJETO
======================================================
ag2_iris.py      → Código principal
iris.csv         → Conjunto de dados
requirements.txt → Dependências
README.txt       → Este documento

======================================================
3. TECNOLOGIAS UTILIZADAS
======================================================
Python 3.11
Pandas
NumPy
Scikit-learn
Matplotlib

======================================================
4. FUNCIONALIDADES
======================================================
- Leitura do dataset Iris
- Conversão da coluna species para 1, 2 e 3
- Separação dos dados em treino e teste
- Treinamento com DecisionTreeClassifier
- Avaliação (acurácia, relatório e matriz de confusão)
- Função interativa para prever novas amostras
- Gráficos de análise:
  * Matriz de confusão visual
  * Distribuição das espécies
  * Gráfico 3D das amostras
  * Importância das features
  * Heatmap de correlação
  * Comparação Real x Previsto

======================================================
5. RESULTADOS
======================================================
Divisão dos dados: 60% treino / 40% teste
Acurácia final: 0.97

O modelo apresentou excelente desempenho, com poucos erros e boa separação
entre as espécies, especialmente entre Setosa e as demais.

======================================================
6. COMO EXECUTAR
======================================================
1) Instalar dependências:
   pip install -r requirements.txt

2) Executar o código:
   python ag2_iris.py


======================================================
7. OBSERVAÇÕES
======================================================
Durante o desenvolvimento houve incompatibilidade entre versões de numpy,
pandas e scikit-learn. O problema foi resolvido reinstalando versões
compatíveis.
