# Unsupervised-Machine-Learning


A classificação não supervisionada em aprendizado de máquina é uma técnica importante que permite explorar e compreender dados sem ter etiquetas prévias. Ao contrário da classificação supervisionada, que tem como objetivo prever uma variável dependente com base em variáveis independentes, a classificação não supervisionada procura encontrar padrões e relações nos dados de entrada. Essa técnica é amplamente utilizada em aplicações como análise de mercado, classificação de imagens e processamento de linguagem natural.

Entre os principais métodos de classificação não supervisionada, podemos destacar:

## Clustering

O objetivo do clustering é agrupar dados semelhantes em grupos (ou clusters) com base nas suas características. Existem vários algoritmos de clustering, como K-Means, Hierarchical Clustering e DBSCAN, cada um com suas próprias vantagens e desvantagens.


Em Python, o método mais comum de clustering é o K-Means. Aqui está um exemplo básico de como implementá-lo usando a biblioteca scikit-learn:

``` 
from sklearn.cluster import KMeans
import numpy as np

# Criação dos dados de exemplo
np.random.seed(0)
X = np.random.randn(100, 2)

# Instanciação do modelo KMeans
kmeans = KMeans(n_clusters=3)

# Ajuste do modelo aos dados
kmeans.fit(X)

# Previsão dos clusters para cada ponto de dados
labels = kmeans.predict(X)

# Exibição dos resultados
print(labels)
```

Neste exemplo, criamos 100 pontos de dados aleatórios em duas dimensões e agrupamos eles em três clusters usando o KMeans. Em seguida, previmos a qual cluster cada ponto de dados pertence e exibimos os rótulos (labels) obtidos.

O processo de clustering é útil quando desejamos descobrir padrões ou grupos de similaridade entre os dados. Alguns exemplos de aplicações incluem segmentação de clientes, análise de imagens e agrupamento de documentos.

Identifica a distância entre observações


Distancia Minkowski:

$$ d_{pq} = [\sum_{j=1}^{k}(|ZX_{jp}-ZX_{jq}|)^m]^\frac{1}{m} $$




Distancia euclidiana: 


$$ d_{pq} = \sqrt{\sum_{j=1}^{k}(ZX_{jp}-ZX_{jq})^2} $$


Distancia euclidiana quadrática: 


$$ d_{pq} = \sum_{j=1}^{k}(ZX_{jp}-ZX_{jq})^2 $$

## Análise de componentes principais (PCA)
A PCA é uma técnica de redução de dimensionalidade que busca encontrar os principais padrões de variação nos dados. Isso pode ser útil para visualizar e compreender dados de alta dimensionalidade.

Análise de Componentes Principais (PCA) é um dos métodos estatísticos de múltiplas variáveis mais simples. A PCA é considerada a WUDQVIRUPDomR OLQHDU yWLPD, dentre as transformadas de imagens, sendo muito utilizada pela comunidade de reconhecimento de padrões.

## Análise de agrupamento hierárquico (HCA)
A HCA é uma técnica de clustering que constrói uma hierarquia de grupos com base na dissimilaridade entre os pontos de dados. Isso permite uma compreensão mais profunda da estrutura dos dados.

## Redução de dimensionalidade baseada em mapas auto-organizáveis (SOMs)
Os SOMs são uma técnica de aprendizado não supervisionado que busca visualizar e agrupar dados de alta dimensão em um espaço bidimensional. Isso pode ser útil para a análise exploratória de dados.

## Modelos de mistura de distribuições (GMM)
O GMM é uma técnica de clustering probabilístico que modela cada cluster como uma mistura de distribuições. Isso permite uma classificação mais suave e flexível dos dados.

