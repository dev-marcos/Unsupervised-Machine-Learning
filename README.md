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
Análise de componentes principais (PCA) é uma técnica de análise multivariada que é usada para reduzir a dimensionalidade de dados com muitas variáveis, ao mesmo tempo que mantém a maior quantidade possível de informações. Ela busca transformar as variáveis originais em um conjunto menor de variáveis lineares, denominadas componentes principais, que capturam a maior parte da variabilidade dos dados.

Por exemplo, suponha que temos um conjunto de dados com 10 variáveis diferentes. Cada variável representa uma característica ou medida diferente de algum objeto. Quando olhamos para essas 10 variáveis juntas, pode ser difícil interpretar a relação entre elas ou identificar padrões nos dados. PCA nos permite simplificar a estrutura dos dados e explorar como as variáveis se relacionam umas com as outras de forma mais fácil.

Em Python, podemos realizar a análise de componentes principais usando a biblioteca scikit-learn. A seguir, mostramos um exemplo simples de como aplicar PCA em um conjunto de dados de flores usando a biblioteca scikit-learn:

```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Carregar o conjunto de dados de flores
iris = load_iris()
X = iris.data
y = iris.target

# Aplicar PCA com dois componentes principais
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualizar os dados em um gráfico de dispersão
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()
```
Neste exemplo, carregamos o conjunto de dados de flores e aplicamos PCA com dois componentes principais usando a função PCA do scikit-learn. Em seguida, transformamos os dados originais em um novo conjunto de dados com apenas duas variáveis, que representam as duas primeiras componentes principais.

Finalmente, visualizamos os dados transformados em um gráfico de dispersão, onde cada ponto representa uma flor e sua cor indica sua classe (setosa, versicolor ou virginica). Podemos ver que as diferentes classes de flores estão bem separadas, o que sugere que as duas primeiras componentes principais capturam a maior parte da variabilidade dos dados.

Em resumo, a análise de componentes principais é uma técnica útil para reduzir a dimensionalidade de conjuntos de dados com muitas variáveis, permitindo a visualização e análise mais simples dos dados. O exemplo acima mostra como aplicar PCA em Python usando a biblioteca scikit-learn para explorar um conjunto de dados de flores.


## Análise de agrupamento hierárquico (HCA)
A HCA é uma técnica de clustering que constrói uma hierarquia de grupos com base na dissimilaridade entre os pontos de dados. Isso permite uma compreensão mais profunda da estrutura dos dados.

## Redução de dimensionalidade baseada em mapas auto-organizáveis (SOMs)
Os SOMs são uma técnica de aprendizado não supervisionado que busca visualizar e agrupar dados de alta dimensão em um espaço bidimensional. Isso pode ser útil para a análise exploratória de dados.

## Modelos de mistura de distribuições (GMM)
O GMM é uma técnica de clustering probabilístico que modela cada cluster como uma mistura de distribuições. Isso permite uma classificação mais suave e flexível dos dados.

