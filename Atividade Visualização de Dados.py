import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
file_path = '/ecommerce_preparados.csv'
data = pd.read_csv(file_path)

# Configuração de estilo
plt.style.use('ggplot')

# 1. Gráfico de Histograma: Distribuição de Notas
plt.figure(figsize=(10, 6))
plt.hist(data['Nota'], bins=15, edgecolor='black', alpha=0.7)
plt.title('Distribuição de Notas dos Produtos')
plt.xlabel('Nota')
plt.ylabel('Frequência')
plt.show()

# 2. Gráfico de Dispersão: Nota vs Número de Avaliações
plt.figure(figsize=(10, 6))
plt.scatter(data['Nota'], data['N_Avaliações'], alpha=0.5)
plt.title('Dispersão: Nota vs Número de Avaliações')
plt.xlabel('Nota')
plt.ylabel('Número de Avaliações')
plt.show()

# 3. Mapa de Calor: Correlação entre variáveis numéricas
plt.figure(figsize=(12, 8))
corr = data.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Mapa de Calor: Correlação entre Variáveis')
plt.show()

# 4. Gráfico de Barra: Frequência por Material
plt.figure(figsize=(12, 6))
data['Nota'].value_counts().plot(kind='bar')
plt.title('Frequência das Notas dos Produtos')
plt.xlabel('Nota')
plt.ylabel('Frequência')
plt.show()

# 5. Gráfico de Pizza: Proporção de Gêneros
gender_counts = data['Gênero'].value_counts()
plt.figure(figsize=(8, 8))
colors = sns.color_palette('pastel')
plt.pie(gender_counts, labels=None, autopct='%1.1f%%', colors=colors)
plt.legend(gender_counts.index, title="Gênero", loc="center left", bbox_to_anchor=(1, 0.5))
plt.title('Distribuição de Produtos por Gênero')
plt.show()

# 6. Gráfico de Densidade: Distribuição de Notas
valid_notas = pd.to_numeric(data['Nota'], errors='coerce').dropna()
plt.figure(figsize=(10, 6))
sns.kdeplot(valid_notas, shade=True, bw_adjust=0.5)
plt.title('Distribuição de Densidade das Notas')
plt.xlabel('Nota')
plt.ylabel('Densidade')
plt.show()

# 7. Gráfico de Regressão: Nota vs Preço (apenas registros válidos)
valid_data = data.dropna(subset=['Nota', 'Preço_MinMax'])
plt.figure(figsize=(10, 6))
sns.regplot(x=valid_data['Nota'], y=valid_data['Preço_MinMax'], scatter_kws={"alpha": 0.3})
plt.title('Regressão: Nota vs Preço (Normalizado)')
plt.xlabel('Nota')
plt.ylabel('Preço Normalizado')
plt.show()
