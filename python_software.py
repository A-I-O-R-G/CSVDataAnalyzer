import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class CSVDataAnalyzer:
    def __init__(self, file_path):
        """
        Inicializa o analisador de dados CSV com o caminho do arquivo.
        :param file_path: Caminho do arquivo CSV a ser analisado.
        """
        self.file_path = file_path
        self.data = None
        self.scaler = None

    def read_csv(self):
        """Lê o arquivo CSV e carrega em um DataFrame."""
        try:
            self.data = pd.read_csv(self.file_path)
            print("Arquivo CSV lido com sucesso.")
        except Exception as e:
            raise ValueError(f"Erro ao ler o arquivo CSV: {e}")

    def preprocess_data(self, method='minmax'):
        """Processa os dados, removendo valores nulos e normalizando."""
        if self.data is None:
            raise ValueError("Os dados não foram lidos. Certifique-se de chamar read_csv() primeiro.")
        
        self.data.dropna(inplace=True)
        if method == 'minmax':
            self.scaler = MinMaxScaler()
            self.data = self.scaler.fit_transform(self.data)
        elif method == 'standard':
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(self.data)
        else:
            raise ValueError(f"Método de normalização '{method}' não suportado.")

        print("Dados pré-processados com sucesso.")

    def basic_statistics(self):
        """Realiza análises estatísticas básicas nos dados."""
        if self.data is None:
            raise ValueError("Os dados não foram lidos. Certifique-se de chamar read_csv() primeiro.")
        
        statistics = {
            "mean": np.mean(self.data, axis=0),
            "median": np.median(self.data, axis=0),
            "mode": [pd.Series(self.data[:, i]).mode().iloc[0] for i in range(self.data.shape[1])],
            "std_dev": np.std(self.data, axis=0)
        }
        return statistics

    def generate_report(self):
        """Gera gráficos de distribuição dos dados."""
        if self.data is None:
            raise ValueError("Os dados não estão disponíveis para gerar o relatório.")

        plt.figure(figsize=(12, 6))
        for i in range(self.data.shape[1]):
            plt.subplot(1, self.data.shape[1], i + 1)
            sns.histplot(self.data[:, i], kde=True)
            plt.title(f'Distribuição da Variável {i + 1}')
            plt.xlabel('Valores')
            plt.ylabel('Frequência')
        plt.tight_layout()
        plt.show()
        print("Relatório de distribuição gerado com sucesso.")

    def save_report(self, filename='report.txt'):
        """Salva as estatísticas básicas em um arquivo de texto."""
        stats = self.basic_statistics()
        if stats:
            with open(filename, 'w') as f:
                for key, value in stats.items():
                    f.write(f'{key}: {value}\n')
            print(f"Relatório salvo em {filename}.")

if __name__ == "__main__":
    file_path = "caminho/para/seu/arquivo.csv"  # Substitua pelo caminho do seu arquivo
    analyzer = CSVDataAnalyzer(file_path)
    
    try:
        analyzer.read_csv()
        analyzer.preprocess_data(method='standard')
        stats = analyzer.basic_statistics()
        print("Estatísticas Básicas:")
        print(stats)
        analyzer.generate_report()
        analyzer.save_report('estatisticas.txt')
    except Exception as e:
        print(f"Erro: {e}")