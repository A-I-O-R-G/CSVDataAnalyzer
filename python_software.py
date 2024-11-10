import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class CSVDataAnalyzer:
    def __init__(self, file_path):
        """
        Inicializa o analisador de dados CSV com o caminho do arquivo.
        :param file_path: Caminho do arquivo CSV a ser analisado.
        """
        self.file_path = file_path
        self.data = None

    def read_csv(self):
        """Lê o arquivo CSV e carregá-lo em um DataFrame."""
        try:
            self.data = pd.read_csv(self.file_path)
            print("Arquivo CSV lido com sucesso.")
        except Exception as e:
            print(f"Erro ao ler o arquivo CSV: {e}")

    def preprocess_data(self):
        """Processa os dados, removendo valores nulos e normalizando."""
        if self.data is not None:
            self.data.dropna(inplace=True)
            # Normalizando os dados (exemplo utilizando a normalização Min-Max)
            self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())
            print("Dados pré-processados com sucesso.")

    def basic_statistics(self):
        """Realiza análises estatísticas básicas nos dados."""
        if self.data is not None:
            statistics = {
                "mean": self.data.mean(),
                "median": self.data.median(),
                "mode": self.data.mode().iloc[0],  # Pega o primeiro modo
                "std_dev": self.data.std()
            }
            return statistics
        else:
            print("Os dados não foram lidos. Certifique-se de chamar read_csv() primeiro.")
            return None

    def generate_report(self):
        """Gera gráficos de distribuição dos dados."""
        if self.data is not None:
            plt.figure(figsize=(12, 6))
            sns.histplot(self.data, kde=True)
            plt.title('Distribuição dos Dados')
            plt.xlabel('Valores')
            plt.ylabel('Frequência')
            plt.show()
            print("Relatório de distribuição gerado com sucesso.")
        else:
            print("Os dados não estão disponíveis para gerar o relatório.")

if __name__ == "__main__":
    # Exemplo de uso do CSVDataAnalyzer
    analyzer = CSVDataAnalyzer("caminho/para/seu/arquivo.csv")
    analyzer.read_csv()
    analyzer.preprocess_data()
    stats = analyzer.basic_statistics()
    if stats:
        print("Estatísticas Básicas:")
        print(stats)
    analyzer.generate_report()