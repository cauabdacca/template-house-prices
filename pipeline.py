import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_squared_log_error

def prever_precos(caminho_arquivo_teste):
    """
    Função obrigatória para o corretor automático.

    Lê o arquivo CSV de teste, aplica todo o pré-processamento via Pipeline
    treinado e retorna as predições de preço em dólares (escala original).

    Parâmetros:
        caminho_arquivo_teste (str): Caminho para o CSV de teste.

    Retorna:
        np.array: Predições de SalePrice em dólares, na mesma ordem das linhas do CSV.
    """
    # 1. Leitura
    df_teste = pd.read_csv(caminho_arquivo_teste)

    # 2. Remove coluna Id se presente (não é feature)
    if 'Id' in df_teste.columns:
        df_teste = df_teste.drop(columns=['Id'])

    # 3. Remove SalePrice se presente (ex: teste_publico com coluna alvo)
    if 'SalePrice' in df_teste.columns:
        df_teste = df_teste.drop(columns=['SalePrice'])

    # 4. Carrega o pipeline completo (pré-processamento + modelo)
    caminho_modelo = os.path.join(os.path.dirname(__file__), 'modelo.joblib')
    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError(
            f"Arquivo '{caminho_modelo}' não encontrado. "
            "Certifique-se de que modelo.joblib está na raiz do repositório."
        )
    pipeline = joblib.load(caminho_modelo)

    # 5. Predição — o pipeline já inclui todo o pré-processamento
    #    O modelo foi treinado com log1p(SalePrice), então revertemos com expm1
    predicoes_log = pipeline.predict(df_teste)
    predicoes = np.expm1(predicoes_log)

    # 6. Garante valores não negativos (segurança para o RMSLE)
    predicoes = np.clip(predicoes, a_min=0, a_max=None)

    return predicoes


if __name__ == "__main__":
    arquivo_teste = os.path.join(os.path.dirname(__file__), 'teste_publico.csv')

    print("─" * 50)
    print("  Validação Local do Pipeline")
    print("─" * 50)

    if not os.path.exists(arquivo_teste):
        print(f"[Aviso] '{arquivo_teste}' não encontrado.")
    else:
        try:
            resultados = prever_precos(arquivo_teste)

            print("✅ Pipeline executado com sucesso!")
            print(f"   Total de predições : {len(resultados)}")
            print(f"   Primeiras 5        : {np.round(resultados[:5], 2)}")
            print(f"   Mín / Máx          : ${resultados.min():,.0f} / ${resultados.max():,.0f}")
            print(f"   Média              : ${resultados.mean():,.0f}")

            # Calcula RMSLE se o arquivo tiver SalePrice
            df_val = pd.read_csv(arquivo_teste)
            if 'SalePrice' in df_val.columns:
                rmsle = np.sqrt(mean_squared_log_error(df_val['SalePrice'], resultados))
                print(f"\n   RMSLE local        : {rmsle:.5f}")
                print(f"   Baseline professor : 0.17543")
                if rmsle < 0.17543:
                    print(f"   ✅ Supera o baseline! ({rmsle:.5f} < 0.17543)")
                else:
                    print(f"   ⚠ Abaixo do baseline.")
            else:
                print("\n   [Nota] SalePrice não encontrado no CSV — RMSLE não calculado.")

        except Exception as e:
            print(f"❌ Erro no pipeline:\n{e}")

    print("─" * 50)
