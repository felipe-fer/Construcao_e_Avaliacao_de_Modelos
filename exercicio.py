from pathlib import Path
import sys
import matplotlib

if "--sem-janelas" in sys.argv:
    matplotlib.use("Agg")

PASTA = Path(__file__).resolve().parent
SAIDA = PASTA / "resultados"
SAIDA.mkdir(exist_ok=True)

print("\n=== 1 Carregar os dados ===")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

dados = pd.read_csv(PASTA / "vendas_loja.csv")
print(dados.head())
print("Linhas e colunas:", dados.shape)
print("Ausentes por coluna:")
print(dados.isna().sum())

print("\n=== 2 Separar os períodos ===")
treino = dados.iloc[:40].copy()
validacao = dados.iloc[40:50].copy()
teste = dados.iloc[50:].copy()

print("Treino:", len(treino))
print("Validação:", len(validacao))
print("Teste:", len(teste))

plt.scatter(treino["publicidade"], treino["vendas"])
plt.xlabel("Publicidade em reais")
plt.ylabel("Vendas em unidades")
plt.title("Dados de treinamento")
plt.savefig(SAIDA / "01_dados_treino.png", dpi=150, bbox_inches="tight")
if "--sem-janelas" not in sys.argv:
    plt.show()
plt.close()

print("\n=== 3 Treinar o modelo A ===")
colunas_a = ["publicidade"]
modelo_a = LinearRegression()
modelo_a.fit(treino[colunas_a], treino["vendas"])

print("Valor inicial:", modelo_a.intercept_)
print("Peso da publicidade:", modelo_a.coef_[0])

print("\n=== 4 Treinar o modelo B ===")
colunas_b = ["publicidade", "desconto"]
modelo_b = LinearRegression()
modelo_b.fit(treino[colunas_b], treino["vendas"])

print("Valor inicial:", modelo_b.intercept_)
print("Pesos das entradas:")
print(pd.Series(modelo_b.coef_, index=colunas_b))

print("\n=== 5 Gerar previsões comparáveis ===")
previsao_a = modelo_a.predict(validacao[colunas_a])
previsao_b = modelo_b.predict(validacao[colunas_b])

comparacao = validacao[["dia", "vendas"]].copy()
comparacao["previsao_a"] = previsao_a
comparacao["previsao_b"] = previsao_b
print(comparacao.round(2).to_string(index=False))

print("\n=== 6 Calcular os erros e a média ===")
comparacao["erro_a"] = previsao_a - validacao["vendas"]
comparacao["erro_b"] = previsao_b - validacao["vendas"]
comparacao["abs_a"] = comparacao["erro_a"].abs()
comparacao["abs_b"] = comparacao["erro_b"].abs()

mae_a = mean_absolute_error(validacao["vendas"], previsao_a)
mae_b = mean_absolute_error(validacao["vendas"], previsao_b)
print(comparacao.round(2).to_string(index=False))
print(f"Erro absoluto médio A: {mae_a:.2f} unidades")
print(f"Erro absoluto médio B: {mae_b:.2f} unidades")
print("Conferência pela média:")
print(comparacao[["abs_a", "abs_b"]].mean())

print("\n=== 7 Visualizar e escolher ===")
grafico = comparacao.set_index("dia")
grafico[["vendas", "previsao_a", "previsao_b"]].plot(
    marker="o", figsize=(9, 4)
)
plt.xlabel("Dia")
plt.ylabel("Unidades")
plt.title("Vendas reais e previsões na validação")
plt.tight_layout()
plt.savefig(SAIDA / "02_comparacao_validacao.png", dpi=150, bbox_inches="tight")
if "--sem-janelas" not in sys.argv:
    plt.show()
plt.close()

if mae_a <= mae_b:
    escolhido = modelo_a
    colunas = colunas_a
    nome = "A"
else:
    escolhido = modelo_b
    colunas = colunas_b
    nome = "B"

print("Modelo escolhido na validação:", nome)

print("\n=== 8 Avaliar o modelo escolhido no teste ===")
previsao_teste = escolhido.predict(teste[colunas])
mae_teste = mean_absolute_error(teste["vendas"], previsao_teste)

resultado_teste = teste[["dia", "vendas"]].copy()
resultado_teste["previsao"] = previsao_teste
print(resultado_teste.round(2).to_string(index=False))
print(f"Modelo {nome} no teste: {mae_teste:.2f} unidades")

print("\n=== 9 Fazer uma nova previsão e salvar ===")
novo_dia = pd.DataFrame({
    "publicidade": [300],
    "desconto": [10]
})
nova_previsao = escolhido.predict(novo_dia[colunas])[0]
print(f"Venda estimada: {nova_previsao:.2f} unidades")

comparacao.to_csv(SAIDA / "comparacao_validacao.csv", index=False)
resultado_teste.to_csv(SAIDA / "resultado_teste.csv", index=False)
