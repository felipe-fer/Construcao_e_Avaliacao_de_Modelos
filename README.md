# Construção e Avaliação de Modelos

Este projeto tem como objetivo avaliar dois modelos de regressão linear para prever vendas de uma loja a partir de dados históricos de publicidade e desconto. O código compara o desempenho dos modelos em um conjunto de validação, escolhe o melhor entre eles e mede o erro final em um conjunto de teste.

## Objetivo

O script realiza as seguintes etapas:

1. Carrega os dados de vendas de um arquivo CSV;
2. Divide os dados em treino, validação e teste;
3. Treina dois modelos de regressão:
   - Modelo A: usa apenas a variável `publicidade`;
   - Modelo B: usa `publicidade` e `desconto`;
4. Compara as previsões na validação usando erro absoluto médio (MAE);
5. Escolhe o melhor modelo com base no conjunto de validação;
6. Avalia o modelo escolhido no conjunto de teste;
7. Gera gráficos e arquivos CSV com os resultados.

## Estrutura do projeto

- `exercicio.py`: script principal com todo o fluxo do projeto;
- `vendas_loja.csv`: dados históricos utilizados para treino, validação e teste;
- `requirements.txt`: dependências do projeto;
- `resultados/`: arquivos gerados pelo programa, como gráficos e CSVs.

## Dependências

O projeto utiliza as bibliotecas abaixo:

- pandas
- matplotlib
- scikit-learn

Essas dependências podem ser instaladas com:

```bash
pip install -r requirements.txt
```

## Como executar

Na raiz do projeto, execute:

```bash
python exercicio.py
```

Se quiser executar sem abrir janelas de visualização (ideal para ambientes sem interface gráfica), use:

```bash
python exercicio.py --sem-janelas
```

## Saída gerada

Ao rodar o script, o programa salva os seguintes artefatos em `resultados/`:

- `01_dados_treino.png`: dispersão dos dados de treinamento
- `02_comparacao_validacao.png`: comparação entre vendas reais e previsões na validação
- `comparacao_validacao.csv`: métricas e previsões do conjunto de validação
- `resultado_teste.csv`: previsões e desempenho no conjunto de teste

## Observações

- O modelo escolhido é definido pela menor média de erro absoluto na fase de validação.
- O script também gera uma previsão para um novo dia fictício com valores fixos de publicidade e desconto.
- O projeto é um exemplo didático de construção e avaliação de modelos preditivos com regressão linear.

## Autor

Projeto desenvolvido como atividade de aprendizado em Machine Learning.
