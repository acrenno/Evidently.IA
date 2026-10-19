# EvidentlyIA: testando o Evidently AI com preços de imóveis

Projeto de estudo para testar o [Evidently AI](https://www.evidentlyai.com/), uma biblioteca open source de avaliação e monitoramento de modelos de machine learning. A estrutura segue a de um projeto anterior de explicabilidade (XAI, com SHAP e LIME), mas aqui a etapa final troca a explicabilidade por **qualidade de dados, drift e desempenho do modelo**.

## Sumário

- [Objetivo](#objetivo)
- [Dataset](#dataset)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Pipeline](#pipeline)
- [Como simular drift](#como-simular-drift)
- [O que será testado no Evidently](#o-que-será-testado-no-evidently)
- [Como executar](#como-executar)
- [Decisões de limpeza](#decisões-de-limpeza)
- [Status](#status)
- [Observações](#observações)

## Objetivo

1. Treinar um modelo de regressão simples para prever o preço de venda de imóveis.
2. Usar o Evidently para avaliar:
   - a **qualidade dos dados**;
   - o **data drift** entre um período de referência e um período atual;
   - o **desempenho do modelo** em cada período;
   - **test suites** com resultado de aprovado/reprovado, simulando um monitoramento.
3. Entender o que a ferramenta entrega, onde ela ajuda e quais são seus limites.

O foco não é obter o melhor modelo, e sim ter um modelo razoável para monitorar.

## Dataset

**Ames Housing**, com vendas residenciais de Ames, Iowa (EUA), entre 2006 e 2010.

| Item | Valor |
|---|---|
| Arquivo | `ames_housing.csv` |
| Linhas | 2.930 (2.925 após a limpeza) |
| Colunas | 83 (80 após a limpeza) |
| Alvo | `SalePrice` (preço de venda) |
| Variável temporal | `Yr Sold` (2006 a 2010) e `Mo Sold` |

O dataset foi baixado do Kaggle e fica em `data/raw/`. A pasta `data/` não é versionada (veja o `.gitignore`), então é preciso baixar o arquivo manualmente antes de rodar os notebooks.

Vendas por ano (após a limpeza):

| Ano | Vendas |
|---|---|
| 2006 | 625 |
| 2007 | 690 |
| 2008 | 621 |
| 2009 | 648 |
| 2010 | 341 (o dataset cobre apenas parte do ano) |

## Estrutura do projeto

```
EVIDENTLYIA/
├── data/
│   ├── raw/                  # dado bruto, nunca alterado
│   └── processed/            # dado limpo (parquet)
├── notebooks/
│   ├── 1_data_cleaning.ipynb
│   ├── 2_eda.ipynb
│   ├── 3_modelling.ipynb
│   └── 4_evidently.ipynb
├── runs/                     # modelo e conjuntos treino/teste salvos
├── reports/                  # relatórios HTML gerados pelo Evidently
├── src/                      # funções reutilizáveis
│   └── __init__.py
├── .gitignore
├── README.md
└── requirements.txt
```

### O que cada parte faz

| Item | Função |
|---|---|
| `1_data_cleaning.ipynb` | Lê o dado bruto, trata nulos, corrige erros, remove outliers e salva a versão limpa. |
| `2_eda.ipynb` | Análise exploratória: distribuição do preço, correlações, outliers e diferenças entre anos. Ajuda a definir a divisão referência/atual. |
| `3_modelling.ipynb` | Cria as features finais, divide treino e teste, treina o modelo (LightGBM), avalia (MAE, RMSE, R²) e salva os artefatos. |
| `4_evidently.ipynb` | Carrega modelo e dados salvos, gera predições e roda os relatórios e testes do Evidently. |
| `src/` | Código Python reutilizável entre notebooks (gráficos, estatísticas, treino e métricas). |
| `runs/` | Modelo (`.pkl`) e `X_train`, `X_test`, `y_train`, `y_test` em parquet, para reaproveitar sem retreinar. |
| `reports/` | Relatórios HTML do Evidently. |

## Pipeline

```
data/raw  →  1_data_cleaning  →  data/processed  →  2_eda
                                        ↓
                                  3_modelling  →  runs/
                                        ↓
                                  4_evidently  →  reports/
```

Cada notebook lê a saída do anterior, então a ordem de execução importa.

## Como simular drift

O dataset não tem um cenário real de produção, então o drift é simulado dividindo os dados no tempo:

- **Referência:** vendas mais antigas, usadas para treinar o modelo (proposta inicial: 2006 a 2008).
- **Atual:** vendas mais recentes, que o modelo "veria" depois de implantado (proposta inicial: 2009 e 2010).

A divisão final será decidida no `2_eda.ipynb`, olhando como preço e características mudam entre os anos. Vale lembrar que 2010 tem poucas vendas, e que o período 2008 a 2009 coincide com a crise imobiliária, o que pode gerar drift real no preço.

## O que será testado no Evidently

- **Data quality:** nulos, duplicados, distribuições e valores fora do esperado.
- **Data drift:** comparação coluna a coluna entre referência e atual.
- **Desempenho do modelo de regressão:** erro por período, distribuição dos resíduos e análise de erros.
- **Test suites:** conjuntos de testes automáticos com aprovado/reprovado.

## Como executar

### 1. Criar o ambiente

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 2. Baixar o dataset

Baixe o `ames_housing.csv` (versão completa, com 2.930 linhas) e salve em `data/raw/`.

### 3. Rodar os notebooks na ordem

```
1_data_cleaning  →  2_eda  →  3_modelling  →  4_evidently
```

Os notebooks usam caminhos relativos (`../data/...`), então devem ser executados de dentro da pasta `notebooks/`.

### 4. Ver os relatórios

Abra os arquivos HTML de `reports/` no navegador.

## Decisões de limpeza

Resumo do que o `1_data_cleaning.ipynb` faz:

| Decisão | Motivo |
|---|---|
| Remover `Unnamed: 0`, `Order` e `PID` | São índice e identificadores, sem valor preditivo. |
| Nomes de colunas em snake_case | Evita problemas com espaços e barras (`Year Remod/Add` vira `year_remod_add`). |
| Vazios de "não possui" viram a categoria `None` | Em colunas como `pool_qc`, `alley` e as de garagem e porão, o vazio significa ausência do item, e não dado faltando. |
| Manter os NaN reais (ex.: `lot_frontage`) | O LightGBM aceita valores ausentes. Imputar é uma decisão de modelagem, feita depois da divisão treino/teste. |
| `garage_yr_blt` acima do ano de venda vira NaN | Havia o ano 2207, impossível. |
| `ms_subclass` vira texto | É um código de tipo de imóvel, e não uma quantidade. |
| Remover casas com `gr_liv_area` acima de 4000 pés² | Recomendação do autor do dataset: 5 vendas atípicas. |


## Observações

- **Versão do Evidently:** a API mudou bastante entre versões e muitos tutoriais antigos não funcionam nas mais recentes. Anote a versão instalada no `requirements.txt` (`pip freeze`) e consulte a documentação dessa versão.
- **Dados fora do Git:** `data/`, `runs/` e `reports/` estão no `.gitignore`, pois são pesados e reproduzíveis.
- **Referência do dataset:** De Cock, D. (2011). *Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project.* Journal of Statistics Education.