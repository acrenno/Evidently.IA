# EvidentlyIA: Monitoramento de Data Drift e Desempenho com Evidently AI

Projeto de Ciência de Dados e Machine Learning focado na construção, avaliação e monitoramento contínuo de modelos preditivos utilizando o dataset **Ames Housing**.

O objetivo principal deste projeto é implementar uma pipeline completa de Machine Learning — da engenharia de dados ao treinamento de regressão com transformação logarítmica do *target* — e integrar a biblioteca **[Evidently AI](https://www.evidentlyai.com/)** para avaliar a qualidade dos dados, detectar *Data Drift* / *Target Drift* e executar suítes de testes automatizados (*PASS/FAIL*).

---

## 📋 Sumário

- [Visão Geral e Objetivos](#-visão-geral-e-objetivos)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Dataset](#-dataset)
- [Pipeline de Machine Learning](#-pipeline-de-machine-learning)
- [Simulação de Drift Temporal](#-simulação-de-drift-temporal)
- [Relatórios e Monitoramento (Evidently AI)](#-relatórios-e-monitoramento-evidently-ai)
- [Como Executar o Projeto](#-como-executar-o-projeto)
- [Decisões Técnicas de Pré-processamento](#-decisões-técnicas-de-pré-processamento)

---

## 🎯 Visão Geral e Objetivos

1. **Modelagem Preditiva Robusta:** Treinar um modelo de regressão para prever o preço de venda de imóveis (`saleprice`) utilizando transformação logarítmica (`log1p`) e **correção de viés estatística (log-normal)** na retransformação dos preços.
2. **Qualidade e Drift de Dados:** Analisar alterações nas distribuições das variáveis preditoras (*Data Drift*) e do preço real de venda (*Target Drift*) entre diferentes janelas temporais.
3. **Monitoramento Automatizado:** Gerar relatórios visuais interativos em HTML e implementar uma *Test Suite* automatizada que aprova ou reprova a qualidade do lote de dados recente antes do *deploy* ou re-treinamento.

---

## 📂 Estrutura do Repositório

```text
envidelnty/
├── README.md                          # Documentação principal do projeto
├── requirements.txt                   # Dependências do ambiente Python
├── .gitignore                         # Regras de exclusão do controle de versão
├── data/
│   ├── raw/
│   │   └── ames_housing.csv          # Dataset bruto original (Kaggle)
│   └── processed/
│       └── ames_clean.parquet        # Dataset limpo pós-processamento
├── models/
│   ├── pipeline_log.joblib            # Pipeline treinada (ColumnTransformer + LightGBM/RandomForest)
│   └── metrics.json                   # Métricas baseline de erro e desvio dos resíduos
├── notebooks/
│   ├── data_cleaning.ipynb            # Notebook 1: Limpeza, remoção de outliers e imputação inicial
│   ├── eda.ipynb                      # Notebook 2: Análise Exploratória de Dados
│   ├── modelling.ipynb                # Notebook 3: Treinamento, log-target e bias correction
│   └── evidently.ipynb                # Notebook 4: Avaliação de Drift e Test Suites com Evidently AI
└── runs/
    ├── evidently_report.html          # Relatório interativo de Drift e Regressão (HTML)
    └── evidently_tests.html           # Painel de Testes Automatizados (PASS / FAIL)
```

---

## 📊 Dataset

O projeto utiliza o dataset **Ames Housing**, que contém o histórico de vendas de imóveis residenciais em Ames, Iowa (EUA), comercializados entre 2006 e 2010.

| Item | Descrição / Valor |
|---|---|
| **Arquivo Bruto** | `data/raw/ames_housing.csv` (2.930 linhas, 83 colunas) |
| **Dataset Processado** | `data/processed/ames_clean.parquet` (2.925 linhas, 80 colunas) |
| **Variável Alvo (`target`)** | `saleprice` (Preço final de venda do imóvel em USD) |
| **Variáveis Temporais** | `yr_sold` (Anos de 2006 a 2010) e `mo_sold` (Mês da venda) |

### Distribuição de Vendas por Ano (Dataset Limpo)
- **2006:** 625 imóveis
- **2007:** 690 imóveis
- **2008:** 621 imóveis
- **2009:** 648 imóveis
- **2010:** 341 imóveis *(dados parciais do ano)*

---

## ⚙️ Pipeline de Machine Learning

O fluxo de dados e artefatos segue uma execução estritamente ordenada:

```text
data/raw/  ──>  data_cleaning.ipynb  ──>  data/processed/  ──>  eda.ipynb
                                                 │
                                                 ▼
                                          modelling.ipynb  ──>  models/
                                                 │
                                                 ▼
                                          evidently.ipynb  ──>  runs/
```

### Detalhes das Etapas:
* **`data_cleaning.ipynb`:** Padroniza nomes de colunas para *snake_case*, trata valores ausentes categóricos como a categoria `'None'` (ex: ausência de garagem/piscina) e remove *outliers* de área habitável (`gr_liv_area` > 4000 sq ft).
* **`eda.ipynb`:** Explora assimetrias na distribuição de preços, correlações lineares/não-lineares e analisa a sazonalidade de vendas ao longo dos anos.
* **`modelling.ipynb`:**
  * Define $y_{\text{log}} = \log(1 + \text{saleprice})$ para estabilizar a variância dos resíduos e suavizar *outliers*.
  * Monta a pipeline do `scikit-learn` com `ColumnTransformer` (`SimpleImputer` + `StandardScaler` para numéricas; `OneHotEncoder` para categóricas).
  * Treina o algoritmo de regressão (**LightGBM** ou **RandomForest**).
  * Aplica a **Correção de Viés Log-Normal** no resgate do preço em Dólares:
    $$\text{Pred}_{\text{corrigido}} = \exp\left(\text{pred}_{\text{log}} + 0,5 \cdot \sigma_{\text{resid}}^2\right) - 1$$
  * Salva a pipeline em `models/pipeline_log.joblib` e métricas em `models/metrics.json`.

---

## 🕒 Simulação de Drift Temporal

Para avaliar o comportamento do modelo em produção, dividimos os dados temporalmente:
* **Base de Referência (`ref_data`):** Imóveis vendidos entre **2006 e 2009** (massa histórica utilizada para treino e linha de base).
* **Base Atual (`cur_data`):** Imóveis vendidos em **2010** (simula um novo lote de produção recebido pelo modelo).

Essa divisão permite detectar o impacto da variação de mercado no desempenho preditivo e verificar se o perfil das casas vendidas em 2010 sofreu alterações em relação ao período de 2006–2009 (período que abrange a crise imobiliária de 2008).

---

## 📈 Relatórios e Monitoramento (Evidently AI)

O notebook **`evidently.ipynb`** carrega o modelo salvo e gera os seguintes relatórios em `runs/`:

1. **`evidently_report.html` (Relatório Visual Interativo):**
   * **Data Drift:** Compara a distribuição de cada uma das colunas entre o período de Referência (2006–2009) e o Atual (2010).
   * **Target Drift:** Avalia se a distribuição do preço real de venda mudou significativamente.
   * **Regression Performance:** Exibe gráficos de resíduos, erro absoluto médio (MAE), erro quadrático médio (RMSE) e coeficiente de determinação ($R^2$) na escala original em Dólares.

2. **`evidently_tests.html` (Test Suite Automatizada):**
   * Avalia regras estritas de produção (ex.: alerta se mais de 5 colunas apresentarem alteração drástica de distribuição) e retorna status de aprovação (**PASS**) ou falha (**FAIL**).

---

## 🚀 Como Executar o Projeto

### 1. Criar o ambiente virtual e instalar dependências

```bash
# Cria o ambiente virtual
python -m venv .venv

# Ativa o ambiente virtual
# No Linux/macOS:
source .venv/bin/activate
# No Windows:
# .venv\Scripts\activate

# Instala as dependências do projeto
pip install -r requirements.txt
```

### 2. Executar os notebooks na ordem
Acesse a pasta `notebooks/` e rode as células de cada notebook na sequência abaixo:

1. `data_cleaning.ipynb`
2. `eda.ipynb`
3. `modelling.ipynb`
4. `evidently.ipynb`

### 3. Abrir os relatórios HTML
Após a execução do `evidently.ipynb`, abra os arquivos `evidently_report.html` e `evidently_tests.html` salvos na pasta `runs/` em qualquer navegador web (Chrome, Edge, Firefox).

---

## 📌 Decisões Técnicas de Pré-processamento

| Ação | Motivo Técnico |
|---|---|
| **Remoção de Identificadores** (`PID`, `Order`) | Variáveis de índice sem poder preditivo para o preço. |
| **Imputação de `'None'` em Categóricas** | A ausência de valor em colunas como `pool_qc` ou `garage_type` representa a não existência do item na casa, e não uma falha na coleta do dado. |
| **Filtro de Outliers (`gr_liv_area` > 4000 sq ft)** | Remoção recomendada pelo autor do dataset (*Dean De Cock*) devido a vendas atípicas de grandes propriedades com preços abaixo do mercado. |
| **`ms_subclass` tratada como Categórica** | Trata-se de um código identificador do tipo de residência, e não de uma grandeza numérica contínua. |