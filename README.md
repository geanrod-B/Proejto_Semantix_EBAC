# Proejto_Semantix_EBAC

Objetivos do projeto
# 
# O projeto será desenvolvido ao longo do curso e seguirá o fluxo:
# 
# 1. Exploração e limpeza dos dados.
# 2. Análise exploratória e storytelling.
# 3. Identificação das variáveis mais relevantes.
# 4. Codificação da variável categórica.
# 5. Separação entre variáveis independentes e dependente.
# 6. Divisão em treino e teste.
# 7. Treinamento e avaliação de modelos.
# 8. Comparação entre dois modelos.
# 9. Visualização dos resultados.
# 10. Conclusão e recomendações para a problemática.
# 
# O modelo final será comparado entre **Árvore de Decisão e XGBoost**


#  Projeto de Predição de Propensão de Compra — Semantix / EBAC

> **Parceria:** Semantix + EBAC — Curso de Ciência de Dados  
> **Objetivo:** Identificar quais clientes têm maior propensão a comprar um automóvel, utilizando Machine Learning.

---

##  Sumário

- [Sobre o Projeto](#-sobre-o-projeto)
- [Dataset](#-dataset)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Etapas do Projeto](#-etapas-do-projeto)
- [Análise Exploratória — Principais Insights](#-análise-exploratória--principais-insights)
- [Modelagem](#-modelagem)
- [Resultados](#-resultados)
- [Importância das Variáveis](#-importância-das-variáveis)
- [Aplicação Prática](#-aplicação-prática)
- [Como Executar](#-como-executar)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Conclusão](#-conclusão)

---

##  Sobre o Projeto

Uma empresa do setor automotivo possui informações básicas sobre seus clientes e deseja utilizar dados para **identificar quais pessoas apresentam maior propensão a comprar um automóvel**.

Com um modelo preditivo, a empresa pode:
- Priorizar clientes com maior probabilidade de compra nas ações de marketing
- Reduzir gastos com campanhas direcionadas a perfis com baixa propensão
- Personalizar a abordagem comercial por faixa de probabilidade

---

##  Dataset

**Arquivo:** `CARRO_CLIENTES.csv`  
**Fonte:** Base fornecida pela EBAC (Escola Britânica de Artes Criativas e Tecnologia)

| Variável | Tipo | Descrição |
|----------|------|-----------|
| `User ID` | int | Identificador único do cliente *(removido da modelagem)* |
| `Gender` | str | Gênero do cliente (Male / Female) |
| `Age` | int | Idade do cliente |
| `AnnualSalary` | int | Renda anual estimada |
| `Purchased` | int | **Variável alvo** — 0 = Não comprou, 1 = Comprou |

- **Total de registros:** 1.000
- **Valores ausentes:** 0
- **Linhas duplicadas:** 0
- **Distribuição do alvo:** 59,8% não compraram (0) | 40,2% compraram (1) — desbalanceamento moderado

---

##  Tecnologias Utilizadas

| Biblioteca | Finalidade |
|------------|------------|
| `pandas` / `numpy` | Tratamento e análise dos dados |
| `matplotlib` / `seaborn` | Visualização e gráficos |
| `scikit-learn` | Pré-processamento, modelos e métricas |
| `xgboost` | Modelo Gradient Boosting |
| `LabelEncoder` | Codificação da variável categórica |
| `StandardScaler` | Padronização das variáveis numéricas |
| `train_test_split` | Divisão treino / teste |
| `GridSearchCV` + `StratifiedKFold` | Otimização de hiperparâmetros com validação cruzada |

---

##  Etapas do Projeto

1. **Carregamento e inspeção** dos dados
2. **Limpeza:** remoção de `User ID` (identificador sem valor preditivo)
3. **Codificação** da variável categórica `Gender` via `LabelEncoder`
4. **Análise exploratória (EDA):** distribuições, correlações e padrões
5. **Separação** entre variáveis preditoras e alvo
6. **Divisão** treino (80%) / teste (20%) com `stratify=y`
7. **Padronização** das variáveis numéricas (fit apenas no treino)
8. **Treinamento** de dois modelos com busca de hiperparâmetros
9. **Avaliação** comparativa por múltiplas métricas
10. **Interpretação** dos resultados e aplicação comercial

---

##  Análise Exploratória — Principais Insights

### Idade
- Quem comprou: **média de 48,2 anos**
- Quem não comprou: **média de 34,7 anos**
- **Correlação com a compra: 0,62** — variável mais relevante

### Renda Anual
- Quem comprou apresenta renda média superior aos não compradores
- **Correlação com a compra: 0,36**

### Gênero
- Diferença pequena na taxa de compra (Mulheres ~42,4% vs Homens ~37,8%)
- **Correlação com a compra: -0,05** — relação linear muito fraca

### Conclusão da EDA
`Age` e `AnnualSalary`, combinadas, separam claramente os grupos de compra no gráfico de dispersão, sugerindo que modelos baseados em árvores (que capturam relações não lineares) são adequados ao problema.

---

## 🤖 Modelagem

### Modelos treinados (com GridSearchCV + 5 folds estratificados)

| Modelo | Hiperparâmetros otimizados | Melhor score na CV |
|--------|---------------------------|--------------------|
| **Árvore de Decisão** | `max_depth=5`, `min_samples_leaf=5`, `min_samples_split=2` | 0,9025 |
| **XGBoost** | `n_estimators=100`, `max_depth=4`, `learning_rate=0.10`, `subsample=1.0` | 0,9125 |

---

##  Resultados (conjunto de teste — 200 amostras)

| Métrica | Árvore de Decisão | XGBoost |
|---------|-------------------|---------|
| **Acurácia** | **0,915** | 0,910 |
| **Precisão** | **0,932** | 0,888 |
| **Recall** | 0,850 | **0,888** |
| **F1-Score** | 0,889 | 0,888 |
| **ROC AUC** | 0,971 | **0,975** |

> Os dois modelos apresentam desempenho muito próximo e elevado. A Árvore de Decisão leva vantagem em acurácia e precisão, enquanto o XGBoost apresenta melhor recall e ROC AUC.

---

##  Importância das Variáveis

| Variável | Correlação com Purchased | Importância (Árvore) | Importância (XGBoost) |
|----------|--------------------------|----------------------|-----------------------|
| `Age` | 0,62 | 0,52 | 0,59 |
| `AnnualSalary` | 0,36 | 0,47 | 0,33 |
| `GenderEncoded` | -0,05 | 0,01 | 0,08 |

**Conclusão:** `Idade` e `Renda Anual` são, de longe, as variáveis mais relevantes para a predição. O gênero tem contribuição marginal, coerente com a análise exploratória.

---

##  Aplicação Prática

A probabilidade de compra gerada pelo modelo pode ser usada como **indicador de prioridade comercial**:

| Faixa de Probabilidade | Ação Recomendada |
|------------------------|------------------|
| **Muito alta (>75%)** | Abordagem comercial prioritária / contato direto |
| **Alta (50–75%)** | Campanhas personalizadas e ofertas direcionadas |
| **Média (25–50%)** | Nutrição de relacionamento / e-mail marketing |
| **Baixa (<25%)** | Comunicação genérica / baixo investimento |

>  O modelo é um **apoio à decisão**, não uma garantia de compra. Deve ser combinado com a expertise da equipe comercial.

---
