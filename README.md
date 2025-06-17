# Classificador de Notícias ❘ TF‑IDF → (LSA) → K‑NN

Projeto em **Python 3.11** que treina e avalia um modelo para diferenciar notícias verdadeiras de falsas usando representação TF‑IDF, redução de dimensionalidade com _Latent Semantic Analysis_ (Truncated SVD) e classificador _K‑Nearest Neighbors_ otimizado por _GridSearchCV_.

---

## Visão geral do conjunto de dados

- **Total de registros:** 18 898 artigos
- **Proporção de notícias verdadeiras:** 49,8 %
- **Arquivos de origem:** `data/True.csv`, `data/Fake.csv`

Esse equilíbrio entre classes permite que métricas globais (p. ex. acurácia) sejam informativas, mas ainda enfatizamos medidas que penalizam erros de classificação assimétricos, como _precision_ e _recall_.

---

## Pipeline de processamento

1. **TF‑IDF (1‑gramas → 3‑gramas)**
   - `max_features` selecionado automaticamente (10 000)
   - `min_df = 3`, `sublinear_tf = True`, _stop‑words_ em inglês
2. **LSA**
   - `TruncatedSVD(n_components = 300)` para compactar a matriz esparsa e realçar relações semânticas latentes
3. **K‑NN**
   - Métrica **cosseno** e pesos **distance**
   - Melhor parâmetro: **k = 5** vizinhos

A pesquisa de hiperparâmetros (`GridSearchCV`, CV estratificado 5×) avaliou 12 combinações; a que maximizou F1‑score foi `k = 5` com 10 000 features TF‑IDF.

---

## Desempenho do modelo

| Métrica                  | Valor      |
| ------------------------ | ---------- |
| Acurácia                 | **0,9305** |
| Precision                | **0,8963** |
| Recall (_Sensibilidade_) | **0,9731** |
| Especificidade           | **0,8882** |
| F1‑Score                 | **0,9332** |
| AUC (ROC)                | **0,9649** |

- **Matriz de confusão (limiar = 0,45)**
  - Verdadeiros Positivos = 9 164
  - Falsos Negativos = 253
  - Verdadeiros Negativos = 8 421
  - Falsos Positivos = 1 060
- O limiar foi ajustado de 0,50 para **0,45** a fim de elevar o _recall_ (≥ 97 %), aceitando leve redução de _precision_.
- A **AUC = 0,965** denota separabilidade quase perfeita: a chance de o classificador pontuar uma notícia verdadeira acima de uma falsa é \~96,5 %.

Gráficos gerados: `figs/confusion_matrix.png`, `figs/roc_curve.png`, `figs/f1_vs_k.png`.

---

## Interpretação

O modelo identifica praticamente todas as notícias verdadeiras, errando apenas 2,7 % delas. O custo dessa sensibilidade alta é uma taxa de falso‑positivo de 11,8 %, refletida na especificidade de 0,8882. Em cenários onde deixar de reconhecer textos legítimos é mais crítico que sinalizar alguns falsos alarmes, esse “viés para segurança” é aceitável.

A AUC superior a 0,96 corrobora a robustez do ranqueamento probabilístico: mesmo se o limiar mudar, o classificador mantém excelente discriminabilidade. Já o gráfico F1 × k mostra queda progressiva quando k > 5, indicando que valores maiores suavizam demais a fronteira de decisão, enquanto k < 5 tenderia ao sobreajuste.

---

## Requisitos

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.\.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

Principais dependências: `numpy`, `pandas`, `scikit‑learn`, `seaborn`, `matplotlib`, `joblib`.

---

## Como reproduzir

```bash
python news_classifier.py        # treina, avalia e salva modelo em modelo_knn_tfidf_otimizado.joblib
```

Os gráficos e métricas serão exibidos na tela e salvos em `figs/`.

---
