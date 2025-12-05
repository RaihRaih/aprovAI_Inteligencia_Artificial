# *Projeto AprovAI — Predição Antecipada de Risco de Reprovação*

O *AprovAI* é um projeto que utiliza técnicas de Machine Learning para prever o risco de reprovação de estudantes antes da divulgação da nota final (G3).
O objetivo é permitir que escolas e professores identifiquem alunos em risco e possam agir de forma preventiva, oferecendo reforço, apoio e intervenções pedagógicas.

---

# *1. Introdução*

O projeto busca responder a uma necessidade comum em ambientes educacionais:

> *É possível prever se um aluno será aprovado ou reprovado antes do fim do período letivo?*

Com base em dados de desempenho parcial (G1, G2) e variáveis sociais/comportamentais, treinamos modelos para antecipar o resultado final — permitindo intervenções mais rápidas e eficazes.

---

# *2. Definição do Problema*

O desafio principal é:

*Prever se um aluno será aprovado (ou reprovado) antes que a nota final G3 seja conhecida.*

Essa previsão antecipada ajuda em:

* Monitoramento de risco
* Redução de reprovação e evasão
* Ações preventivas e reforço direcionado
* Suporte pedagógico individualizado

---

# *3. Descrição da Base de Dados*

O projeto utiliza duas bases do Kaggle (Maths e Portuguese):

*Dataset:* [https://www.kaggle.com/datasets/whenamancodes/student-performance/data](https://www.kaggle.com/datasets/whenamancodes/student-performance/data)

As bases contêm originalmente *33 variáveis* com informações pessoais, familiares, comportamentais e acadêmicas.

## *3.1. Visão Geral das Categorias*

### *a) Informações pessoais e familiares*

Incluem:

* sex, age, address, famsize, Pstatus
* Escolaridade dos pais: Medu, Fedu
* Ocupação dos pais: Mjob, Fjob
* Relações familiares: famrel, guardian, famsup

### *b) Hábitos e rotina escolar*

Informações sobre comportamento:

* studytime, traveltime, freetime
* goout (socialização)
* Dalc, Walc (consumo de álcool)
* absences (faltas)

### *c) Notas*

* G1 — nota do 1º período
* G2 — nota do 2º período
* G3 — nota final

---

## *3.2. Seleção das Colunas Usadas no Modelo*

O projeto utiliza exatamente as colunas preparadas por:

python
def preparar_df(df, nome):
    colunas = [
        "G1", "G2",
        "traveltime", "freetime", "studytime",
        "absences", "famsize", "Pstatus",
        "G3"
    ]


### *Colunas utilizadas no treinamento*

| Tipo                   | Colunas                               | Justificativa                                      |
| ---------------------- | ------------------------------------- | -------------------------------------------------- |
| *Desempenho parcial* | G1, G2                            | São os melhores preditores da nota final.          |
| *Rotina escolar*     | studytime, traveltime, freetime | Medem dedicação, tempo livre e deslocamento.       |
| *Frequência*         | absences                            | Faltas estão fortemente ligadas a reprovação.      |
| *Contexto familiar*  | famsize, Pstatus                  | Indicadores sociais que podem impactar desempenho. |
| *Desempenho final*   | G3                                  | Usada apenas para criar o target.                  |

---

## *3.3. Criação da Variável-Alvo (Target)*

A coluna *G3* é usada para gerar a coluna *aprovado*, usada pelos modelos:

* *1 = aprovado (G3 ≥ 10)*
* *0 = reprovado (G3 < 10)*

Após isso, *G3 é removida* do conjunto de treinamento, simulando previsão antecipada.

Isso significa:

> O modelo aprende com G1, G2 e variáveis sociais para tentar prever se o aluno será aprovado antes da nota final existir.

---

# *4. Pré-processamento*

O pipeline de preparação inclui:

## *4.1. Seleção das colunas mais relevantes*

Escolhemos um conjunto enxuto e eficiente:

* G1, G2
* studytime, traveltime, freetime
* absences, famsize, Pstatus

## *4.2. Conversão de variáveis categóricas*

Variáveis como famsize e Pstatus são transformadas via:

python
LabelEncoder()


Isso é necessário para que os modelos consigam operar sobre variáveis textuais.

## *4.3. Divisão em treino e teste*

A base é dividida em:

* *80% treino*
* *20% teste*

Usando:

python
train_test_split(..., random_state=42)


Garantindo reprodutibilidade dos resultados.

---

# *5. Modelos Utilizados*

Dois modelos de Machine learning clássicos e eficazes foram utilizados KNN e Arvóre de decisão.
Um modelo de Rede Neural MLPClassifier (Multi-layer Perceptron).

---

## *5.1. KNN — K-Nearest Neighbors*

### Como funciona:

Classifica um novo aluno com base nos alunos mais “parecidos” com ele.

### Etapas realizadas:

* Padronização de variáveis (StandardScaler)
* Treinamento com KNN
* Avaliação por:

  * Acurácia
  * Relatório de Classificação
  * Matriz de Confusão
* Gráficos de dispersão para compreender relações entre variáveis

### Motivos para uso:

* Simples de entender
* Boa performance em datasets menores
* Ajuda a visualizar padrões globais

---

## *5.2. Árvore de Decisão*

### Como funciona:

Cria regras do tipo:


Se G2 < 10 → risco alto
Se absences > 8 → risco moderado
...


### Etapas realizadas:

* Treinamento com DecisionTreeClassifier
* Visualização da árvore com plot_tree
* Avaliação com:

  * Acurácia
  * Relatório de classificação
  * Matriz de confusão

### Por que usar:

* Muito interpretável
* Rápida
* Mostra importância das variáveis
* Permite compreender decisões como um professor faria

---

## *5.3. Rede Neural de Classificação*

### Como funciona:

A Rede Neural Multilayer Perceptron (MLP) aprende padrões combinando várias camadas de neurônios artificiais.
Ela identifica relações não lineares entre notas, comportamento e variáveis sociais, ajustando seus pesos durante o treinamento para melhorar a precisão da previsão de aprovação.

### Etapas realizadas:
* Padronização das variáveis numéricas (StandardScaler)
* Codificação de variáveis categóricas (OneHotEncoder)
* Construção do modelo com duas camadas ocultas (64 e 32 neurônios)
* Treinamento usando Pipeline
* Avaliação com:

    * Acurácia
    * Relatório de classificação
    * Precisão, Recall e F1-Score
    * Matriz de Confusão

* Cálculo da probabilidade de aprovação para cada aluno

### Por que usar:
* Aprende padrões mais complexos que modelos lineares
* Funciona bem com dados mistos (numéricos e categóricos)
* Capta relações não lineares entre variáveis
* Gera também a probabilidade de aprovação, não só a classe
* Boa opção quando muitos fatores influenciam o desempenho
---

# *6. Resultados Obtidos*

---

## *6.1. Machine Learning*

De forma geral, os resultados foram consistentes entre as duas bases e ambos modelos:

### *Acurácia média:* entre *0.65 e 0.75*

### Principais achados:

* A classe *aprovado* é mais fácil de prever
* A classe *reprovado* sofre com desbalanceamento (poucos casos)
* G2 é a variável mais importante para ambos os modelos
* Faltas (absences) também têm impacto relevante
* Fatores sociais têm influência menor, porém existente

### Gráficos produzidos:

* Dispersão por variável (KNN)
* Matriz de confusão
* Árvore de decisão completa

---

## *6.2.Rede Neural*

Os resultados da rede neural foram estáveis e próximos aos modelos tradicionais, mostrando boa capacidade de generalização.

### *Acurácia média:* entre *0.70 e 0.78*

### Principais achados:

* A rede neural identificou bem a classe aprovado
* A classe reprovado continuou mais difícil devido ao desbalanceamento
* As variáveis G1 e G2 foram as mais influentes no processo de classificação
* Faltas (absences) também contribuíram de forma relevante
* Variáveis sociais apresentaram influência menor, mas ajudaram a melhorar o ajuste

### Gráficos produzidos:

* Matriz de confusão
* Relatório de classificação
* Probabilidade individual de aprovação para cada aluno

---

# *7. Conclusões*

* *G1 e G2* são excelentes preditores do desempenho final.
* *Árvore de decisão* é a melhor opção para interpretar e justificar decisões.
* *KNN* funciona, mas pode ser sensível à escala e aos dados categóricos.
* *A Rede Neural (MLP)*  obteve desempenho ligeiramente superior e capturou padrões mais complexos, mostrando-se útil para prever risco de reprovação de forma antecipada.
* Fatores sociais funcionam como complementos, não determinantes.
* O modelo é útil para identificar alunos em risco antes do fechamento das notas.

---

# *8. Próximos Passos*

Para evoluir o projeto:

### 🔹 Refinamento de modelos

* Ajuste de hiperparâmetros (GridSearch)
* Uso de class_weight='balanced'

### 🔹 Modelos adicionais

* Random Forest
* Regressão Logística
* SVM

### 🔹 Melhorias no pipeline

* Balanceamento da base (SMOTE)
* Análise de correlação aprofundada

### 🔹 Interface prática

* Criar ferramenta online (Flask, Streamlit ou Gradio)
* Permitir aos professores testar alunos individualmente
