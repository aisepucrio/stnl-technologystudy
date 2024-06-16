

# Introdução ao SciPy

by Maria Isabel Nicolau

Bem-vindo ao repositório **"Introdução ao SciPy"**! Este repositório foi criado para fornecer uma visão geral sobre a biblioteca SciPy, com um foco especial no módulo `scipy.stats`. Aqui você encontrará exemplos de código e explicações sobre como utilizar alguns dos testes estatísticos mais comuns oferecidos por essa poderosa biblioteca.

## Sobre a Biblioteca SciPy

SciPy é uma biblioteca de código aberto em Python que é usada para resolver problemas matemáticos, científicos e de engenharia. Ela é construída sobre o NumPy, que é outra biblioteca essencial para computação numérica em Python. SciPy fornece funcionalidades adicionais que abrangem desde integração numérica, álgebra linear, otimização, até estatísticas e processamento de sinal.

Algumas das principais áreas de aplicação do SciPy incluem:

- Álgebra linear
- Integração numérica
- Otimização
- Processamento de sinais
- Estatísticas
- Processamento de imagens

## Módulo scipy.stats

O módulo `scipy.stats` é uma parte integral do SciPy que oferece um conjunto abrangente de ferramentas para estatísticas e análise de dados. Ele inclui uma variedade de distribuições estatísticas, funções de estatísticas descritivas e testes estatísticos que são essenciais para análise de dados.

Neste repositório, focaremos em dois testes estatísticos populares fornecidos pelo `scipy.stats`: o teste t independente (`ttest_ind`) e o teste de qui-quadrado de contingência (`chi2_contingency`).

### Teste t Independente (ttest_ind)

O teste t independente é usado para comparar as médias de duas amostras independentes para determinar se há evidência estatística de que as médias das populações são diferentes. Este teste é útil quando você deseja comparar dois grupos distintos, por exemplo, os resultados de um tratamento experimental versus um grupo de controle.

#### Exemplo de Uso:

```python
from scipy.stats import ttest_ind

# Dados de exemplo
grupo1 = [5.1, 7.3, 6.8, 7.9, 5.6]
grupo2 = [6.3, 7.8, 8.5, 6.7, 7.2]

# Realizando o teste t
stat, p_value = ttest_ind(grupo1, grupo2)

print(f'Estatística t: {stat}')
print(f'Valor p: {p_value}')
```

### Teste de Qui-Quadrado de Contingência (chi2_contingency)

O teste de qui-quadrado de contingência é utilizado para determinar se existe uma associação significativa entre duas variáveis categóricas. Este teste é frequentemente aplicado em tabelas de contingência, onde os dados são categorizados em diferentes grupos.

#### Exemplo de Uso:

```python
from scipy.stats import chi2_contingency

# Tabela de contingência de exemplo
tabela = [[10, 20, 30], [6, 9, 17]]

# Realizando o teste de qui-quadrado
chi2, p_value, dof, expected = chi2_contingency(tabela)

print(f'Estatística qui-quadrado: {chi2}')
print(f'Valor p: {p_value}')
print(f'Graus de liberdade: {dof}')
print(f'Frequências esperadas:\n {expected}')
```

## Conclusão

Este repositório é um ponto de partida para aprender sobre a biblioteca SciPy e suas capacidades estatísticas. Os exemplos fornecidos mostram como realizar testes t independentes e testes de qui-quadrado de contingência usando `scipy.stats`. 

Esperamos que este repositório seja útil para sua jornada na análise de dados com Python e SciPy!
