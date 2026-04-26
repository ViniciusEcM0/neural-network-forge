# Rede Neural do Zero em python

Este projeto é uma jornada de estudos para entender redes neurais desde seus cálculos mais básicos, sem usar PyTorch, TensorFlow ou NumPy.

A ideia não é criar uma biblioteca altamente eficiente que rivalize com PyTorch nem nada do tipo, é apenas reconstruir, passo a passo, os principais conceitos por trás de uma rede neural

- neurônio linear
- pesos e bias
- previsão
- erro
- função de perda
- gradiente
- descida do gradiente
- múltiplas entradas
- produto escalar
- múltiplos neurônios em uma camada
- funções de atiivação
- ReLU e Leaky ReLU
- múltiplas camadas em uma rede

## Objetivo

Entender, na prática, como uma rede neural funciona em seu interior, sem utilizar bibliotecas pre-prontas.
Talvez esse projeto pareça um pouco denso, mas realmente estamos partindo de um ponto muito em baixo.

O projeto busca implementar manualmente lógicas semelhantes ao que temos em PyTorch, partindo de uma simples função:
y = x * w + b

## Estrutura atual do Projeto

project/
│
├── main.py
├── neuron.py
├── layer.py
├── network.py
├── activation.py
├── data.py
├── README
└── docs/
    └── derivacao_gradiente.md

### neuron.py
Contém a classe Neuron. Um neurônio recebe uma lista de entradas, calcula a soma ponderada, aplica uma função de ativação e ajusta seus pesos durante o treino.
No geral, a forma é:
z = x1*w1 + x2*w2 + ... + b
a = activation(z)

Onde:
- w(weight) são os pesos treináveis.
- b(bias) é o valor de viés daquele neurônio
- z é a saída bruta daquele neurônio, antes da ativação
- a é a saída após a ativação

### layer.py

Contem a classe Layer. Uma camada é composta por vários neurônios recebendo os mesmos inputs em paralelo.
Por exemplo:

layer = [neuron_1, neuron_2, neuron_3]
inputs = [x1, x2]

neuron 1 -> output 1
neuron 2 -> output 2
neuron 3 -> output 3

saida da camada -> [output_1, output_2, output_3]

### network.py

Contém a classe Network. Uma rede é a sequência de camadas.
A saída de uma camada viira a entrada da próxima:
inputs -> layer_1 -> layer_2 -> output

A classe Network foi desenvolvida pensando em flexibilidade, funcionando da seguinte forma. Exemplo:
Network([2, 4, 3, 1], internal_act="relu", last_act="linear")

Isso representa:

Input layer        Hidden layer 1        Hidden layer 2        Output layer
  2 inputs            4 neurons             3 neurons            1 neuron┼

              ┌──────── (h1) ───────┐
              │                     │──────── (h1) ───────┐                     
   x1 ────────┼──────── (h2) ───────┤                     │
              │                     │──────── (h2) ───────┼─── (y)                     
   x2 ────────┼──────── (h3) ───────┤                     │
              │                     │──────── (h3) ───────┘
              └──────── (h4) ───────┘

### activation.py
 Contém as funções de atvação e suas derivadas. Atualmente, o projeto trabalha com as seguintes funções:
 - linear
 - relu
 - leaky_relu

 Cada função de ativação também possui sua derivada, usada no cálculo do gradiente. Por exemplo:

 def relu(z):
    return max(0, z)

def relu_derivative(z):
    return 1 if z > 0 else 0

### data.py
Contém apenas datasets simples criados manualmente para testar o aprendizado em pequena escala. Exemplo:
dataset = [
    ([1, 1], [6, 3, 5]),
    ([2, 1], [8, 2, 10]),
    ([1, 2], [9, 7, 3]),
]

Cada item possui: (inputs, expected_outputs)
No geral, é utilizado somente para testes extremamente simples, definindo funções matemáticas, para fazer o input do valor de x para os neurônios treinarem até que retornem o valor de y correto, porém é realmente feito para ser MUITO simples

### docs/derivacao_gradiente.md
Contém a explicação matemática detalhada da derivação do gradiente. Esse arquivo está aí só pra manter o código limpo, para não ter um comentário de mais de 100 linhas no meio da classe.
Mas basicamente ele explica como chegamos em fórmulas como:
gradient_w = -2 * error * activation_derivative(z) * input
gradient_b = -2 * error * activation_derivative(z)

## Conceitos implementados até agora
### Neurônio linear simples
Primeiro foi implementado um neurônio com só uma entrada:
y = x*w + b

O neurônio aprendia ajustando w e b para reduzir o erro entre a previsão e o valor esperado de y

### Função de perda
A peerda usada inicialmente é o erro ao quadrado:
loss = error ** 2

Isso faz com que erros positivos e negativos sejam tratados da mesma forma, e erros maiores sejam exponencialmente mais penalizados.

### Gradiente e descida do gradiente
O ajuste dos pesos é feito usando a derivada da perda em relação aos parâmetros.
Para um neurônio linear simples:
gradient_w = -2 * error * x
gradient_b = -2 * error

E a atualização:
w = w - learning_rate * gradient_w
b = b - learning_rate * gradient_b

### Multiplas entradas
O neurônio foi expandido para receber mais de uma entrada, basicamente, tendo mais pesos por neurônio, ou mais "parâmetros", se preferir:
inputs = [x1, x2, x3]
weights = [w1, w2, w3]

Como o neurônio sozinho tem só um output, esses inputs precisam ser transformados em 1 output, para isso usamos produto escalar:
z = x1*w1 + x2*w2 + x3*w3 + b

que é a mesma coisa que:
z = dot(inputs, weights) + bias

### Camada com múltiplos neurônios
Depois foi criada a classe Layer, onde vários neurônios recebem os mesmos inputs e produzem, cada um, um output.
Com isso é possivel aprender várias funções ao mesmo tempo. Por exemplo:

input = [x1, x2]

output 1 = 2*x1 + 3*x2 + 1
output 2 = -1*x1 + 4*x2
output 3 = 5*x1 - 2*x2 + 2

### Funções de ativação
A saída bruta do neurônio é chamada de z, z é basicamente o y de uma função linear, precisamos aplicar uma função de ativação nele.
Uma função de ativação serve basicamente como um botão de ligar e desligar o neurônio, em uma rede grade, onde os neurônios trabalham em conjunto e cada um desenvolve seus pesos de acordo com um padrão percebido diferente, não vamos querer que todos os neurônios funcionem em todos os casos, se a entrada não corresponde ao que aquele neurônio se especializou, ele não deve ser ativado, e nem penalizado pela resposta final. Por isso aplicamos alguma função de ativação:

a = activation(z)

Isso permite que o neurônio deixe de ser uma transformação linear, até agora foram testadas as seguintes ativações:
- linear
- relu
- leaky_relu

### ReLU
A relu basicamente faz um corte na linha da função linear:

if z > 0:
    return z
else:
    return 0

A função linear, por regra, desenha uma linha no plano cartesiano, visualmente, o que relu faz é tornar toda a parte da linha reta que passa para o lado negativo de y(vertical) em uma linha reta no eixo x, com um y constante em 0, traçando a linha diagonal novamente somente onde y é positivo.

### Leaky Relu
A leaky_relu é basicamente igual a relu, mas ela não zera totalmente o valor de y quando é menor de 0, apenas torna ele muito pequeno:

if z > 0:
    return z
else:
    return 0.01 * z

Isso reduz o risco de algum neurônio "morrer", no caso, ficar em um ponto onde ele é sempre menor que 0 e nunca é atualizado, leaky_relu é muito útil principalmente para camadas internas de neurônios, onde eles morrerem é mais comum.

## Exemplo de Uso
from network import Network
from data import dataset

lr = 0.01
epochs = 500

network = Network(
    layers=[2, 4, 3],
    internal_activ="relu",
    last_activ="linear"
)

loss_history = network.train(dataset, lr, epochs)

print(network.predict([5, 1]))

## Limitações atuais
Estou fazendo tudo manualmente do zero, e como o objetivo é didático, tento fazer de forma mais simples, então ainda faltam muitas atualizações para ser uma biblioteca funcional de Deep Learning. Atualmente ele não tem:
- vetorização eficiente
- autograd automático
- batches reais
- backpropagation completa entre múltiplas camadas
- otimizadores avançados
- datasets reais

A intenção é que tudo isso seja implementado aos poucos, de forma simplificada para ser fácil de entender o funcionamento.