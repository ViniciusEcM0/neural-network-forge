AQUI O BAGULHO FICA SINISTRO

y_previsto = x * w + b

erro = y_real - y_previsto
erro = y_real - (x * w + b)     <-- substituindo y_previsto
erro = y_real - x * w - b

perda = erro²     <-- perda é o quadrado do erro por conveniência, anula o sinal e torna o aumento exponencial
perda = (y_real - x * w - b)²     <-- substituindo erro

L(w) = (y_real - x * w - b)²     <-- perda se torna uma função de w, L é de loss
A = y_real - b
L(w) = (A - x * w)²
(A - x * w)² = (A - x * w) * (A - x * w) = A² - 2*A*x*w + x² * w²
L(w) = x²*w² - 2*A*x*w + A²
L(w) = x²*w² - 2*(y_real - b)*x*w + (y_real - b)²     <-- substituindo A

L(w) se tornou uma função quadrática de y = a*w² + b*w + c, formando uma parábola no plano
a = x²
b = -2*(y_real - b)*x
c = (y_real - b)²

AGORA FICA PIOR AINDA, FAZER DERIVADA
Faremos uma derivada pelo seguinte motivo, L(w) é a nossa perda, ou seja, o nosso medidor de erro, quanto menor ele for, mais "correto"
nosso neurônio está trabalhando, e ao mesmo tempo, ele é o y da função quadrática, claro que isso não é por acaso, basicamente,
quanto mais "em baixo" na parábola nosso ponto w estiver posicionado, em uma coordenada (w, y), menor o erro. A derivada vai nos
servir para descobrir para onde ir, basicamente, o que uma derivada faz, é separar um ponto minusculo, quase 0, e tomar esse ponto como
uma pequena linha na margem da parábola, e essa linha tem a inclinação imediata daquele ponto da parábola, pois ela é justamente a
linha que tangencia aquele ponto da parábola. A derivada nos retorna o referente ao "a" na função linear ax + b, ou seja, o que define
se a linha é crescente(sendo maior que 0), ou decrescente(sendo menor que 0), considerando que estamos em uma parábola, se a linha que
acompanha a sua inclinação está em forma decrescente, o ponto w está na esquerda do plano, precisando ser somado para aproximar y de 0,
se estiver em sua forma crescente, está do lado direito, precisando ser subtraido para o mesmo fim, sendo assim, se derivada < 0; w+, 
se derivada > 0; w-. Agora, calculando a derivada.

f'(w) = [f(w + h) - f(w)] / h

Primeiro termo:
f(w) = k*w²     <-- primeiro termo da função quadrática
f(w + h) = k*(w + h)²     <-- ponto próximo
(w + h)² = w² + 2*w*h + h²
f(w + h) = k*(w² + 2*w*h + h²)
f(w + h) = k*w² + 2*k*w*h + k*h²

f'(w) = [f(w + h) - f(w)] / h     <-- agora usamos a definição da derivada
f'(w) = [k*w² + 2*k*w*h + k*h² - k*w²] / h
f'(w) = [2*k*w*h + k*h²] / h     <-- cancela k*w² com -k*w²
f'(w) = [h*(2*k*w + k*h)] / h     <-- h em evidência para cancelar com a divisão externa por h
f'(w) = 2*k*w + k*h

Dada a natureza da derivada, é presumido que h -> 0 (h "tende a" 0), sendo assim podemos tratar ele como 0
f'(w) = 2*k*w     <-- k*0 = 0
derivada de k*w² = 2*k*w

Essa foi a prova por definição de derivada, a partir de agora vou usar a fórmula pq também sou filho de Deus:
derivada de k*wⁿ = n*k*wⁿ⁻¹

L(w) = x²*w² - 2*(y_real - b)*x*w + (y_real - b)²
derivada de x²*w² = 2*x²*w

derivada de -2*(y_real - b)*x*w
k = -2*(y_real - b)*x
n = 1
derivada de -2*(y_real - b)*x*w¹ = 1 * [-2*(y_real - b)*x] * w¹⁻¹ = -2*(y_real - b)*x

derivada de (y_real - b)² = 0     <-- não tem w, como nossa derivada é em relação ao w, o que não tem w é constante, a derivada é 0

Agora montamos na função
L(w) = x²*w² - 2*(y_real - b)*x*w + (y_real - b)²
dL/dw = 2*x²*w - 2*(y_real - b)*x + 0
dL/dw = 2*x*(x*w - (y_real - b))     <-- 2*x em evidência
dL/dw = 2*x*(x*w - y_real + b)
dL/dw = 2*x*(x*w + b - y_real)     <-- x*w + b dentro dos parênteses é igual a y_previsto
dL/dw = 2*x*(y_previsto - y_real)     <-- y_previsto - y_real é igual a -erro, já que y_real - y_previsto = erro
dL/dw = 2*x*(-erro)
dL/dw = -2*erro*x
Essa é a derivada  da perda em relação ao peso w, ela diz a inclinação da curva da perda no ponto w
Se dL/dw < 0, aumentar w reduz a perda.
Se dL/dw > 0, diminuir w reduz a perda.

ajuste_w = 2 * erro * x

AGORA FAREMOS A DERIVADA DA PERDA EM RELAÇÃO AO BIAS b

y_previsto = x*w + b

erro = y_real - y_previsto
erro = y_real - (x*w + b)
erro = y_real - x*w - b

perda = erro²

L(b) = (y_real - x*w - b)²
A = y_real - x*w     <-- y_real, x e w são constantes na derivada, já que não se relacionam com b
L(b) = (A - b)²
L(b) = A² -2*A*b + b²
 
Então termo por termo:
derivada da A² = 0     <-- não se relaciona com b, então é uma constante com valor 0 na derivada
derivada de -2*A*b = -2*A*b⁰ = -2*A
derivada de b² = 2b

dL/db = 0 - 2*A + 2*b
dL/db = 2*(b - A)
dL/db = 2*(b - (y_real - x*w))     <-- substituindo A por seus valores constantes
dL/db = 2*(b - y_real + x*w)
dL/db = 2*(x*w + b - y_real)
dL/db = 2*(y_previsto - y_real)     <-- w*x + b = y_previsto
dL/db = -2*erro 

Gradient descent:
novo_b = b - taxa_aprendizado * dL/db
novo_b = b - taxa_aprendizado * (-2*erro)
novo_b = b + taxa_aprendizado * (2*erro)
ajuste_b = 2*erro
novo_b = b + taxa_aprendizado * ajuste_b