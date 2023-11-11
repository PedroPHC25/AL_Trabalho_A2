import numpy as np
import matplotlib.pyplot as plt

data = np.random.randint(100, size = (100, 2))

# print(data)

# plt.scatter(data[:, 0], data[:, 1])
# plt.show()

means = np.array([np.mean(data[:, 0]), np.mean(data[:, 1])])

# print(means)

data = data - means

stds = np.array([np.std(data[:, 0]), np.std(data[:, 1])])
# print(stds)

data = data / stds

# print(data)

# plt.scatter(data[:, 0], data[:, 1])
# plt.show()

#######################################################################

np.random.seed(5)

x = [x + np.random.randint(-5, 5) for x in range(-50, 50)]
y = [x + np.random.randint(-20, 20) for x in range (-50, 50)]

x_column = np.array(x)
y_column = np.array(y)

x_column = (x_column - np.mean(x_column))/np.std(x_column)
y_column = (y_column - np.mean(y_column))/np.std(y_column)

dados = np.column_stack((x_column, y_column))
# print(dados)

cov = np.dot(dados.transpose(), dados)
print("Matriz de covariância: ", cov)

# plt.scatter(dados[:, 0], dados[:, 1])
# plt.show()

autovalores, autovetores = np.linalg.eig(cov)

print("Autovalores:", autovalores)
print("Autovetores:", autovetores)

vetor0 = autovalores[0] * autovetores[1]
vetor1 = autovalores[1] * autovetores[0]

# print("Vetor 0:", vetor0)
# print("Vetor 1:", vetor1)

plt.figure(figsize = (5, 5))

plt.scatter(dados[:, 0], dados[:, 1])
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.xlabel("Variável x")
plt.ylabel("Variável y")
plt.show()

plt.figure(figsize = (5, 5))
plt.scatter(dados[:, 0], dados[:, 1], alpha = 0.5)
plt.arrow(0, 0, vetor0[0]/100, vetor0[1]/100, 
          head_width=0.1, 
          head_length=0.2, 
          fc='black', 
          ec='black', 
          label='PC1')
plt.arrow(0, 0, vetor1[0]/100, vetor1[1]/100, 
          head_width=0.1, 
          head_length=0.2, 
          fc='gray', 
          ec='gray', 
          label='PC2')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.xlabel("Variável x")
plt.ylabel("Variável y")
plt.show()

variancia_total = np.sum(autovalores)

variancia_por_variavel = (autovalores/variancia_total)*100

# print("Variância por componente:", variancia_por_variavel)

plt.bar(["PC1", "PC2"], variancia_por_variavel)
plt.xlabel("Componentes principais")
plt.ylabel("Participação na variância (%)")
plt.show()

novo_dado = np.dot(autovetores, dados.transpose())

# print(novo_dado)

plt.scatter(novo_dado[1, :], novo_dado[0, :])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.show()