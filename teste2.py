import os
import cv2
import numpy as np

# Criando o path para onde as iamgens estão
famousim = "imagens_artistas/"
# Criando uma lista para adicionar as imagens
famous_images = []

# Fazendo um for loop para iterar sobre todas as imagens
for image in (os.listdir(famousim)): 
    # Adionanco o path com cada imagem para termos o arquivo da imgens
    path = os.path.join(famousim, image)
    # Lendo a imagem
    img = cv2.imread(path)
    # Transformando a imagem em preto e branco 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Adicionando a imgem na lista
    famous_images.append(img)

# Transformando a lista num array
famous_images_np = np.array(famous_images)
# print(famous_images_np.shape)

