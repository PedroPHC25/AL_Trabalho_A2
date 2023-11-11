import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_base_matrix(folder):
    famous_images = []
    for artist in (os.listdir(folder)):
        folder_path = os.path.join(folder, artist)
        for image in (os.listdir(folder_path)): 
            path = os.path.join(folder, artist, image)
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            famous_images.append(img)

    famous_images_np = np.array(famous_images)
    train_famous_np_matrix = famous_images_np.reshape(famous_images_np.shape[0], famous_images_np.shape[1]*famous_images_np.shape[2])
    mean_train_face = np.mean(train_famous_np_matrix, axis=0)
    centered_train_face = train_famous_np_matrix - mean_train_face
    V_Face_Famous = np.linalg.svd(centered_train_face, full_matrices=False)[2]
    return V_Face_Famous


matrix_ws = create_base_matrix("imagens_artistas")

def projection(image, matrix):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.array(img)
    img = img.reshape(img.shape[0]*img.shape[1], 1)
    coordinates = np.dot(matrix, img)
    new_img = np.dot(matrix.T, coordinates)
    return new_img


def projected_image(img):
    plt.figure()
    plt.imshow(img.reshape(231,195), cmap='gray')

teste = projection("imagens_artistas\TS\Captura de tela 2023-11-11 140831.png", matrix_ws)

# print(projection("imagens_artistas\TS\Captura de tela 2023-11-11 140831.png", matrix_ws))
# print(projection("imagens_artistas\TS\Captura de tela 2023-11-11 140831.png", matrix_ws).shape)
projected_image(teste)