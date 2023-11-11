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
    # print(train_famous_np_matrix, train_famous_np_matrix.shape)
    mean_train_face = np.mean(train_famous_np_matrix, axis=0)
    # print(mean_train_face, mean_train_face.shape)
    centered_train_face = (train_famous_np_matrix - mean_train_face)/np.std(train_famous_np_matrix)
    # print(centered_train_face, centered_train_face.shape)
    V_Face_Famous = np.linalg.svd(centered_train_face, full_matrices=False)[2]
    return mean_train_face, V_Face_Famous


mean_1, matrix_ws = create_base_matrix("imagens_artistas")

# print(matrix_ws, matrix_ws.shape)

# print(mean_1.shape)

def projection(image, matrix, mean):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.array(img)
    img = img.flatten()
    coordinates = np.dot(matrix, img)
    new_img = np.dot(matrix.T, coordinates)
    # print(new_img)
    new_img = new_img + mean
    # print(mean)
    # print(new_img)
    return new_img


def projected_image(img):
    plt.figure()
    plt.imshow(img.reshape(231,195), cmap='gray')
    plt.show()

teste = projection("imagens_artistas\TS\Captura de tela 2023-11-11 140856.png", matrix_ws, mean_1)
projected_image(teste)
print(teste)



# print(projection("imagens_artistas\TS\Captura de tela 2023-11-11 140831.png", matrix_ws))
# print(projection("imagens_artistas\TS\Captura de tela 2023-11-11 140831.png", matrix_ws).shape)
# projected_image(teste)