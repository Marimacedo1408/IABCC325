import cv2 
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

# Constantes utilizadas no modelo
EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():
    # Verifica se os argumentos de linha de comando foram passados corretamente
    if len(sys.argv) not in [2, 3]:
        sys.exit("Uso: python traffic.py data_directory [model.h5]")

    # Carrega as imagens e os respectivos rótulos a partir do diretório fornecido
    images, labels = load_data(sys.argv[1])

    # Converte os rótulos para formato categórico
    labels = tf.keras.utils.to_categorical(labels)
    # Divide os dados em conjuntos de treino e teste
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Cria e compila o modelo de rede neural
    model = get_model()

    # Treina o modelo usando os dados de treinamento
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Avalia o desempenho do modelo com os dados de teste
    model.evaluate(x_test, y_test, verbose=2)

    # Salva o modelo em um arquivo, se o nome do arquivo for fornecido
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Modelo salvo em {filename}.")


def load_data(data_dir):
    """
    Carrega dados de imagem a partir do diretório `data_dir`.

    Assume que `data_dir` contém uma pasta para cada categoria, numerada
    de 0 até NUM_CATEGORIES - 1. Em cada pasta há arquivos de imagem.

    Retorna uma tupla `(images, labels)`, onde:
    - images: lista de imagens no formato numpy ndarray com dimensões IMG_WIDTH x IMG_HEIGHT x 3.
    - labels: lista de rótulos inteiros correspondentes a cada imagem.
    """

    print(f'Carregando imagens do diretório "{data_dir}"')

    images = []
    labels = []

    # Itera por cada pasta (categoria) dentro do diretório
    for foldername in os.listdir(data_dir):
        # Verifica se o nome da pasta é um número inteiro
        try:
            int(foldername)
        except ValueError:
            print("Aviso! Nome de pasta não é inteiro no diretório de dados! Pulando...")
            continue

        folder_path = os.path.join(data_dir, foldername)
        # Itera por cada arquivo de imagem na pasta atual
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            # Lê a imagem utilizando OpenCV
            img = cv2.imread(img_path)
            # Verifica se a imagem foi carregada corretamente
            if img is None:
                print(f"Aviso: Imagem {img_path} não pôde ser carregada. Pulando...")
                continue
            # Redimensiona a imagem para as dimensões definidas
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            # Normaliza os valores dos pixels para o intervalo [0, 1]
            img = img / 255.0

            # Adiciona a imagem e seu rótulo às listas correspondentes
            images.append(img)
            labels.append(int(foldername))

    # Verifica se o número de imagens corresponde ao número de rótulos carregados
    if len(images) != len(labels):
        sys.exit('Erro ao carregar dados: número de imagens não corresponde ao número de rótulos!')
    else:
        print(f'{len(images)} imagens com {len(labels)} rótulos carregadas com sucesso!')

    return (images, labels)


def get_model():
    """
    Retorna um modelo compilado de rede neural convolucional.
    O `input_shape` da primeira camada é (IMG_WIDTH, IMG_HEIGHT, 3) e a camada
    de saída possui NUM_CATEGORIES unidades, uma para cada categoria.
    """

    # Cria o modelo de rede neural utilizando Keras
    model = tf.keras.models.Sequential([
        # Primeira camada convolucional com 64 filtros, tamanho 3x3, ativação ReLU e padding 'same'
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        # Camada de pooling para reduzir a dimensionalidade
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Segunda camada convolucional com 64 filtros, tamanho 3x3, ativação ReLU e padding 'same'
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        # Outra camada de pooling
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Achata as saídas das camadas convolucionais para passar para as camadas densas
        tf.keras.layers.Flatten(),

        # Camada densa com 512 unidades e ativação ReLU
        tf.keras.layers.Dense(512, activation="relu"),
        # Camada de dropout com taxa de 50% para reduzir overfitting
        tf.keras.layers.Dropout(0.5),

        # Camada de saída com NUM_CATEGORIES unidades e ativação softmax para classificação
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Compila o modelo com o otimizador Adam e função de perda categorical_crossentropy
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


if __name__ == "__main__":
    main()
