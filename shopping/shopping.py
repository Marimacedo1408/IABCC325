import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # Dicionário que mapeia os meses para valores numéricos
    meses = {
        'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3,
        'May': 4, 'June': 5, 'Jul': 6, 'Aug': 7,
        'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11
    }

    # Dicionário que mapeia os tipos de visitante para inteiros
    visitantes = {'Returning_Visitor': 1, 'New_Visitor': 0, 'Other': 0}

    # Dicionário que converte valores booleanos em inteiros
    booleanos = {'TRUE': 1, 'FALSE': 0}

    # Inicializa as listas para as evidências e os rótulos
    evidence = []
    labels = []

    # Abre o arquivo CSV e lê os dados como dicionário
    with open(filename, newline='') as arquivo_csv:
        leitor_csv = csv.DictReader(arquivo_csv, delimiter=',')
        print('Carregando dados do arquivo CSV...')
        total_linhas = 0

        # Itera sobre cada linha do CSV
        for linha_csv in leitor_csv:
            total_linhas += 1
            linha_evidencia = []

            # Adiciona os valores convertidos à lista de evidências
            linha_evidencia.append(int(linha_csv['Administrative']))
            linha_evidencia.append(float(linha_csv['Administrative_Duration']))
            linha_evidencia.append(int(linha_csv['Informational']))
            linha_evidencia.append(float(linha_csv['Informational_Duration']))
            linha_evidencia.append(int(linha_csv['ProductRelated']))
            linha_evidencia.append(float(linha_csv['ProductRelated_Duration']))
            linha_evidencia.append(float(linha_csv['BounceRates']))
            linha_evidencia.append(float(linha_csv['ExitRates']))
            linha_evidencia.append(float(linha_csv['PageValues']))
            linha_evidencia.append(float(linha_csv['SpecialDay']))
            linha_evidencia.append(meses[linha_csv['Month']])
            linha_evidencia.append(int(linha_csv['OperatingSystems']))
            linha_evidencia.append(int(linha_csv['Browser']))
            linha_evidencia.append(int(linha_csv['Region']))
            linha_evidencia.append(int(linha_csv['TrafficType']))
            linha_evidencia.append(visitantes[linha_csv['VisitorType']])
            linha_evidencia.append(booleanos[linha_csv['Weekend']])

            # Adiciona a linha de evidências à lista principal
            evidence.append(linha_evidencia)

            # Converte o valor de Revenue e adiciona à lista de rótulos
            labels.append(booleanos[linha_csv['Revenue']])

        # Verifica se a quantidade de evidências é igual à de rótulos
        if len(evidence) != len(labels):
            sys.exit('Erro ao carregar dados! O número de evidências não corresponde ao número de rótulos.')

        print('Dados carregados com sucesso! Total de linhas:', total_linhas)
        return evidence, labels


def train_model(evidence, labels, k=1):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # Exibe uma mensagem informando o valor de k que está sendo utilizado
    print('Ajustando o modelo usando o classificador k-Nearest Neighbours com k =', k)

    # Cria uma instância do classificador k-NN com o número de vizinhos definido por k
    modelo = KNeighborsClassifier(n_neighbors=k)
    
    # Treina o modelo utilizando os dados de evidência e os rótulos
    modelo.fit(evidence, labels)
    
    # Retorna o modelo treinado
    return modelo


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    # Conta o número total de rótulos positivos (compras) e negativos (não compras)
    total_positivos = labels.count(1)
    total_negativos = labels.count(0)

    # Inicializa os contadores para acertos em positivos e negativos
    acertos_positivos = 0
    acertos_negativos = 0

    # Percorre cada índice para comparar a previsão com o rótulo real
    for i in range(len(predictions)):
        if predictions[i] == labels[i]:
            # Se a previsão for positiva e estiver correta, incrementa os acertos positivos
            if predictions[i] == 1:
                acertos_positivos += 1
            # Caso contrário, se for negativa e correta, incrementa os acertos negativos
            else:
                acertos_negativos += 1

    # Calcula a sensibilidade (taxa de verdadeiros positivos)
    sensibilidade = acertos_positivos / total_positivos
    # Calcula a especificidade (taxa de verdadeiros negativos)
    especificidade = acertos_negativos / total_negativos

    return sensibilidade, especificidade


if __name__ == "__main__":
    main()
