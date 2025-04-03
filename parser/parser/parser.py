import nltk
import sys
import re

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | NP VP Conj NP VP | NP VP Conj VP
NP -> N | Det N | Det AP N | P NP | NP P NP
VP -> V | Adv VP | V Adv | VP NP | V NP Adv
AP -> Adj | AP Adj
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # Se um arquivo foi especificado, lê a sentença do arquivo
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Caso contrário, obtém a sentença via input
    else:
        s = input("Sentença: ")

    # Converte a entrada para uma lista de palavras
    s = preprocess(s)

    # Tenta analisar a sentença
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Não foi possível analisar a sentença.")
        return

    # Imprime cada árvore com os "chunks" de frases nominais
    for tree in trees:
        tree.pretty_print()

        print("Chunks de Frases Nominais")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Converte `sentence` para uma lista de suas palavras.
    Realiza o pré-processamento da sentença convertendo todos os caracteres para minúsculas
    e removendo qualquer palavra que não contenha pelo menos um caractere alfabético.
    """
    # Regex para corresponder palavras que contenham pelo menos um a-z, A-Z:
    test = re.compile('[a-zA-Z]')

    # Tokeniza usando o nltk:
    tokens = nltk.word_tokenize(sentence)

    # Retorna uma lista de strings minúsculas que correspondem à Regex:
    return [entry.lower() for entry in tokens if test.match(entry)]


def np_chunk(tree):
    """
    Retorna uma lista de todos os "chunks" de frases nominais na árvore da sentença.
    Um "chunk" de frase nominal é definido como qualquer subárvore da sentença
    cujo rótulo seja "NP" e que não contenha outras frases nominais como subárvores.
    """

    chunks = []

    # Converte a Árvore para uma Árvore Parentada
    ptree = nltk.tree.ParentedTree.convert(tree)

    # Itera por todas as subárvores na árvore:
    for subtree in ptree.subtrees():
        # Se a subárvore for rotulada como um substantivo, então o pai é um "chunk" de frase nominal
        if subtree.label() == "N":
            chunks.append(subtree.parent())

    return chunks


if __name__ == "__main__":
    main()
