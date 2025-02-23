import sys

from math import inf
from crossword import *
from copy import deepcopy

BACKTRACK_COUNTER = 0
WORDS_TESTED = 0

class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        if not interleaving:
            print('Solving Crossword with single arc consistency enforcement...')
            return self.backtrack(dict())
        else:
            print('Solving Crossword with interleaved backtracking and arc consistency enforcement...')
            return self.backtrack_ac3(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for v in self.domains:
            self.domains[v] = {palavra for palavra in self.domains[v] if len(palavra) == v.length}
    
    
    def overlap_satisfied(self, x, y, val_x, val_y):
        """
        Retorna True se os valores val_x e val_y satisfazem a restrição de sobreposição entre x e y.
        Caso não haja sobreposição, retorna True.
        """
        if not self.crossword.overlaps[x, y]:
            return True
        x_index, y_index = self.crossword.overlaps[x, y]
        return val_x[x_index] == val_y[y_index]

                
    def revise(self, x, y):
        """
        Torna a variável x arc-consistente em relação à variável y.
        Remove de self.domains[x] os valores que não possuem nenhum correspondente compatível em self.domains[y].
        Retorna True se o domínio de x foi alterado, caso contrário, retorna False.
        """
        revisao = False
        remover = set()
        
        for valor_x in self.domains[x]:
            consistente = False
            for valor_y in self.domains[y]:
                # Garante que a palavra não seja a mesma e que a sobreposição seja satisfeita
                if valor_x != valor_y and self.overlap_satisfied(x, y, valor_x, valor_y):
                    consistente = True
                    break
            if not consistente:
                remover.add(valor_x)
                revisao = True

        self.domains[x] = self.domains[x] - remover
        return revisao


    def ac3(self, arcs=None):
        """
        Atualiza self.domains para que cada variável seja arc-consistente.
        Se arcs for None, inicia com todos os arcos possíveis; caso contrário, usa a lista fornecida.
        Retorna True se os domínios forem mantidos consistentes; caso algum domínio fique vazio, retorna False.
        """
        if arcs is None:
            arcs = []
            for x in self.domains:
                for y in self.domains:
                    if x != y:
                        arcs.append((x, y))
        while arcs:
            x, y = arcs.pop()
            if self.revise(x, y):
                if not self.domains[x]:
                    return False
                for z in self.crossword.neighbors(x) - {y}:
                    arcs.append((z, x))
        return True

    

    def assignment_complete(self, assignment):
        """
        Retorna True se a atribuição for completa (todas as variáveis possuem um valor), ou False caso contrário.
        """
        for var in self.domains:
            if var not in assignment:
                return False
        return True


    def consistent(self, assignment):
        """
        Retorna True se a atribuição for consistente com todas as restrições:
        - Todas as palavras são únicas.
        - Cada palavra tem o comprimento correto.
        - As interseções entre variáveis satisfazem a restrição.
        """
        palavras_usadas = []
        for x in assignment:
            valor_x = assignment[x]
            if valor_x in palavras_usadas:
                return False
            palavras_usadas.append(valor_x)
            if len(valor_x) != x.length:
                return False
            for y in self.crossword.neighbors(x):
                if y in assignment:
                    valor_y = assignment[y]
                    if not self.overlap_satisfied(x, y, valor_x, valor_y):
                        return False
        return True



    def order_domain_values(self, var, assignment):
        """
        Retorna uma lista de valores no domínio de var, ordenados pelo número de valores eliminados
        dos domínios dos vizinhos (do menos para o mais eliminador).
        """
        contagem = {valor: 0 for valor in self.domains[var]}
        for valor in self.domains[var]:
            for vizinho in self.crossword.neighbors(var):
                for outro_valor in self.domains[vizinho]:
                    if not self.overlap_satisfied(var, vizinho, valor, outro_valor):
                        contagem[valor] += 1
        return sorted(list(contagem.keys()), key=lambda x: contagem[x])



       

    def select_unassigned_variable(self, assignment):
        """
        Retorna uma variável não atribuída, escolhida pela heurística MRV e, em caso de empate,
        pela variável com maior número de vizinhos.
        """
        nao_atribuidas = set(self.domains.keys()) - set(assignment.keys())
        variaveis = list(nao_atribuidas)
        variaveis.sort(key=lambda var: (len(self.domains[var]), -len(self.crossword.neighbors(var))))
        return variaveis[0]



    def backtrack(self, assignment):
        """
        Utiliza busca por retrocesso para encontrar uma atribuição completa, se possível.
        Retorna a atribuição completa ou None se não houver solução.
        """
        global WORDS_TESTED, BACKTRACK_COUNTER
        BACKTRACK_COUNTER += 1

        if self.assignment_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)
        for val in self.order_domain_values(var, assignment):
            assignment[var] = val
            WORDS_TESTED += 1
            if self.consistent(assignment):
                result = self.backtrack(assignment)
                if result is not None:
                    return result
            del assignment[var]
        return None


    
    def backtrack_ac3(self, assignment):
        """
        Utiliza busca por retrocesso intercalada com inferência (AC3) para encontrar uma atribuição completa.
        Retorna a atribuição completa ou None se não houver solução.
        """
        global WORDS_TESTED, BACKTRACK_COUNTER
        BACKTRACK_COUNTER += 1

        if self.assignment_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)
        pre_assignment_domains = deepcopy(self.domains)
        for val in self.order_domain_values(var, assignment):
            assignment[var] = val
            WORDS_TESTED += 1
            if self.consistent(assignment):
                self.domains[var] = {val}
                self.ac3([(outro, var) for outro in self.crossword.neighbors(var)])
                result = self.backtrack_ac3(assignment)
                if result is not None:
                    return result
            del assignment[var]
            self.domains = pre_assignment_domains
        return None




def main():

    # Verifica se o número de argumentos está correto
    if len(sys.argv) not in [3, 4]:
        sys.exit("Uso: python generate.py estrutura palavras [saida]")

    # Faz o parse dos argumentos da linha de comando
    estrutura = sys.argv[1]
    palavras = sys.argv[2]
    saida = sys.argv[3] if len(sys.argv) == 4 else None

    # Cria o cruzadinha a partir dos arquivos fornecidos
    cruzadinha = Crossword(estrutura, palavras)
    criador = CrosswordCreator(cruzadinha)
    atribuicao = criador.solve()

    # Exibe o resultado
    if atribuicao is None:
        print("Sem solução.")
    else:
        criador.print(atribuicao)
        if saida:
            criador.save(atribuicao, saida)



if __name__ == "__main__":
    main()
