import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFBertForMaskedLM

# Modelo de linguagem mascarada pré-treinado
MODELO = "bert-base-uncased"

# Número de previsões a gerar
K = 3

# Constantes para gerar diagramas de atenção
FONTE = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
TAMANHO_GRID = 40
PIXELS_POR_PALAVRA = 200

def main():
    # Solicita o texto do usuário
    texto = input("Texto: ")

    # Carrega o tokenizer do modelo pré-treinado
    # O tokenizer converte o texto em tokens que o modelo pode entender
    tokenizer = AutoTokenizer.from_pretrained(MODELO)
    
    # Tokeniza a entrada
    # Converte o texto em uma representação numérica (tokens)
    tokens_entrada = tokenizer(texto, return_tensors="tf")
    
    # Obtém o índice do token de máscara
    indice_token_mascara = get_mask_token_index(tokenizer.mask_token_id, tokens_entrada)
    if indice_token_mascara is None:
        sys.exit(f"A entrada deve incluir o token de máscara {tokenizer.mask_token}.")

    # Carrega o modelo BERT pré-treinado
    # O BERT é um modelo de linguagem que pode prever palavras mascaradas
    modelo = TFBertForMaskedLM.from_pretrained(MODELO)
    
    # Processa a entrada com o modelo, obtendo também as atenções
    resultado = modelo(**tokens_entrada, output_attentions=True)

    # Obtém os logits para o token mascarado
    logits_token_mascara = resultado.logits[0, indice_token_mascara]
    
    # Gera previsões
    # Seleciona os K tokens mais prováveis para substituir o token mascarado
    top_tokens = tf.math.top_k(logits_token_mascara, K).indices.numpy()
    
    # Imprime as previsões substituindo o token de máscara
    for token in top_tokens:
        print(texto.replace(tokenizer.mask_token, tokenizer.decode([token])))

    # Visualiza as atenções do modelo
    # Mostra como o modelo presta atenção a diferentes partes do texto
    visualize_attentions(tokens_entrada.tokens(), resultado.attentions)

def get_mask_token_index(id_token_mascara, tokens_entrada):
    """
    Retorna o índice do token com o `id_token_mascara` especificado, ou
    `None` se não estiver presente nos `tokens_entrada`.
    """
    for i, token in enumerate(tokens_entrada.input_ids[0]):
        if token == id_token_mascara:
            return i
    return None

def get_color_for_attention_score(peso_atencao):
    """
    Retorna uma tupla de três inteiros representando um tom de cinza para o
    `peso_atencao` dado. Cada valor deve estar no intervalo [0, 255].
    """
    peso_atencao = peso_atencao.numpy()
    return (round(peso_atencao * 255), round(peso_atencao * 255), round(peso_atencao * 255))

def visualize_attentions(tokens, atenções):
    """
    Produz uma representação gráfica dos scores de auto-atenção.
    Para cada camada de atenção, um diagrama é gerado para cada cabeça de atenção na camada.
    Cada diagrama inclui a lista de `tokens` na sentença. O nome do arquivo para cada diagrama
    inclui tanto o número da camada (começando em 1) quanto o número da cabeça (começando em 1).
    """
    # As atenções mostram a importância que o modelo dá a cada token em relação aos outros
    for i, camada in enumerate(atenções):
        for k in range(len(camada[0])):
            numero_camada = i + 1
            numero_cabeça = k + 1
            generate_diagram(
                numero_camada,
                numero_cabeça,
                tokens,
                atenções[i][0][k]
            )

def generate_diagram(numero_camada, numero_cabeça, tokens, pesos_atencao):
    """
    Gera um diagrama representando os scores de auto-atenção para uma única
    cabeça de atenção. O diagrama mostra uma linha e coluna para cada um dos
    `tokens`, e as células são coloridas com base nos `pesos_atencao`, com células
    mais claras correspondendo a scores de atenção mais altos.
    O diagrama é salvo com um nome de arquivo que inclui tanto o `numero_camada`
    quanto o `numero_cabeça`.
    """
    # Cria uma nova imagem com fundo preto
    tamanho_imagem = TAMANHO_GRID * len(tokens) + PIXELS_POR_PALAVRA
    imagem = Image.new("RGBA", (tamanho_imagem, tamanho_imagem), "black")
    desenho = ImageDraw.Draw(imagem)

    # Desenha cada token na imagem
    for i, token in enumerate(tokens):
        # Desenha as colunas de tokens
        imagem_token = Image.new("RGBA", (tamanho_imagem, tamanho_imagem), (0, 0, 0, 0))
        desenho_token = ImageDraw.Draw(imagem_token)
        desenho_token.text(
            (tamanho_imagem - PIXELS_POR_PALAVRA, PIXELS_POR_PALAVRA + i * TAMANHO_GRID),
            token,
            fill="white",
            font=FONTE
        )
        imagem_token = imagem_token.rotate(90)
        imagem.paste(imagem_token, mask=imagem_token)

        # Desenha as linhas de tokens
        _, _, largura, _ = desenho.textbbox((0, 0), token, font=FONTE)
        desenho.text(
            (PIXELS_POR_PALAVRA - largura, PIXELS_POR_PALAVRA + i * TAMANHO_GRID),
            token,
            fill="white",
            font=FONTE
        )

    # Desenha cada célula com base nos pesos de atenção
    for i in range(len(tokens)):
        y = PIXELS_POR_PALAVRA + i * TAMANHO_GRID
        for j in range(len(tokens)):
            x = PIXELS_POR_PALAVRA + j * TAMANHO_GRID
            cor = get_color_for_attention_score(pesos_atencao[i][j])
            desenho.rectangle((x, y, x + TAMANHO_GRID, y + TAMANHO_GRID), fill=cor)

    # Salva a imagem
    imagem.save(f"Atenção_Camada{numero_camada}_Cabeça{numero_cabeça}.png")

if __name__ == "__main__":
    main()