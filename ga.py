from random import randint
import math
import matplotlib.pyplot as plt
import numpy as np

class AlgoritimoGenetico():

    def __init__(self, x_min, x_max, y_min, y_max, precisao, tam_populacao, taxa_mutacao, taxa_crossover, num_geracoes):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        self.precisao = precisao

        self.tam_populacao = tam_populacao
        self.taxa_mutacao = taxa_mutacao
        self.taxa_crossover = taxa_crossover
        self.num_geracoes = num_geracoes
        # calcula o número de bits do x_min e x_máx no formato binário com sinal
        qtd_bits_x_min = len(bin(x_min).replace('0b', '' if x_min < 0 else '+'))
        qtd_bits_x_max = len(bin(x_max).replace('0b', '' if x_max < 0 else '+'))

        qtd_bits_y_min = len(bin(y_min).replace('0b', '' if x_min < 0 else '+'))
        qtd_bits_y_max = len(bin(y_max).replace('0b', '' if x_max < 0 else '+'))


        # o maior número de bits representa o número de bits a ser utilizado para gerar individuos
        self.num_bits_x = qtd_bits_x_max if qtd_bits_x_max >= qtd_bits_x_min else qtd_bits_x_min
        self.num_bits_y = qtd_bits_y_max if qtd_bits_y_max >= qtd_bits_y_min else qtd_bits_y_min

        # gera os individuos da população
        self.gerarpopulacao()
    
    def listtostring(self, s):
        string = ""

        for e in s:
            string+=e
        
        return string


    def gerarpopulacao(self):
        self.populacao = [[] for i in range(self.tam_populacao)]

        for individuo in self.populacao:
            num_x = randint(self.x_min,self.x_max)
            num_y = randint(self.y_min,self.y_max)
            # converte o número sorteado para formato binário com sinal
            num_bin_x = bin(num_x).replace('0b', '' if num_x < 0 else '+').zfill(self.num_bits_x)
            str(num_bin_x)
            num_bin_y = bin(num_y).replace('0b', '' if num_y < 0 else '+').zfill(self.num_bits_y)
            str(num_bin_y)

            num_bin = num_bin_x+ "#" + num_bin_y
            
            for bit in num_bin:
                individuo.append(bit)

            

                

    def _funcao_objetivo(self, num_bin):
        """
            Calcula a função objetivo utilizada para avlaiar as soluções produzidas
        """
        
        # converte o número binário para o formato inteiro
        f = num_bin.index('#')
        numx=num_bin[:f]
        numy =num_bin[f+1:]
        
        numx = int(''.join(numx), 2)
        numy = int(''.join(numy), 2)
        # calcula e retorna o resultado da função objetivo
        obj = (0.5 - (math.sin(math.sqrt(numx**2+numy**2))**2-0.5/(1.0+0.0001*(numx**2+numy**2))**2))
        return obj
    
    def widowing(self, num_bin):
        """
            Calcula a função objetivo utilizada para avlaiar as soluções produzidas
        """
        
        # converte o número binário para o formato inteiro
        f = num_bin.index('#')
        numx=num_bin[:f]
        numy =num_bin[f+1:]
        const = 0.08
        numx = int(''.join(numx), 2)
        numy = int(''.join(numy), 2)
        # calcula e retorna o resultado da função objetivo
        obj = (0.5 - (math.sin(math.sqrt(numx**2+numy**2))**2-0.5/(1.0+0.0001*(numx**2+numy**2))**2))
        return obj-const
    


    
    
    
    def avaliar(self):
        """
            Avalia as souluções produzidas, associando uma nota/avalição a cada elemento da população
        """
        self.avaliacao = []
        for individuo in self.populacao:
            self.avaliacao.append(self._funcao_objetivo(individuo))

    def selecionar(self):
        """
            Realiza a seleção do individuo mais apto por torneio, considerando N = 2
        """
        # agrupa os individuos com suas avaliações para gerar os participantes do torneio
        participantes_torneio = list(zip(self.populacao, self.avaliacao))
        # escolhe dois individuos aleatoriamente
        individuo_1 = participantes_torneio[randint(0, self.tam_populacao - 1)]
        individuo_2 = participantes_torneio[randint(0, self.tam_populacao - 1)]
        # retorna individuo com a maior avaliação, ou seja, o vencedor do torneio
        return individuo_1[0] if individuo_1[1] >= individuo_2[1] else individuo_2[0]
    
    def _ajustar(self, individuo):
        """
            Caso o individuo esteja fora dos limites de x, ele é ajustado de acordo com o limite mais próximo
        """
        f = individuo.index('#')
        numx=individuo[:f]
        numy =individuo[f+1:]

        if int(''.join(numx), 2) < self.x_min:
            # se o individuo é menor que o limite mínimo, ele é substituido pelo próprio limite mínimo
            ajuste = bin(self.x_min).replace('0b', '' if self.x_min < 0 else '+').zfill(self.num_bits_x)
            for indice, bit in enumerate(ajuste):
                numx[indice] = bit
        elif int(''.join(numx), 2) > self.x_max:
            # se o individuo é maior que o limite máximo, ele é substituido pelo próprio limite máximo
            ajuste = bin(self.x_max).replace('0b', '' if self.x_max < 0 else '+').zfill(self.num_bits_x)
            for indice, bit in enumerate(ajuste):
                numx[indice] = bit

        if int(''.join(numy), 2) < self.y_min:
            # se o individuo é menor que o limite mínimo, ele é substituido pelo próprio limite mínimo
            ajuste = bin(self.y_min).replace('0b', '' if self.x_min < 0 else '+').zfill(self.num_bits_y)
            for indice, bit in enumerate(ajuste):
                numx[indice] = bit
        elif int(''.join(numy), 2) > self.y_max:
            # se o individuo é maior que o limite máximo, ele é substituido pelo próprio limite máximo
            ajuste = bin(self.y_max).replace('0b', '' if self.x_max < 0 else '+').zfill(self.num_bits_y)
            for indice, bit in enumerate(ajuste):
                numy[indice] = bit

    def crossover(self, pai, mae):
        """
            Aplica o crossover de acordo com uma dada probabilidade (taxa de crossover)
        """
        f = pai.index('#')
        paix=pai[:f]
        paiy =pai[f+1:]

        f = mae.index('#')
        maex=mae[:f]
        maey =mae[f+1:]

        if randint(0,1) <= self.taxa_crossover:
            # caso o crossover seja aplicado os pais trocam suas caldas e com isso geram dois filhos
            ponto_de_corte = randint(1, len(paix) - 1)#todos nem o mesmo lenght, escolhi paix por conveniencia

            filho_1x = paix[:ponto_de_corte] + maex[ponto_de_corte:]
            filho_1x = self.listtostring(filho_1x)

            filho_2x = maex[:ponto_de_corte] + paix[ponto_de_corte:]
            filho_2x = self.listtostring(filho_2x)


            filho_1y = paiy[:ponto_de_corte] + maey[ponto_de_corte:]
            filho_1y = self.listtostring(filho_1y)

            filho_2y = maey[:ponto_de_corte] + paiy[ponto_de_corte:]
            filho_2y = self.listtostring(filho_2y)
            # se algum dos filhos estiver fora dos limites de x, ele é ajustado de acordo com o limite
            # mais próximo
            
            filho_1_A = filho_1x+ "#" + filho_1y
            filho_1 = []
            
            filho_2_A = filho_2x+ "#" + filho_2y
            filho_2 = []

            for bit in filho_1_A:
                filho_1.append(bit)

            for bit in filho_2_A:
                filho_2.append(bit)

            self._ajustar(filho_1)
            self._ajustar(filho_2)    
        else:
            # caso contrário os filhos são cópias exatas dos pais
            filho_1 = pai[:]
            filho_2 = mae[:]

        # retorna os filhos obtidos pelo crossover
        return (filho_1, filho_2)
    
    def mutar(self, individuo):
        """
            Realiza a mutação dos bits de um indiviuo conforme uma dada probabilidade
            (taxa de mutação)
        """
        # cria a tabela com as regras de mutação
        tabela_mutacao = str.maketrans('+-01', '-+10')
        # caso a taxa de mutação seja atingida, ela é realizada em um bit aleatório
        for bit in range(len(individuo)):
            if randint(1,10000)/10000 <= self.taxa_mutacao and (individuo[bit]==1 or individuo[bit]==0):
                individuo[bit] = randint(0,1)
            
        
            '''bit = randint(0, self.num_bits - 1)
            individuo[bit] = individuo[bit].translate(tabela_mutacao)
        
        '''
        # se o individuo estiver fora dos limites de x, ele é ajustado de acordo com o
        # limite mais próximo
        self._ajustar(individuo)

    def econtrar_filho_mais_apto(self):
        """
            Busca o individuo com a melhor avaliação dentro da população
        """
        # agrupa os individuos com suas avaliações para gerar os candidatos
        candidatos = zip(self.populacao, self.avaliacao)
        # retorna o candidato com a melhor avaliação, ou seja, o mais apto da população
        return max(candidatos, key=lambda elemento: elemento[1])





def main():
    y = []
    # executa o algoritmo por "num_gerações"
    
    for ex in range(20):
        print("Experimento número: ", ex+1)
        algoritmo_genetico = AlgoritimoGenetico(-100, 100, -100, 100, 4, 100, 0.008, 0.65, 40)
        somaaptos = 0
        algoritmo_genetico.avaliar()
        for i in range(algoritmo_genetico.num_geracoes):
            # imprime o resultado a cada geração, começando da população original
            filhoapto = algoritmo_genetico.econtrar_filho_mais_apto()
            somaaptos = somaaptos + filhoapto[1]
            print( 'Resultado {}: {}'.format(i, filhoapto ))
            # cria uma nova população e a preenche enquanto não estiver completa
            nova_populacao = []
            while len(nova_populacao) < algoritmo_genetico.tam_populacao:
                # seleciona os pais
                pai = algoritmo_genetico.selecionar()
                mae = algoritmo_genetico.selecionar()
                # realiza o crossover dos pais para gerar os filhos
                filho_1, filho_2 = algoritmo_genetico.crossover(pai, mae)
                # realiza a mutação dos filhos e os adiciona à nova população
                algoritmo_genetico.mutar(filho_1)
                algoritmo_genetico.mutar(filho_2)
                nova_populacao.append(filho_1)
                nova_populacao.append(filho_2)
            # substitui a população antiga pela nova e realiza sua avaliação
            algoritmo_genetico.populacao = nova_populacao
            algoritmo_genetico.avaliar()

        

        # procura o filho mais apto dentro da população e exibe o resultado do algoritmo genético
        mediaaptos = somaaptos/(i+1)
        filhoapto = algoritmo_genetico.econtrar_filho_mais_apto()
        print( 'Resultado {}: {}'.format(i+1, filhoapto) )
        print('Média experimento: ', mediaaptos)
        y.append(mediaaptos)
        print("################################################")

    fig, ax = plt.subplots()

    ax.stairs(y, linewidth=2.5)

    ax.set(xlim=(0, 20), xticks=np.arange(0, 20),
       ylim=(0.9, 1), yticks=np.arange(0.9, 1.5))

    plt.savefig('Médias.png', format='png')
    plt.show()    
    

    # encerra a execução da função main
    return 0

if __name__ == '__main__':
    main()
