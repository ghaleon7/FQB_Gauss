# -*- coding: utf-8 -*-
"""
Esta classe procura definir e tratar as formas quadráticas binárias como as
estudou Carl Friedrich Gauss. 
Estudamos os polinómios de 2 variáveis: ax^2 + 2bxy + cy^2.
Nesta classe temos feito:
    - Operações fundamentais entre formas quadráticas binárias
    - Representação como tuplo e como matriz
    - Determinante e discriminante
    - Mudança de coordenadas por via de transformações lineares
    - Algoritmos de redução de formas quadráticas
    - Tipo de forma quadrática (definida positiva, definida negativa, indefinida)
    - Composição de formas quadráticas com determinante negativo
"""
import numpy as np
from numpy import sqrt as sq
from ._GaussUtils import gcdxy, int_converter, verify, b_abs_min, b_sqrt 

############# DEFINIÇÃO DA CLASSE ###############

class Gauss_QF:
    def __init__(self,a,b,c):
        """
        Esta classe trata de formas quadráticas binárias.
        Argumentos: 
            a,b,c - ints
        Exemplo:
            f = Gauss_QF(3,1,332444)
        """
        args = [a,b,c]
        if not all([isinstance(x,(int, float, np.float16, np.float32, 
                                  np.float64,np.int16, np.int32, np.int64)
                                ) for x in args]):
            raise TypeError("Foi um símbolo não-numérico") 
        x = int_converter(a,b,c)
        if verify(*x):
            self.a,self.b,self.c = x
        else:
            raise TypeError("Necessitamos de coeficientes inteiros!")
            
    @classmethod
    def from_matrix(cls, matrix):
        """
        Este método recebe uma matriz simétrica e identifica-a com uma 
        forma quadrática binária. 
        Argumentos: 
            matrix - A matriz da forma quadrática; list, np.array
        Devolve: 
            f_matrix - A forma quadrática associada; Gauss_QF
        Exemplo:
            f = np.array([[1, 1],
                          [1, 3]])
            g = Gauss_QF.from_matrix(f)
            print(g)
                >> A forma quadrática é (1,1,3) e 
                    é interpretada como: 1x^2+2xy+3y^2 
        """
        if not isinstance(matrix, np.ndarray):
            raise TypeError("O nosso objecto deve ser uma matriz")
        elif matrix.shape != (2,2):
            raise TypeError("A matriz deve ser do tipo 2x2")
        elif not np.allclose(matrix,matrix.T):
            raise ValueError("Queremos uma matriz simétrica")
        f_matrix = cls(int(matrix[0][0]), 
                        int(matrix[0][1]), 
                        int(matrix[1][1]))
        return f_matrix
        
    @classmethod
    def from_tuple(cls, tupl):
        """
        Este método recebe um tuplo ou lista com 3 elementos e identifica-o
        com uma forma quadrática binária.
        Argumentos:
            tupl - O terno com os coeficientes da forma; list, tuple
        Devolve:
            cl_tuple - Uma forma quadrática Gauss_QF(a,b,c)
        Exemplo:
            t = (1,2,3)
            g = Gauss_QF.from_tuple(t)
            print(g)
            >> A forma quadrática é (1,2,3) e é interpretada 
                como: 1x^2+4xy+3y^2
        """
        if not isinstance(tupl, (tuple,list)):
            raise TypeError("O nosso objecto deve ser um tuplo ou lista")
        if len(tupl) !=3:
            raise TypeError("O nosso objeto deve ter 3 elementos")
        cl_tuple = cls(tupl[0], tupl[1], tupl[2])
        return cl_tuple

    @property
    def determinant(self):
        """
        Este método calcula o determinante de uma forma quadrática.
        Devolve: 
            - determinant - o determinante da forma quadrática; int
        Exemplo:
            f = Gauss_QF(1,2,3)
            f.determinant
                >> 1
        """
        determinant = (self.b)**2 - (self.a)*(self.c) 
        return determinant
    
    @property
    def discriminant(self):
        """ 
        Este método calcula o discriminante de uma forma quadrática.
        Devolve:
            - discriminant - o discriminante da forma quadrática; int
        Exemplo:
            f = Gauss_QF(1,2,3)
            f.discriminant
                >> 4
        """
        discriminant = 4*(self.b)**2 - 4*(self.a)*(self.c) 
        return discriminant
    
    @property
    def rep_tuple(self):
        """
        Devolve a representação de uma forma quadrática binária como um tuplo.
        Devolve:
            f - A representação da forma quadrática como um terno; tuple
        Exemplo:
            f = Gauss_QF(1,2,3)
            f.rep_tuple
                >> (1,2,3)
        """
        f = (self.a, self.b, self.c)
        return f
        
    @property
    def rep_matrix(self):
        """
        Este método calcula a representação matricial de uma forma 
        quadrática binária. 
        Devolve: 
            F - A representação matricial da forma quadrática; np.array
        Exemplo:
            f = Gauss_QF(1,2,3)
            f.rep_matrix
                >> array([[1, 2],
                          [2, 3]])
        """
        F = np.array([[self.a, self.b], [self.b, self.c]])
        return F
        
    @property
    def is_reduced(self):
        """
        Testa se uma forma quadrática é reduzida.
        Devolve:
            bool - Caso seja reduzida, deve devolver True. 
                    False, caso contrário
        Exemplo:
            f = Gauss_QF(1,3,3)
            f.is_reduced
                >> False
        """
        if self.determinant < 0:
            cond_a = abs(2*self.b) <= abs(self.a) <= abs(self.c)
            cond_b = self.a <= sq(4*abs(self.determinant)/3)
            if cond_a and cond_b:
                return True
            else:
                return False
        else:
            if sq(self.determinant).is_integer():
                raise ValueError("O determinante não pode ser quadrado.")
            d = self.determinant
            cond_a = sq(d)-(self.b) <= abs(self.a) <=sq(d)+(self.b)
            cond_b = 0 < self.b <= sq(d)
            
            if cond_a and cond_b:
                return True
            else:
                return False
                
    def values(self, x, y):
        """
        Avalia o valor da forma quadrática f em (x,y), i.e., calcula f(x,y)
        Argumentos:
            x,y - Os objetos dos quais queremos calcular a imagem; int
        Devolve:
            res - A imagem da forma quadrática por x e y; int
        Exemplo:
            f = Gauss_QF(1,2,1)
            f.values(1,1)
                >> 4
        """
        coef_vect = [x,y]
        if not all([isinstance(i,(int, np.int32, np.int64)) 
                    for i in coef_vect]):
            raise TypeError("Procuramos inteiros.")
        res = (self.a)*x**2 + 2*(self.b)*x*y + (self.c)*y**2
        return res
        
    def reducing(self):
        """
        Método auxiliar que executa o algoritmo de redução de Gauss
        mostrando alguns cálculos intermédios. No processo calcula o 
        processo de redução de uma forma quadrática binária.
        Argumentos:
            - self: a forma quadrática binária
        Devolve:
           - Dicionário com 3 entradas: Forma, Progressão, Comprimento
        Exemplo:
            - f = Gauss_QF(5,6,10)
            - print(f.reducing())
                >> {'Forma':(6,2,9),
                    'Progressão':{'Forma_0':(9,7,11),
                                  'Forma_1':(11,4,6),
                                  'Forma_2':(6,2,9)},
                    'Comprimento': 3}
        """
        red = [self.a, self.b, self.c]
        res_dict = {'Forma': '0', 'Progressão':'0', 'Comprimento':'0'}
        progression = [tuple(red)]
        
        
        if self.is_reduced:
            res_dict['Forma'] = tuple(red)
            res_dict['Progressão'] = progression
            res_dict['Comprimento'] = 1
            return res_dict
        else:
            i = 1
            if self.determinant < 0:  
               while not Gauss_QF(*red).is_reduced:
                    red[0] = red[2]
                    red[1] = b_abs_min(red[1], red[0])
                    red[2] = int((red[1]**2 - self.determinant)/red[0])
                    progression.append(tuple(red))
                    i+=1
               res_dict['Forma'] = tuple(red)
               res_dict['Progressão'] = progression
               res_dict['Comprimento'] = i
               return res_dict
            if self.determinant >= 0:
                if sq(self.determinant).is_integer():
                    raise TypeError("O determinante não pode ser quadrado.")
                d = self.determinant
                while not Gauss_QF(*red).is_reduced:
                      red[0] = red[2]
                      red[1] = b_sqrt(red[1], red[0], sq(d)- red[0], sq(d))
                      red[2] = int((red[1]**2 - d)/red[0])
                      progression.append(tuple(red))
                      i+=1
                res_dict['Forma'] = tuple(red)
                res_dict['Progressão'] = progression
                res_dict['Comprimento'] = i
                return res_dict
                
    @property           
    def reduced(self):
        """
        Usa a função reducing para devolver uma forma quadrática 
        reduzida propriamente equivalente a f.
        Devolve:
            - f: A forma quadrática binária reduzida; Gauss_QF
        Exemplos:
            f = Gauss_QF(5,6,10)
            print(f.reduced())
                >> 'A forma quadrática é (6,2,9) e é interpretada 
                    como: 6x^2+4xy+9y^2'
        """
        prog = self.reducing()
        f = Gauss_QF(*prog['Forma'])
        return f
        
    def period(self):
        """
        Calcula o período de uma forma quadrática binária de 
        determinante positivo. Devolve uma lista com duas listas:
            - O processo de redução a uma forma quadrática reduzida
            - O período da forma quadrática reduzida equivalente à original
        Devolve:
            progression - list
        Exemplo:
            Gauss_QF(2,4,7).period()
                >> [[(2, 4, 7), (7, -4, 2), (2, 0, -1), (-1, 1, 1)], 
                [(1, 1, -1), (-1, 1, 1)]]
        """
        if self.determinant < 0:
            raise TypeError(
            "O conceito de período não existe para determinante negativo."
            )
        elif sq(self.determinant).is_integer():
            raise ValueError("O determinante não pode ser quadrado.")
        else:
            cap = self.reducing()['Progressão']
            red, ori = list(cap[-1]), list(cap[-1])
            d = self.determinant
            progression = []
            while not tuple(red) in progression:
                ori[0] = ori[2]
                ori[1] = b_sqrt(ori[1], ori[0], sq(d)- ori[0], sq(d))
                ori[2] = int((ori[1]**2 - d)/ori[0])
                progression.append(tuple(ori))
            progression.pop()
            progression.insert(0, tuple(red))
            return [cap, progression]

    def transform_linear(self, a_11, a_12, a_21, a_22):
        """
        Este método calcula a mudança de coordenadas 
        de uma forma quadrática binária.
        Argumentos: 
            a_11,a_12,a_21,a_22 - Entradas da matriz de 
                                    mudança de coordenadas; int
        Exemplos:
            g = Gauss_QF(1,0,1)
            h_0 = g.transform_linear(1,1,1,1)
            print(h_0)
                >> A forma inserida é (6,12,6) e é interpretada como: 
                    6x^2 + 24xy + 6y^2
        """
        if verify(*int_converter(a_11, a_12, a_21, a_22)):    
            matM = np.array([[a_11,a_12],[a_21,a_22]])
            transformed = ((matM.T).dot(self.rep_matrix)).dot(matM)
        trans_a = int(transformed[0][0])
        trans_b = int(transformed[0][1])
        trans_c = int(transformed[1][1])
        transformed_tuple = self.__class__(trans_a,trans_b,trans_c)
        return transformed_tuple    
        
    @property
    def typus(self):
        """
        Este método avalia se uma forma quadrática binária é definida, 
        semidefinida ou indefinida. 
        Exemplo:
            f = Gauss_QF(1,2,3)
            f.typus
                >> 'A forma inserida é definida positiva'            
        """
        if self.discriminant > 0:
            return 'A forma inserida é indefinida'
        elif self.discriminant == 0:
            return 'A forma inserida é semidefinida'
        else:
            if self.a > 0:
                return 'A forma inserida é definida positiva'
            else:
                return 'A forma inserida é definida negativa'

#################### OPERAÇÕES COM FORMAS #############
    
    def is_equal(self, other):
        """
        Criamos um objecto product iterável, o qual consiste em todos 
        os pares de índices possíveis. Em seguida, verificamos se a 
        representação matricial de ambas as formas quadráticas corresponde.
        Exemplo:
           f,g = Gauss_QF(1,2,3), Gauss_QF(1,2,3)
           f == g
               >> True
        """
        if not isinstance(other, self.__class__):
            raise TypeError("A forma inserida não é instância da classe")
        product_indices = ((x,y) for x in [0,1] for y in [0,1])
        truth_value = all([self.rep_matrix[i][j] == other.rep_matrix[i][j] 
                for (i,j) in product_indices])
        return truth_value
        
    def __eq__(self, other):
        """
        Definimos o que significa duas instâncias da classe serem iguais.
        """
        return self.is_equal(other)
        
    def add_forms(self, other):
        """
        Este método soma 2 instâncias da classe e devolve a 
        sua representação matricial.
        Exemplos:
            f, h = Gauss_QF(1,2,3), Gauss_QF(3,5,6)
            f + h
                >> array([[4, 7],
                          [7, 9]])
        """
        if not isinstance(other, self.__class__):
            raise TypeError("A forma inserida não é instância da classe")
        product_indices = ((x,y) for x in [0,1] for y in [0,1])
        add = np.array([self.rep_matrix[i][j] + other.rep_matrix[i][j]
                        for (i,j) in product_indices])
        add2 = np.array([[add[0], add[1]], [add[2], add[3]]])
        return add2
        
    def __add__(self,other):
        """
        Definimos o que significa somar duas instâncias da classe.
        """
        return self.add_forms(other)
        
    def subtract_forms(self, other):
        """
        Este método subtrai 2 instâncias da classe.
        Devolve a sua representação matricial.
        Exemplos:
            f, h = Gauss_QF(1,2,3), Gauss_QF(3,5,7)
            f - h
                >> array([[-2, -3],
                          [-3, -4]])
        """
        if not isinstance(other, self.__class__):
            raise TypeError("A forma inserida não é instância da classe")
        product_indices = ((x,y) for x in [0,1] for y in [0,1])
        sub = np.array([self.rep_matrix[i][j] - other.rep_matrix[i][j]
                        for (i,j) in product_indices])
        sub2 = np.array([[sub[0], sub[1]], [sub[2], sub[3]]])
        return sub2
    
    def __sub__(self,other):
        """
        Definimos o que significa subtrair duas instâncias da classe.
        """
        return self.subtract_forms(other)

    def __str__(self):
        """
        Este método permite obter uma representação de uma forma
        quadrática em string. Devolve uma mensagem com a representação
        do tuplo e o polinómio em duas variáveis a que está associado.
        """
        x = (self.a,self.b,self.c)
        a,b,c = self.a,2*self.b,self.c
        if self.b >= 0:
            b = "+{}".format(b)
        if self.c >= 0:
            c = "+{}".format(c)
        y = (a,b,c)
        s1 = "A forma quadrática é ({},{},{}) ". format(*x)
        s2 = "e é interpretada como: {}x^2{}xy{}y^2".format(*y) 
        return s1 + s2
        
    def compose(self, other):
       """
        Implementa o algoritmo de Shanks para a composição de formas 
        com determinante negativo. A forma resultante pode não ser
        reduzida, pelo que será necessário usar a função reduced
        para obter uma forma reduzida propriamente equivalente.
        Este algoritmo é uma implementação do algoritmo em
        A Course in Computational Algebraic Number Theory,
        de Henri Cohen. 3.ªed.Berlin, Heidelberg: Springer, 1996.
        isbn: 3-540-55640-0, p. 247
        Argumentos:
            - self, other: duas formas quadráticas a compor;
                            Gauss_QF
        Devolve:
            - composed: a forma quadrática resultante;
                            Gauss_QF
        Exemplo:
            f, g = Gauss_QF(3,1,332444), Gauss_QF(3,1,332444)
            h = f.compose(f)
            print(h)
                >> A forma quadrática é (9,7,110820) e é interpretada 
                    como: 9x^2+14xy+110820y^2
        """
        if not isinstance(other, self.__class__):
            raise TypeError("Só podemos compor formas quadráticas")
        if not self.discriminant < 0 or not other.discriminant < 0:
            raise TypeError("Só podemos compor formas "+ \
                            "com discriminante negativo")
                            
        if self.a > other.a:
            return other.compose(self)
        s = self.b + other.b
        n = 2*other.b - s
        if other.a % self.a == 0:
            y_1, d = 0, self.a
        else:
            y_1 = gcdxy(other.a, self.a)[1]
            d = gcdxy(other.a, self.a)[0]
        if s % d == 0:
            y_2, x_2, d_1 = -1, 0, d
        else:
            d_1 = gcdxy(s,d)[0]
            x_2 = gcdxy(s,d)[1]
            y_2 = -gcdxy(s,d)[2]
        v_1 = int(self.a/d_1)
        v_2 = int(other.a/d_1)
        r = (y_1*y_2*n - x_2*(other.c))%(v_1)
        b_3 = 2*(other.b + v_2*r)
        a_3 = v_1*v_2
        c_3 = int((b_3**2 - 4*self.determinant)/(4*a_3))
        composed = self.__class__(a_3, int(b_3/2), c_3)
        return composed
