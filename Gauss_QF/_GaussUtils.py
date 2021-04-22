import numpy as np

############# FUNÇÕES AUXILIARES ###############

def gcdxy(a, b):
    """
    Esta função calcula o máximo divisor comum entre a e b 
    de forma a resolver a equação ax + by = mdc(a,b).
    Visto em: https://www.geeksforgeeks.org/python-program-
    for-basic-and-extended-euclidean-algorithms-2/
    Argumentos:
        - a,b - int
    Devolve:
        - gcd, x, y - int
    Exemplo:
        gcdxy(15,8)
            >> (1, -1, 2)
    """
    if a == 0 :   
        return b,0,1         
    gcd,x_i,y_i = gcdxy(b%a, a)  
    x,y = y_i - (b//a) * x_i, x_i
    return gcd,x,y 

def int_converter(*args):
    """
    Esta função serve para converter floats que também são inteiros 
    em inteiros.
    Argumentos:
        a,b,c - números inteiros
    Devolve:
        (a,b,c) - vetor de inteiros
    Exemplo:        
        float_converter(1.0, 2.0, 3.0)
            >> (1,2,3)
    """
    args = list(args)
    if all([isinstance(x,(int, np.int32, np.int64)) for x in args]):
        return tuple(args)
    elif any([isinstance(x,(float, np.float32, np.float64)) for x in args]):
        i = 0
        while i < len(args):
            if isinstance(args[i], (int, np.int32, np.int64)):
                args[i] = args[i]               
            elif args[i].is_integer():
                args[i] = int(args[i])
            i+=1
        return tuple(args)       
        
def verify(*args):
    """
    Esta função serve para verificar se as instâncias estão bem inicializadas.
    Se fizermos como F = QF(pi,2,3), ou F = QF(1.0, 2.0, 3.0) lança TypeError.
    Argumentos: 
        a,b,c - ints
    Devolve: 
        bool
    Exemplos:
        verify(np.pi,2,3)
            >> TypeError: Foi introduzido um número não-inteiro
        verify(1.0, 2.0, 3.0)
            >> TypeError: Foi introduzido um número não-inteiro
        verify(1,2,3)
            >> True
    """
    #Isto acautela que todas as instâncias são inteiros
    args = list(args)
    if not all([isinstance(x,(int, np.int32, np.int64)) for x in args]):
        raise TypeError("Foi introduzido um número não-inteiro")
    else:
        return True
        
def b_abs_min(b,a):
    """
    Método auxiliar que calcula o resíduo absoluto mínimo 
    de um número -b módulo a. Usa-se no método reducing para formas de 
    determinante negativo.
    """
    res = -b % a
    res_1 = res - a
    x = min(abs(res), abs(res_1))
    if x == abs(res):
        return res
    else:
        return res_1

def b_sqrt(b, a, lim_inf, lim_sup):
    """
    Método auxiliar que calcula o representante de um número -b módulo a
    Usa-se no método reducing para formas de determinante positivo
    """
    res = (-b)%a
    while res < lim_inf:
        res = res + abs(a)
    while lim_sup < res:
        res = res - abs(a)
    return res
