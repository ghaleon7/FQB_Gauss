import numpy as np
from numpy import sqrt as sq
from ._GaussUtils import gcdxy

def prod(lst):
    """
    Esta função calcula o produto de todos os elementos de uma lista.
    Argumentos:
        lst - list
    Devolve:
        p - int
    Exemplo:
        prod([1,2,3]):
            >>> 6
    """
    if not isinstance(lst, list):
        raise TypeError(f"O utilizador introduziu um argumento {type(lst)}," +\
                        " precisamos que seja uma lista.")
    p = 1
    for i in lst:
        p *=i
    return p

def prime_lst(upper):
    """
    Esta função permite obter uma lista dos números primos de 2
    até um limite superior dado.
    Argumentos:
        - upper: limite superior; int
    Devolve:
        - lst: lista de primos até ao limite superior 
    Exemplos:
        - prime_lst(10):
            >> [2,3,5,7]
    """
    if not isinstance(upper, (int, np.int32, np.int64)):
        raise TypeError("Precisamos de um limite superior inteiro.")
    if upper > 100000:
        raise ValueError("O limite superior deve ser inferior a 100000")
    lst = []
    for nr in range(2, upper + 1):
       for i in range(2, int(sq(nr))+1):
           if (nr % i) == 0:
               break
       else:
           lst.append(nr)
    return lst


def divm(n,d):
    """
    Executa o teste de divisibilidade. n é o inteiro a testar,
    d é o possível divisor. Supomos que n está escrito em base 10.
    Argumentos:
        - n,d - int
    Devolve:
        - stry - string
    Exemplo:
        divm(15,7)
            >> "O número 15 não é divisivel por 7."
    """
    if gcdxy(10,d)[0] != 1:
        raise TypeError("O divisor tem de ser coprimo com 10.")
    else:    
        a = n//10
        b = n%10
        inv = gcdxy(10,d)[1]
        N = abs(a + inv*b)
        while N > 10:
            a = N//10
            b = N%10
            N = abs(a + inv*b)
        div = N%d
        if bool(div):
            stry = "O número {} não é divisivel por {}.".format(n,d)
            return stry
        else:
            stry = "{} = {}*{}".format(n,n//d,d)
            return stry
        
