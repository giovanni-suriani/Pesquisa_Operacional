
""" Módulo para resolver o problema de programação linear utilizando o método Simplex em tabela"""
import logging
import settings
import re
import math
from sympy import Matrix, pprint, pretty
from fractions import Fraction
from collections import defaultdict

# Configuração do logger
import os
import logging

VERBOSE = settings.VERBOSE

explain = settings.PRECISO_EXPLICAR

logger = logging.getLogger("top_module.child")

if not logger.hasHandlers() and __name__ == '__main__':
    logging.config.dictConfig(settings.LOGGING)
    logger = logging.getLogger("top_module")  # __main__
    logging.getLogger("top_module.child").setLevel(logging.WARN)  # Logger filho de top_module
    print(f"sem handler, executando como top_module o arquivo {os.path.basename(__file__)}")

logger.debug(f"nome arquivo bicho = {os.path.basename(__file__)}")

import str_padrao_problema as spp
# import primal_dual as pd
import simplex_algorithm as sa



def assemble_line(row:list, first_column_size:int, other_columns_size:int, decimal=False):
    """ 
    cria uma linha da tabela var | x1 | x2 | ... | LD 
    Args:
        var (str): nome da variável
        first_column_size (int): tamanho da primeira coluna
        other_columns_size (int): tamanho das outras colunas
        amount_columns (int): quantidade de colunas
    Returns:
        str: linha formatada
    """
    # Formatando pela direita
    first_col = ""
    other_col = ""
    blank = " " * (first_column_size - (len(row[0]) + 1))
    first_col = f"{blank}{row[0]} |"
    
    for col in row[1:]:
        if decimal:
            col = spp.convert_if_str_float_is_int(col)
            blank = " " * (other_columns_size - (len(str(col)) + 1))
        else:
            blank = " " * (other_columns_size - (len(str(col)) + 1))
        other_col += f"{blank}{col} |"
    
    other_col = other_col[:-1]  # Remove o último "|"
    instance_other_col = other_col.split("|")[0]
    assert len(first_col) - 1 == first_column_size, f"len first_col: {len(first_col) - 1}, esperado = {first_column_size}"
    assert len(instance_other_col) == other_columns_size, f"len other_col: {len(instance_other_col)}, esperado = {other_columns_size}"
    return f"{first_col}{other_col}"

def concatenate_list_matrix_horizontal(matrices:list):
    """
    Concatena uma lista de matrizes horizontalmente.
    Args:
        matrices (list): lista de matrizes
    Returns:
        list: matriz concatenada
    """
    result = []
    for i in range(len(matrices[0])):
        row = []
        for matrix in matrices:
            row.extend(matrix[i])
        result.append(row)
    return result

def column_list_to_row_list(column:list):
    """
    Transforma uma lista de colunas em uma lista de linhas.
    Args:
        column (list): lista de colunas
    Returns:
        list: lista de matriz em forma de linha (transposto)
    """
    row = []
    for i in range(len(column[0])):
        row.append([column[j][i] for j in range(len(column))])
    return row[0]

def fill_with_zeros(x_to_fill_with_zero:list, x:list, values:list):
    """
    Preenche a lista x com zeros para que tenha o mesmo tamanho que x_to_fill.
    Args:
        x_to_fill_with_zero (list): variaveis a serem preenchidas com zeros dentro de x
        x (list): lista de referência
    """
    filled_with_zeros = []
    """ for i in range(len(x_to_fill_with_zero[0])):
        if i >= len(x):
            filled_with_zeros.append(0)
        else:
            filled_with_zeros.append(values[i])
    return filled_with_zeros """
    flat_x_to_fill = column_list_to_row_list(x_to_fill_with_zero)
    flat_x = column_list_to_row_list(x)
    values = values[0]
    i = 0
    for x_to_fill in flat_x:
        if x_to_fill in flat_x_to_fill:
            filled_with_zeros.append(0)
        else:
            filled_with_zeros.append(values[i])
            i += 1
    
    return filled_with_zeros
    #for 
    
    """ for i in range(len(x_to_fill)):
        if i >= len(x):
            x_to_fill.append(0)
    return x_to_fill """

def z_line_tableau(A_basic:list, A_non_basic:list, c_basic:list, c_non_basic:list, 
                    x_basic:list, x_non_basic:list,x:list):
    """
    Cria a tabela simplex a partir dos dados fornecidos.
    Args:
        A_basic (list): matriz de coeficientes das variáveis básicas
        A_non_basic (list): matriz de coeficientes das variáveis não básicas
    """
    #     A_basic = sa.get_A_basic(A, I0, x)
    #     A_non_basic = sa.get_A_non_basic(A, basis, x)
    #     c_basic = sa.get_c_basic(c, basis, x)
    #     c_non_basic = sa.get_c_non_basic(c, basis, x)
    mult_simplex = sa.calculate_simplex_multiplier_vector(A_basic, c_basic, "sergio")
    z_line = sa.calculate_reduced_costs(mult_simplex, A_non_basic, c_non_basic, "sergio")
    z_line = fill_with_zeros(x_basic, x, z_line)
    return z_line
    
def LD_piece_tableau(A_basic:list, A_non_basic:list, c_basic:list, c_non_basic:list, x:list, b:list):
    """
    Cria a tabela simplex a partir dos dados fornecidos.
    Args:
        A_basic (list): matriz de coeficientes das variáveis básicas
        A_non_basic (list): matriz de coeficientes das variáveis não básicas
    """
    mult_simplex = sa.calculate_simplex_multiplier_vector(A_basic, c_basic, "sergio")
    simplex_multiplier_vector = sa.calculate_simplex_multiplier_vector(A_basic, c_basic, "sergio")
    mat_b = Matrix(b)
    mat_simplex_multiplier_vector = Matrix(simplex_multiplier_vector)
    first_row = mat_simplex_multiplier_vector * mat_b
    mat_A_basic = Matrix(A_basic)
    mat_rows_left = mat_A_basic.inv() * mat_b
    mat_first_row = Matrix(first_row)
    LD_piece = sa.concatenate_matrices_vertical([mat_first_row, mat_rows_left]) # talvez nao funcione pq tm que ser Matrix
    return LD_piece

class SimplexTableau:
    def __init__(self, type_func:str, header:list, body:list, decimal:bool=False, iter:int=1, title:str="",):
        self.title = title
        self.type_func = type_func
        self.header = header
        self.body = body
        self.iter = iter
        self.decimal = decimal
        self.num_rows = len(body)
        self.num_cols = len(header)
        
    def iter_simplex(self):
        """
        Executa uma iteração do método simplex.
        """
        k_index = self.body[0].index(max(self.body[0]))
        if explain:
            logger.debug(f"argmax {self.body[0]}, k = {k_index}")
            logger.info(f"Escolhendo a variável de entrada: {self.header[k_index + 1]}")
        
            
    def __getitem__(self, key):
        """
        Permite acessar os elementos da tabela como se fosse uma lista.
        Exemplo: tableau[0] retorna a primeira linha da tabela.
        """
        if isinstance(key, int):
            return self.body[key]
        elif isinstance(key, str):
            for row in self.body:
                if row[0] == key:
                    return row
            raise KeyError(f"Variável '{key}' não encontrada na tabela.")
        else:
            raise TypeError("A chave deve ser um inteiro ou uma string.")
        
    def __setitem__(self, key, value):
        """
        Permite modificar os elementos da tabela.
        Exemplo: tableau[0] = [1, 2, 3, 4] altera a primeira linha da tabela.
        """
        if isinstance(key, int):
            self.body[key] = value
        elif isinstance(key, str):
            for i, row in enumerate(self.body):
                if row[0] == key:
                    self.body[i] = value
                    return
            raise KeyError(f"Variável '{key}' não encontrada na tabela.")
        else:
            raise TypeError("A chave deve ser um inteiro ou uma string.")    
    
    def __str__(self):
        # Making the super header
        if self.type_func == "max":
            tag = "Maximização"
        elif self.type_func == "min":
            tag = "Minimização"
        else:
            raise ValueError("Tipo de função inválido. Deve ser 'max' ou 'min'.")
        super_header = f"\n── Tabela {self.title} – iteração {self.iter}"
        
        # Making the header part
        biggest_var_len = max(len(var) for var in self.header)
        header_str =  assemble_line(self.header, biggest_var_len + 1, 12)
        intermediate_line = "-" * math.ceil(len(header_str) * 1.25)
        
        # Making the body part
        body = ""
        for body_line in self.body:
            body_line = assemble_line(body_line, biggest_var_len + 1, 12)
            body += f"{body_line}\n"
       
        return f"{super_header}\n{header_str}\n{intermediate_line}\n{body}"


    def create_tableau(A:list, b:list, c:list, x:list, I0:list=None, type_func:str="min"):
        """
        Cria a tabela simplex a partir dos dados fornecidos.
        Args:
            A (list): matriz de coeficientes das restrições
            b (list): vetor de constantes das restrições
            c (list): vetor de coeficientes da função objetivo
            I0 (list): lista de índices das variáveis básicas
            x (list): lista de índices das variáveis não básicas
        """
        x_list = [x_var[0] for x_var in x]
        header = [" "] + x_list + ["LD"]
        
        if not I0:
            I0 = sa.find_initial_basis(A, b, c)

        basis = I0
        A_basic = sa.get_A_basic(A, I0, x)
        A_non_basic = sa.get_A_non_basic(A, basis, x)
        c_basic = sa.get_c_basic(c, basis, x)
        c_non_basic = sa.get_c_non_basic(c, basis, x)
        x_non_basic = sa.get_x_non_basic(x, basis)
        z_line = z_line_tableau(A_basic, A_non_basic, c_basic, c_non_basic, basis, x_non_basic,x)
        LD_piece = LD_piece_tableau(A_basic, A_non_basic, c_basic, c_non_basic, x, b)
        LD_head = LD_piece.pop(0)
        z_line.append(LD_head[0])
        z_line.insert(0, "z")
        middle = concatenate_list_matrix_horizontal([basis, A, LD_piece])
        middle.insert(0, z_line)
        body = middle
        #body = concatenate_list_matrix_horizontal([basis, z_line, LD_piece])
        #basis.insert(0, ["z"])
        #body = sa.concatenate_matrices_horizontal([Matrix(basis),Matrix(j_piece), Matrix(LD_piece)])
        #for list in body:
            #list[0] = str(list[0])
        
        #print(body[0])
        return SimplexTableau(type_func, header, body, title="Tabela Simplex Generica")
        

x = [['x1'], ['x2'], ['x3'], ['x4'], ['x5']]
x_non_basic = [['x1'], ['x2']]
x_basic = [['x3'], ['x4'], ['x5']]
values = [[1, 1]]
A = [[2,  1, 1, 0, 0],
    [1, 2, 0, 1, 0]]
little_A = [[2], [1]]


A = [[2,  1, 1, 0, 0],
    [1, 2, 0, 1, 0],
    [0, 1, 0, 0, 1]]
b = [[8], [7], [3]]
c = [-1, -1, 0, 0, 0]
I0 = [["x3"], ["x4"], ["x5"]]
x = [["x1"], ["x2"], ["x3"], ["x4"], ["x5"]] 


bob = SimplexTableau.create_tableau(A, b, c, x, I0, type_func="min")
logger.debug(f"bob0 = {bob[0]}")

# logging.debug(f"vai se danar {bob[0]}")

#print(concatenate_list_matrix_horizontal([A, little_A]))

""" A = [[2,  1, 1, 0, 0],
    [1, 2, 0, 1, 0],
    [0, 1, 0, 0, 1]]
b = [[8], [7], [3]]
c = [-1, -1, 0, 0, 0]
I0 = [["x3"], ["x4"], ["x5"]]
x = [["x1"], ["x2"], ["x3"], ["x4"], ["x5"]] 

bob =SimplexTableau.create_tableau(A, b, c, x, I0, type_func="min") """



""" 
header = [" ", "x1", "x2", "x3", "LD"]
body = [
    ["z", 24, 0, 0, 0],
    ["x1", 0, 0, 0, 0],
]
bob = SimplexTableau("max", header, body, title="Problema Teste")
print(bob.body) """


""" 
 Cruzeiro lista
A = [[1,  1, 1, 0, 0],
    [1, -1, 0, 1, 0],
    [-1, 1, 0, 0, 1]]
b = [[6], [4], [4]]
c = [-1, -2, 0, 0, 0]
I0 = [["s1"], ["s2"], ["s3"]]
x = [["x1"], ["x2"], ["s1"], ["s2"], ["s3"]] 
"""




def test_battery_solve_simplex_table():
    """Executa a bateria de testes do módulo solve_simplex_table."""
    # TODO: escreva asserts ou chame suas funções de teste aqui
    pass


def check_health_status():
    """Roda testes utilitários e do módulo atual; levanta exceção se algo falhar."""
    logger.setLevel(logging.INFO)
    logger.info("Iniciando testes utilitários …")
    try:
        # Chame aqui sua suíte genérica; ajuste se necessário
        logger.info("Utilitários passaram!")

        logger.info(f"Iniciando testes de solve_simplex_table …")
        test_battery_solve_simplex_table()
        logger.info("Todos os testes passaram com sucesso!")

    except Exception as e:
        logger.error("Falha em algum teste:")
        logger.exception(e)
        raise
