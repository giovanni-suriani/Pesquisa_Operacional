
""" Módulo para resolver o problema de programação linear utilizando o método Simplex em tabela"""
import logging
import settings
import re
import math

# Configuração do logger
logging.config.dictConfig(settings.LOGGING)
logger = logging.getLogger("top_module")  # __main__

import str_padrao_problema as spp
import primal_dual as pd

VERBOSE = settings.VERBOSE

explain = settings.PRECISO_EXPLICAR

logger.debug("Iniciando o módulo solve_simplex_tabela")


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
        
    def k_index(self):
        """
        Retorna o índice k da variavel a entrar na base
        """
        for i, row in enumerate(self.body):
            if row[0] == "k":
                return i
        return None
    
    
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

header = [" ", "x1", "x2", "x3", "LD"]
body = [
    ["z", 0, 0, 0, 0],
    ["x1", 0, 0, 0, 0],
]
bob = SimplexTableau("max", header, body, title="Problema Teste")

print(bob)

def bateria_de_testes_solve_simplex_tabela():
    pass
    
def check_health_status():
    pass