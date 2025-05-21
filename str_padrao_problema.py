import re
import logging
from fractions import Fraction
from pprint import pprint
import sympy as sp
from sympy import Matrix, pprint, pretty
import sys, os

Fraction.__str__
import settings

# logging.config.dictConfig(settings.LOGGING)
# logging.config.dictConfig(settings.LOGGING)
logger = logging.getLogger("top_module.child")  

if not logger.hasHandlers() and __name__ == '__main__':
    logging.config.dictConfig(settings.LOGGING)
    logger = logging.getLogger("top_module")  # __main__
    print(f"sem handler, executando como top_module")

logger.debug("str_padrao_problema.py")

VERBOSE = settings.VERBOSE

VARIAVEL_ADICIONADA = 1

VARIAVEL_ALTERADA = 2

VARIAVEL_IRRESTRITA_MODIFICADA = 3

display_variable_priority = {
    "normal": 0,
    "slack": 100,
    "artificial":200,
}


regex_variable_expr = r'\s*\w([^\w\d]){0,2}\d+\s*'

# padrao = re.compile(
#     r'(?i)'  # ignore maiúsculas/minúsculas
#     r'(?P<tipo_funcao>max|min)\s+'  # grupo nomeado: tipo_funcao
#     r'(?P<funcao>('
#         r'(?:[-+]?\s*'                  # sinal opcional
#         r'(?:\d+(?:\.\d+)?|\d+/\d+)?'   # constante: inteiro, decimal ou fração
#         r'\s*\w([^\w\d]){0,2}\d+\s*)+'                # variável com número (ex: π1, x2)
#     r'))',
#     re.UNICODE
#     )

""" 
class LinhaRestricao:
        def __init__(self, restricao:str, detailed:bool = False):
            self.detailed = detailed
            constantes_lhs, variaveis_lhs, simbolo, valor_rhs = extrai_restricao(restricao)
            constantes_e_variaveis_lhs = dict(zip(variaveis_lhs, constantes_lhs))
            self.constantes_e_variaveis_lhs = constantes_e_variaveis_lhs
            self.simbolo = simbolo
            self.valor_rhs = valor_rhs
        
        def __str__(self):
            restricao = monta_restricao(self.constantes_e_variaveis_lhs, self.simbolo, 
                                        self.valor_rhs, detailed=self.detailed)
            return restricao[0]
        
        def __repr__(self):
            return self.__str__()
 """

def teste_gpt():
    logger.info(f"teste_gpt {logger.name}")

def convert_if_str_float_is_int(value:str) -> str:
    """
    Converte uma string que representa um número decimal em um inteiro, se possível.
    Args:
        value (str): valor a ser convertido
    Returns:
        str: valor convertido
    """
    float_value = float(value)
    if float_value.is_integer():
        return str(int(float_value))
    else:
        return value

from sympy import Matrix, pretty


from sympy import Matrix, pretty

def _side_by_side_with_labels(left: Matrix, right: Matrix, label_left: str, 
                              label_right: str, sep: str = " │ ") -> str:
    """Returns a nicely aligned multiline string with labeled left and right matrices."""
    left_lines = pretty(left).splitlines()
    right_lines = pretty(right).splitlines()

    # Compute sizes
    height = max(len(left_lines), len(right_lines))
    width_left = max(len(l) for l in left_lines)
    width_right = max(len(r) for r in right_lines)
    
    # Pad shorter matrix with blank lines
    left_lines += [" " * width_left] * (height - len(left_lines))
    right_lines += [" " * width_right] * (height - len(right_lines))

    # Prepare labels centered over their columns
    label_line = f"{label_left.center(width_left)}{sep}{label_right.center(width_right)}"

    # Combine lines
    body_lines = [f"{l.ljust(width_left)}{sep}{r}" for l, r in zip(left_lines, right_lines)]

    return "\n".join([label_line, *body_lines])

def display_matrix_f_obj(A: list, b: list, c: list, x: list) -> None:
    """
    Pretty-print the LP components:
        A │ b   (with labels)
        c │ x   (with labels)
    """
    A_m, b_m = Matrix(A), Matrix(b)
    c_m, x_m = Matrix(c), Matrix(x)

    # Force column vector display
    if c_m.shape[0] == 1:
        c_m = c_m.T
    if x_m.shape[0] == 1:
        x_m = x_m.T

    a_b_row = _side_by_side_with_labels(A_m, b_m, "[ A ]", "[ b ]")
    c_x_row = _side_by_side_with_labels(c_m, x_m, "[ c ]", "[ x ]")
    return "\n" + a_b_row + "\n\n" + c_x_row

def check_variable_unbounded_shatter_constraint(constraint:str) -> bool:
    re.compile(
        regex_variable_expr + r'\s*=\s*' + regex_variable_expr + r'\s*\-\s*' + regex_variable_expr
    )
    if re.search(
        r'\b' + regex_variable_expr + r'\s*=\s*' + regex_variable_expr + r'\s*\+\s*' + regex_variable_expr,
        constraint
    ):
        return True
    return False

def check_ge_zero(constantes_lhs:list, valor_rhs:list, simbolo:str = "≥") -> bool:
    """
    Checa se a restrição é maior ou igual a zero.
    Args:
        constantes_lhs (list[Fraction]): coeficientes do lado esquerdo
        valor_rhs (Fraction): valor do lado direito da restrição
        simbolo (str): símbolo de comparação
    Returns:
        bool: True se a restrição for do tipo x >= 0, False caso contrário
    """
    non_zer_vars = 0
    for constante in constantes_lhs:
        if constante != 0:
            non_zer_vars += 1
    if non_zer_vars == 1: 
        if simbolo == ">=" or simbolo == "≥":
            if valor_rhs == 0:
                return True
    return False

def check_le_zero(constants_lhs:list, value_rhs:list, symbol:str = "≤") -> bool:
    """
    Checa se a restrição é menor ou igual a zero.
    Args:
        constantes_lhs (list[Fraction]): coeficientes do lado esquerdo
        valor_rhs (Fraction): valor do lado direito da restrição
        simbolo (str): símbolo de comparação
    Returns:
        bool: True se a restrição for do tipo x <= 0, False caso contrário
    """
    non_zer_vars = 0
    for constant in constants_lhs:
        if constant != 0:
            non_zer_vars += 1
    if non_zer_vars == 1: 
        if symbol == "<=" or symbol == "≤":
            if value_rhs == 0:
                return True
    return False

def check_variable_constraint_ge_le__positive(constraint:str) -> bool:
    """ 
    Checa se a restricao eh do tipo x1 >= 0 ou  -x1 <= 0, ou seja, a variavel eh positiva
    Args:
        constraint (str): string da restrição, ex: "x4 ≥ 0"
    Returns:
        bool: True se a restrição for do tipo x >= 0 ou x <= 0, False caso contrário
    """
    constants_lhs, variables_lhs, symbol, value_rhs = extrai_restricao(constraint)
    if check_ge_zero(constants_lhs, value_rhs, symbol):
        if all(constant >= 0 for constant in constants_lhs):
            return True
    
    elif check_le_zero(constants_lhs, value_rhs, symbol):
        if all(constant <= 0 for constant in constants_lhs):
            return True
        
    return False

def check_variable_constraint(constraint:str, positive_lhs:bool = False) -> bool:
    """
    Checa se a restrição é de variavel x >= 0 ou x <= 0, irrestrita, shattered constraint (x1 = x'1 - x''1).
    Args:
        constraint (str): string da restrição, ex: "x4 ≥ 0"
        positive_lhs (bool): se True, retorna o sinal da restricao considerando o lado esquerdo como positivo
    Returns:
        bool: True se a restrição for do tipo x >= 0 ou x <= 0, False caso contrário
        str: qual tipo: (>=, <=, irrestrito, shatter)
    """
    constants_lhs, variables_lhs, symbol, value_rhs = extrai_restricao(constraint)
    if check_ge_zero(constants_lhs, value_rhs, symbol):
        if all(constant >= 0 for constant in constants_lhs):
            return True, ">="
        
    elif check_le_zero(constants_lhs, value_rhs, symbol):
        if all(constant <= 0 for constant in constants_lhs):
            return True, "<="
        
    elif check_variable_unbounded_shatter_constraint(constraint):
        return True, "shatter"
    
    non_zero_vars = 0
    for constant in constants_lhs:
        if constant != 0:
            non_zero_vars += 1
            
    if non_zero_vars == 1 and symbol == "irrestrito":
        return True, "irrestrito"
        
        
    
    return False, ""

def check_variable_constraint_std_form(constraint:str) -> bool:
    """
    Checa se a restrição está na forma padrão (xi >= 0).
    Args:
        constraint (str): string da restrição, ex: "x4 ≥ 0"
    """
    allowed_values = [0,1]
    constants_lhs, variables_lhs, symbol, value_rhs = extrai_restricao(constraint)
    if check_ge_zero(constants_lhs, value_rhs, symbol):
        if all(constant in allowed_values for constant in constants_lhs):
            return True
    return False

def remove_variable_constraints(constraints:list) -> None:
    """
    Remove as restrições de variáveis de uma lista de restrições.
    Args:
        restricoes (list): lista de restrições
    Returns:
        None, procedimento
    """
    for i in range(len(constraints)-1, -1, -1):
        check_var_constraint, _ = check_variable_constraint(constraints[i])
        if check_var_constraint:
            del constraints[i]

def variable_constraint_to_std_form(variable_constraint:str) -> bool:
    """ 
    Transforma de maneira forcada uma restricao de variavel em uma restricao do tipo xi >= 0\n
    Funciona para restricoes xi irrestrito, -xi <= 0 tambem
    Args:
        variable_constraint (str): restricao a ser verificada
    Returns:
        list: Constraints na forma padrao
    """
    variable_constraints = []
    constants_lhs, variables_lhs, symbol, value_rhs = extrai_restricao(variable_constraint)
    # Checando se a restricao é do tipo x1 = x'1 - x''1
    if check_variable_unbounded_shatter_constraint(variable_constraint):
        new_var_constrain1 = f"{variable_constraint[0]}'{variable_constraint[1:]} >= 0"
        new_var_constrain2 = f"{variable_constraint[0]}''{variable_constraint[1:]} >= 0"
        variable_constraints.append(new_var_constrain1, new_var_constrain2)
        #logger.debug(f"Variável irrestrita encontrada na restrição: {constraint}")
        return variable_constraints
    
    
    # Checando se a restricao
    non_zero_vars = 0 
    variable_foo = None
    for variable, constant in zip(variables_lhs, constants_lhs):
        if constant != 0:
            non_zero_vars += 1
            variable_foo = variable
    
    
    
    if non_zero_vars == 1:
    # Checando se a restricao é do tipo irrestrito
        if symbol == "irrestrito":
             new_var_constrain1 = f"{variable_constraint[0]}'{variable_constraint[1:]} >= 0"
             new_var_constrain2 = f"{variable_constraint[0]}''{variable_constraint[1:]} >= 0"
             variable_constraints.append(new_var_constrain1, new_var_constrain2)
             return variable_constraints
    # Se a restricao é do tipo >= ou <= ou =
        elif check_variable_constraint_ge_le__positive(variable_constraint):
                var_constraint = f"{variable_foo} >= 0"
                variable_constraints.append(var_constraint)
                return variable_constraints
        else:
            var_constraint = f"{variable_foo} >= 0"
    
    raise ValueError(f"Perdeu é máfia, brincadeira, deu ruim na linha {sys._getframe().f_lineno}")

def remove_ge_le_constraints(constraints:list) -> list:
    """
    Remove as restrições do tipo x >= 0 ou x <= 0 da lista de restricoes.
    Args:
        restricoes (list): lista de restrições
    Returns:
        None
    """
    to_remove = []
    for constraint in constraints:
        constants, variables, symbol, value_rhs = extrai_restricao(constraint)
        if check_ge_zero(constants, value_rhs, symbol) or check_le_zero(constants, value_rhs, symbol):
            to_remove.append(constraint)

    for constraint in to_remove:
        constraints.remove(constraint)

def flip_symbol_in_constraint(constraint:str, flip_lhs:bool=True, flip_rhs:bool=True) -> str:
    """
    Troca o sinal da restrição ex: 2x1 + 3x2 <= 1 se torna -2x1 - 3x2 >= -1 \n
    x1 >= 0 se torna x1 <= 0, 
    Args:
        constraint (str): string da restrição, ex: "2x1 + π2 + 3x4 ≥ 2/3"
        flip_lhs (bool): se True, troca o sinal do lado esquerdo
        flip_rhs (bool): se True, troca o sinal do lado direito
    Returns:
        str: string da restrição com o sinal trocado
    """
    constants_lhs, variables_lhs, symbol, value_rhs = extrai_restricao(constraint)
    if symbol == ">=" or symbol == "≥":
        symbol = "<="
        if flip_lhs:
            constants_lhs = [-constant for constant in constants_lhs]
        if flip_rhs:
            value_rhs = -value_rhs
    elif symbol == "<=" or symbol == "≤":
        symbol = ">="
        if flip_lhs:
            constants_lhs = [-constant for constant in constants_lhs]
        if flip_rhs:
            value_rhs = -value_rhs
    elif symbol == "=":
        logger.warning(f"Simbolo de igualdade passado, sem troca de sinal")
    return monta_restricao(dict(zip(variables_lhs, constants_lhs)), symbol, value_rhs)[0]

def change_unbounded_variable_format_in_f_obj(variable:str, f_obj:str, detailed:bool=False) -> str:
    """
    Troca o formato da variável irrestrita na função objetivo.
    Args:
        variable (str): variável a ter o formato trocado (ex: 'x1')
        f_obj (str): função objetivo como string (ex: 'min 2x1 + x2')
        detailed (bool): se True, retorna a função com termos em 0
    Returns:
        str: função objetivo com a variável irrestrita no formato x'1 + x''1
    """
    function_type, _, constants, variables = extrai_f_obj(f_obj)
    constants_and_variables = dict(zip(variables, constants))
    if variable in variables:
        new_var1 = variable[0] + "'" + variable[1:]
        new_var2 = variable[0] + "''" + variable[1:]
        constants_and_variables[new_var1] = constants_and_variables[variable]
        constants_and_variables[new_var2] = -constants_and_variables[variable]
        del constants_and_variables[variable]
    else:
        raise ValueError(f"Variável '{variable}' não encontrada na função objetivo.")
    
    return monta_f_obj(function_type, constants_and_variables, standard_form=False, detailed=detailed)

def change_unbounded_variable_format_in_constraint(variable:str, constraints:list, detailed = False) -> str:
    """ 
    Deleta a variavel passada das restricoes\n
    Adiciona duas variaveis novas na forma x'1 - x''1\n
    Adiciona as restricoes dessas variaveis novas
    Args:
        variable (str): variável a ter o sinal trocado
        restricoes (list): lista de restrições
        detailed (bool): se True, retorna a função com termos em 0
    Returns:
        None, procedimento
    """
    new_var1 = variable[0] + "'" + variable[1:]
    new_var2 = variable[0] + "''" + variable[1:]
    for i, constraint in enumerate(constraints):
        if check_variable_unbounded_shatter_constraint(constraint):
            #logger.debug(f"Variável irrestrita encontrada na restrição: {constraint}")
            continue
        constants, variables, symbol, value_rhs = extrai_restricao(constraint)
        constants_and_variables = dict(zip(variables, constants))
        if variable in variables:
            constants_and_variables[new_var1] = constants_and_variables[variable]
            constants_and_variables[new_var2] = -constants_and_variables[variable]
            del constants_and_variables[variable]
            constraints[i], _ = monta_restricao(constants_and_variables, symbol, value_rhs, detailed=detailed)
    
    constraints.append(f"{new_var1} >= 0")
    constraints.append(f"{new_var2} >= 0")
    
    return constraints

def change_variable_sign_in_constraints(variable:str, constraints:list, detailed = False, standard_form = False) -> None:
    """ 
    Troca o sinal da variável nas restrições
    Args:
        variable (str): variável a ter o sinal trocado
        restricoes (list): lista de restrições
        detailed (bool): se True, retorna a função com termos em 0
    Returns:
        None, procedimento
    """
    
    
    for i, constraint in enumerate(constraints):
        constants, variables, symbol, value_rhs = extrai_restricao(constraint)
        constants_and_variables = dict(zip(variables, constants))
        if variable in variables:
            check_var_constraint,_ = check_variable_constraint(constraint)
            if standard_form and check_var_constraint:
                new_constraints = variable_constraint_to_std_form(constraint)
                # Se for padrão, troca o sinal, apenas adiciona a variável zerada
                if len(new_constraints) == 1:
                    constraints[i] = new_constraints[0]
                elif len(new_constraints) == 2:
                    constraints[i] = new_constraints[0]
                    constraints[i+1] = new_constraints[1]
                continue
            constants_and_variables[variable] = -constants_and_variables[variable]
            constraints[i], _ = monta_restricao(constants_and_variables, symbol, value_rhs, detailed=detailed)
    
    return constraints

def change_variable_sign_in_f_obj(variable:str, f_obj:str, detailed:bool=False) -> str:
    """
    Troca o sinal do termo que contém a variável especificada na função objetivo.
    Args:
        variable (str): variável a ter o sinal trocado (ex: 'x1')
        f_obj (str): função objetivo como string (ex: 'min 2x1 + x2')
        detailed (bool): se True, retorna a função com termos em 0
    Returns:
        str: função objetivo com o sinal da variável trocado
    """
    function_type, _, constants, variables = extrai_f_obj(f_obj)
    
    constants_and_variables = dict(zip(variables, constants))
    if variable in variables:
        constants_and_variables[variable] = -constants_and_variables[variable]
    else:
        raise ValueError(f"Variável '{variable}' não encontrada na função objetivo.")
    return monta_f_obj(function_type, constants_and_variables, standard_form=False, detailed=detailed)

def adding_zero_vars_on_constants_variables_dict(constants_and_variables_owner:dict, constants_and_variables_expr:dict) -> dict:
    """
    Adiciona variáveis zeradas no dicionário de constantes e variáveis.
    Args:
        constantes_e_variaveis (dict): dicionário de constantes e variáveis
        expressao (str): expressão a ser analisada
    Returns:
        dict: dicionário atualizado com variáveis zeradas
    """
    new_dict = {}
    for variable, _ in constants_and_variables_owner.items():
        if variable not in constants_and_variables_expr:
            new_dict[variable] = 0
        else:
            new_dict[variable] = constants_and_variables_expr[variable]
    return new_dict

def str_list_fraction(fractions:list) -> str:
    """ Transforma uma lista de frações em uma string formatada.
    Args:
        fractions (list[Fraction]): lista de frações
    Returns:
        str: string formatada com as frações
    """
    str_frac = ""
    for fraction in fractions:
        str_frac += str(fraction) + " "
    return str_frac.strip()

def extrair_constantes_e_variaveis(expr:str, extract_pure_constants:bool = False) -> tuple:
    """
    Retorna listas de constantes (como Fraction) e variáveis de uma expressão
    Args:
        expr (str): expressão a ser analisada, ex: "3x1 + 2x2 - 5x3"
        extract_pure_constants (bool): se True, extrai constantes puras ex: "3", "-5", "2/3"
    Returns:
        Tuple:
            - constantes (list[Fraction]): lista de coeficientes, constantes puras no fim
            - variaveis (list[str]): lista de variáveis
    """
    padrao = re.compile(
        r'(?P<constante_associada>[+-]?\s*(?:\d+(?:\.\d+)?|\d+/\d+)?)(?P<variavel>\w([^\w\d]){0,2}\d+)'  # aceita int, decimal, fração
        r'|' # ou
        r'(?P<constante_pura>\b[+-]?\s*(?:\d+(?:\.\d+)?|\d+/\d+)?\b)',  
        re.UNICODE
    )
    termos = re.finditer(padrao, expr)
    constantes = []
    variaveis = []
    constantes_puras = []
    for termo in termos:
        variavel = termo.group("variavel")
        coef = termo.group("constante_associada")
        constante_pura = termo.group("constante_pura")
        if variavel:
            variaveis.append(variavel)
            if coef:
                coef = coef.replace(" ", "")
                if coef in ("", "+"):
                    coef = Fraction(1)
                elif coef == "-":
                    coef = Fraction(-1)
                else:
                    # converte decimal para Fraction também
                    coef = Fraction(coef)
            else:
                coef = 1
            constantes.append(coef)
            
        else:
            if constante_pura:
                constantes_puras.append(constante_pura)
           
    if extract_pure_constants:
        constantes = constantes + constantes_puras
    """ constantes = []
    variaveis = []
    for coef_raw, var in termos:
        coef = coef_raw.replace(" ", "")
        if coef in ("", "+"):
            coef = Fraction(1)
        elif coef == "-":
            coef = Fraction(-1)
        else:
            # converte decimal para Fraction também
            coef = Fraction(coef)
        constantes.append(coef)
        variaveis.append(var) """

    return constantes, variaveis

def extrai_f_obj(f_obj:str) -> tuple:
    """
    Extrai a função objetivo no formato padrão.

    Args:
        f_obj (Any): String da função objetivo, ex: "min 3x1 + 2x2"

    Returns:
        Tuple:
            - tipo_funcao (str): "min" ou "max"
            - funcao_objetivo (str): parte da função com variáveis
            - constantes (list[Fraction]): coeficientes extraídos
            - variaveis (list[str]): nomes das variáveis
    """
    padrao = re.compile(
    r'(?i)'  # ignore maiúsculas/minúsculas
    r'(?P<tipo_funcao>max|min)\s+'  # grupo nomeado: tipo_funcao
    r'(?P<funcao>('
        r'(?:[-+]?\s*'                  # sinal opcional
        r'(?:\d+(?:\.\d+)?|\d+/\d+)?'   # constante: inteiro, decimal ou fração
        + regex_variable_expr +
        r')+'# variável com número (ex: π1, x2)
    r'))',
    re.UNICODE
    )
    match = re.search(padrao, f_obj)
    assert match, f"Expressão inválida '{f_obj}'"
    tipo_funcao = match.group("tipo_funcao")
    funcao_objetivo = match.group("funcao")
    constantes, variaveis = extrair_constantes_e_variaveis(funcao_objetivo)

    assert tipo_funcao, f"Tipo de função inválido '{tipo_funcao}'"
    assert funcao_objetivo, f"Função objetivo inválida '{funcao_objetivo}'"
    assert constantes, f"Constantes inválidas '{constantes}'"
    assert variaveis, f"Variáveis inválidas '{variaveis}'"

    if VERBOSE:
        logger.debug(
            f"Tipo da função: {tipo_funcao}, Função objetivo: {funcao_objetivo}, Constantes: {str_list_fraction(constantes)}, Variaveis: {variaveis}"
        )
    
    return tipo_funcao, funcao_objetivo, constantes, variaveis

def extrai_restricao(restricao:str) -> tuple:
    """
    Extrai os componentes de uma restrição.
    Args:
        restricao (str): string da restrição, ex: "2x1 + π2 + 3x4 ≥ 2/3"
    Returns:
        Tuple:
            - constantes (list[Fraction]): coeficientes extraídos
            - variaveis (list[str]): nomes das variáveis
            - simbolo (str): símbolo de comparação
            - valor (Fraction): valor do lado direito da restrição
    """
    padrao = re.compile(
    r'(?P<restricao_lhs>('                     # ← início do grupo nomeado para lado esquerdo
        r'([-+]?\s*'
        r'(\d+(\.\d+)?|\d+/\d+)?'
        #r'\s*[^\W\d_]\d+\s*)+'                 # variável com número (ex: π1, x2)
        + regex_variable_expr +
        r')+'                 # variável com número (ex: π1, x2)
    r'))\s*'
    r'(?:'                                      # ← início da escolha entre dois caminhos (opcionalidade)
        r'(?P<simbolo_com_rhs><=|>=|<|>|=|≥|≤)'  # símbolo de comparação
        r'\s*'
        r'(?P<restricao_rhs>('
            r'([-+]?\s*(?:\d+/\d+|\d+(?:\.\d+)?))'
            r'|' 
            r'(?P<shatter>(\s*' + regex_variable_expr + r'\s*\-' + regex_variable_expr + r'))'
        r'))'
    r'|'
        r'(?P<simbolo_irrestrito>irrestrito)'   # literal "irrestrito", sem RHS
    r')\s*$',
    re.UNICODE
)
    match = re.search(padrao, restricao)
    assert match, f"Expressão inválida {restricao}"
    restricao_lhs = match.group("restricao_lhs")
    restricao_simbolo = match.group("simbolo_com_rhs") or match.group("simbolo_irrestrito")
    if restricao_simbolo == "irrestrito":
        restricao_rhs = 0
    elif match.group("shatter"):
        restricao_rhs = match.group("shatter")
    else:
        restricao_rhs = Fraction(match.group("restricao_rhs"))
    constantes_lhs, variaveis_lhs = extrair_constantes_e_variaveis(restricao_lhs)
    #logger.debug(f"Restrição LHS: {restricao_lhs}, Simbolo: {restricao_simbolo}, Valor: {restricao_rhs}")
    if VERBOSE:
        logger.debug(f"Constantes LHS: {str_list_fraction(constantes_lhs)}, Variáveis LHS: {variaveis_lhs} \n simbolo: {restricao_simbolo}, Valor: {restricao_rhs}")
    return constantes_lhs, variaveis_lhs, restricao_simbolo, restricao_rhs

def standard_display_variable(constante, variavel, first_var:bool, show_zero:bool = False, decimal:bool = False) -> str:
    if show_zero and first_var and constante == 0:
        return f"0{variavel}"
    
    if show_zero and first_var is False and show_zero and constante == 0:
        return f"+ 0{variavel}"

    if show_zero is False and constante == 0:
        return ""
            

    if not decimal:
        if first_var:
            if constante == 1:
                return f"{variavel}"
            elif constante == -1:
                return f"-{variavel}"
            else:
                return f"{constante}{variavel}"
        else:
            if constante == 1:
                return f"+ {variavel}"
            elif constante == -1:
                return f"- {variavel}"
            if constante > 0:
                return f"+ {constante}{variavel}"
            if constante < 0:
                return f"- {abs(constante)}{variavel}"
        
    else:
        if first_var:
            if constante == 1:
                return f"{variavel}"
            elif constante == -1:
                return f"-{variavel}"
            else:
                var = convert_if_str_float_is_int(str(float(constante)))
                return f"{var}{variavel}"
        else:
            if constante == 1:
                return f"+ {variavel}"
            elif constante == -1:
                return f"- {variavel}"
            elif constante > 0:
                var = convert_if_str_float_is_int(str(float(constante)))
                return f"+ {var}{variavel}"
            if constante < 0:
                var = convert_if_str_float_is_int(str(abs(float(constante))))
                return f"- {var}{variavel}"
        
    raise ValueError(f"Erro inesperado")

def monta_f_obj(tipo_funcao:str, constantes_e_variaveis:dict, standard_form:bool = False, 
                detailed:bool = False, decimal:bool = False) -> str:
    """
    Monta a função objetivo a partir dos componentes extraídos.
    Args:
        tipo_funcao (str): "max" ou "min"
        constantes (dict): coeficientes extraídos
        standard_form (bool): se True, retorna a função na forma padrão (MIN)
        detailed (bool): se True, retorna a função com termos em 0
        decimal (bool): se True transforma os números em decimal
    Returns:
        str: função objetivo montada
    """
    if standard_form:
        if tipo_funcao.lower() == "max":
            for variavel, constante in constantes_e_variaveis.items():
                constantes_e_variaveis[variavel] = -constante
            tipo_funcao = "min"
            
    
    # Ordenando o dicionario por prioridade, passivel de otimizacao...

    # 1. Labeling Passivel de otimizacao
    # 1. Labeling Passivel de otimizacao
    labeled = {}
    for variable, constant in constantes_e_variaveis.items():
        if variable[0] == "s":
            labeled[variable] = (constant,  display_variable_priority["slack"])

        elif variable[0] == "a":
            labeled[variable] = (constant,  display_variable_priority["artificial"])

        else:
            labeled[variable] = (constant,  display_variable_priority["normal"])

    # 2. Ordenando por prioridade && numero da variavel
    labeled = dict(
        sorted(
            labeled.items(),
            key=lambda x: x[1][1] + int(re.search(r'\d+', x[0]).group()),
        )
    )
   
    # 3. Atribuindo o dicionario passado com as variaveis na ordem correta
    constantes_e_variaveis.clear()
    for var in labeled:
        constantes_e_variaveis[var] = labeled[var][0]
    
                
    funcao_objetivo = f"{tipo_funcao} "
    for i, (variavel, constante) in enumerate(constantes_e_variaveis.items()):
        if funcao_objetivo == f"{tipo_funcao} ":
            primeira = True
        else:
            primeira = False

        if primeira:
            funcao_objetivo += standard_display_variable(constante, variavel, primeira, show_zero=detailed, decimal=decimal)
        else:
            variavel = standard_display_variable(constante, variavel, primeira, show_zero=detailed, decimal=decimal)
            if variavel != "":
                funcao_objetivo += " " + variavel
        
    logger.debug(f"Função objetivo montada: {funcao_objetivo}")
    return funcao_objetivo.strip()

def monta_restricao(constantes_e_variaveis_lhs:dict, simbolo:str, valor_rhs:Fraction, standard_form:bool = False,
                    detailed:bool = False, decimal:bool = False, slack_var:str = "s1",
                    unbounded_vars:str="x'1") -> tuple:
    """
    Monta a restrição a partir dos componentes extraídos.
    Args:
        constantes_e_variav|eis_lhs (dict): coeficientes extraídos do lado esquerdo
        simbolo (str): símbolo de comparação
        valor_rhs (Fraction): valor do lado direito da restrição
        standard_form (bool): se True, retorna a função na forma padrão (MIN) com a restricao no formato de =
        detailed (bool): se True, retorna a função com termos em 0
        decimal (bool): se True transforma os números em decimal
    Returns:
        tuple:
            str: restrição montada\n
            int: nada acontece = 0 
                variavel adicionada = 1, 
                variavel alterada = 2, 
                duas variaveis adicionadas = 3 (variaveis irrestritas)  
    """
    # Checando se é restrição x1 >= 0 ou irrestrita
    non_zero_var = 0
    constantes_lhs = list(constantes_e_variaveis_lhs.values())

    constant_foo = 0
    for variavel, constante in constantes_e_variaveis_lhs.items():
        if constante != 0:
            non_zero_var += 1
            constant_foo = constante
            variable_foo = variavel

    change_var = 0

    if simbolo == "irrestrito":
        if standard_form is False:
            if VERBOSE:
                logger.debug(f"{standard_display_variable(abs(constant_foo), variable_foo, True)} irrestrito")
            return f"{standard_display_variable(abs(constant_foo), variable_foo, True)} irrestrito", 0
        if standard_form is True:
            logger.warning(f"CUIDADO, condição de variavel puramente irrestrita NAO implementada")
            new_var1 = f"{variable_foo[0]}'{variable_foo[1:]}"
            new_var2 = f"{variable_foo[0]}''{variable_foo[1:]}"
            linha1 = f"{variable_foo} = {new_var1} - {new_var2}"
            linha2 = f"{standard_display_variable(1, new_var1, True)} ≥ 0"
            linha3 = f"{standard_display_variable(1, new_var2, True)} ≥ 0"
            return f"{linha1}\n{linha2}\n{linha3}", VARIAVEL_IRRESTRITA_MODIFICADA

    # Checando se é restrição x1 >= 0 ou <= 0
    if standard_form is True:
        if non_zero_var == 1:
            if (simbolo == ">=" or simbolo == "≥") and valor_rhs == 0 :
                if constant_foo < 0:
                    change_var = 2
                    lhs = standard_display_variable(abs(constant_foo), variable_foo, first_var=True, decimal=decimal)
                    if VERBOSE:
                        logger.debug(f"{lhs} {simbolo} {valor_rhs} {change_var}")
                    return f"{lhs} {simbolo} {valor_rhs}", change_var
                else:
                    lhs = standard_display_variable(constant_foo, variable_foo, first_var=True, decimal=decimal)
                    if VERBOSE:
                        logger.debug(f"{lhs} {simbolo} {valor_rhs} {change_var}")
                    return f"{lhs} {simbolo} {valor_rhs}", change_var
            elif (simbolo == "<=" or simbolo == "≤") and valor_rhs == 0:
                if constant_foo > 0:
                    change_var = 2
                    lhs = standard_display_variable(constant_foo, variable_foo, first_var=True, decimal=decimal)
                    if VERBOSE:
                        logger.debug(f"{lhs} >= {valor_rhs} {change_var}")
                    return f"{lhs} >= {valor_rhs}", change_var
                else:
                    lhs = standard_display_variable(abs(constant_foo), variable_foo, first_var=True, decimal=decimal)
                    logger.debug(f"{lhs} >= {valor_rhs} {change_var}")
                    return f"{lhs} >= {valor_rhs}", change_var

    # Deixando na forma de menor igual (<=)
    if standard_form is True:
        if simbolo == ">=" or simbolo == "≥":
            simbolo = "<="
            valor_rhs = -valor_rhs
            for variavel, constante in constantes_e_variaveis_lhs.items():
                constantes_e_variaveis_lhs[variavel] = -constante
        if simbolo != "=":
            constantes_e_variaveis_lhs[slack_var] = 1
            simbolo = "="
            change_var = 1

        # adicionando variáveis de folga

    # Ordenando o dicionario por prioridade

    # 1. Labeling Passivel de otimizacao
    labeled = {}
    for variable, constant in constantes_e_variaveis_lhs.items():
        if variable[0] == "s":
            labeled[variable] = (constant,  display_variable_priority["slack"])

        elif variable[0] == "a":
            labeled[variable] = (constant,  display_variable_priority["artificial"])

        else:
            labeled[variable] = (constant,  display_variable_priority["normal"])

    # 2. Ordenando por prioridade && numero da variavel
    labeled = dict(
        sorted(
            labeled.items(),
            key=lambda x: x[1][1] + int(re.search(r'\d+', x[0]).group()),
        )
    )
   
    # 3. Atribuindo o dicionario passado com as variaveis na ordem correta
    constantes_e_variaveis_lhs.clear()
    for var in labeled:
        constantes_e_variaveis_lhs[var] = labeled[var][0]
        
    lhs = ""
    # Primeiro, apenas desigualdades de <=
    for i, (variavel, constante) in enumerate(constantes_e_variaveis_lhs.items()):
        if lhs == "":
            primeira = True
        else:
            primeira = False

        if primeira:
            lhs += standard_display_variable(constante, variavel, primeira, show_zero=detailed, decimal=decimal)
        else:
            variavel = standard_display_variable(constante, variavel, primeira, show_zero=detailed, decimal=decimal)
            if variavel != "":
                lhs += " " + variavel
    if VERBOSE:
        logger.debug(f"{lhs.strip()} {simbolo} {valor_rhs} {change_var}")
    return f"{lhs.strip()} {simbolo} {valor_rhs}", change_var

    # Segundo, variávies de folga

def extract_variables_problem(f_obj:str="", constraints:list=[]) -> list:
    """ 
    Extrai as variáveis de um problema de programação linear.
    A funcao objetivo e as restricoes sao opcionais
    Args:
        f_obj (str): função objetivo
        restricoes (list): lista de restrições
    Returns:
        list: lista de variáveis
    """
    all_variables = []
    if f_obj:
        _ , _, _, variables = extrai_f_obj(f_obj)
        all_variables.extend(variables)
    if constraints:
        for constraint in constraints:
            _, variables_lhs, _, _ = extrai_restricao(constraint)
            for variable in variables_lhs:
                if variable not in all_variables:
                    all_variables.append(variable)
    return all_variables

def extract_ge_le_constraints(constraints:list, positive_lhs:bool = False) -> list:
    """
    Extrai as restrições do tipo x >= 0 ou x <= 0 de uma lista de restricoes.
    Args:
        constraints (list): lista de restrições
        positive_lhs (bool): se True, formas -x1 >= 0 se tornam x1 <= 0
    Returns:
        list: lista de restrições do tipo x >= 0 ou x <= 0
    """
    ge_le_constraints = []
    for constraint in constraints:
        constants_lhs, variables_lhs, symbol, value_rhs = extrai_restricao(constraint)
        if check_ge_zero(constants_lhs, value_rhs, symbol) or check_le_zero(constants_lhs, value_rhs, symbol):
            if positive_lhs:
                # Flipping the sign of the constant and symbol
                for constant in constants_lhs:
                    if constant < 0:
                        if symbol == ">=" or symbol == "≥":
                            symbol = "<="
                        elif symbol == "<=" or symbol == "≤":
                            symbol = ">="
                        break
                constraint = assemble_variables_constraints(variables_lhs, symbols=[symbol])[0]
            ge_le_constraints.append(constraint)
    return ge_le_constraints

def extract_variable_constraints(constraints:list, standard_form=False, positive_lhs=False) -> list:
    """
    Extrai as restricoes de variavel de uma lista de restricoes.\n
    restricoes da forma x >= 0 ou x <= 0 ou x irrestrito\n
    Args:
        restricoes (list): lista de restrições
        standard_form (bool): se True, as variáveis estão na forma padrão( >=0 )
        positive_lhs (bool): se True, formas -x1 >= 0 se tornam x1 <= 0 (NAO NECESSARIAMENTE FORMA PADRAO)
    Returns:
        list: lista de restrições que são variáveis
    """
    variable_constraints = []
    for constraint in constraints:
        check_var_constraint, _ = check_variable_constraint(constraint)
        if standard_form:
                constraint = variable_constraint_to_std_form(constraint)
                variable_constraints.append(constraint)
                
        elif positive_lhs:
            constants_lhs, variables_lhs, symbol, value_rhs = extrai_restricao(constraint)
            if check_ge_zero(constants_lhs, value_rhs, symbol) or check_le_zero(constants_lhs, value_rhs, symbol):
                for constant in constants_lhs:
                    if constant < 0:
                        if symbol == ">=" or symbol == "≥":
                            symbol = "<="
                        elif symbol == "<=" or symbol == "≤":
                            symbol = ">="
                        break
                constraint = assemble_variables_constraints(variables_lhs, symbols=[symbol])[0]
                variable_constraints.append(constraint)
            
        elif check_var_constraint:
            variable_constraints.append(constraint)
            
    return variable_constraints

def extract_non_var_constraints(constraints:list) -> list:
    """
    Extrai as restrições que não são variáveis de uma lista de restricoes.
    Args:
        restricoes (list): lista de restrições
    Returns:
        list: lista de restrições que não são variáveis
    """
    non_var_constraints = constraints.copy()
    remove_variable_constraints(non_var_constraints)
    return non_var_constraints

def extract_constraints_signs(restricoes:list) -> list:
    """
    Extrai os sinais das restrições de uma lista de restricoes.
    Args:
        restricoes (list): lista de restrições
    Returns:
        list: lista de sinais das restrições
    """
    constraints_signs = []
    for restricao in restricoes:
        _, _, symbol, _ = extrai_restricao(restricao)
        constraints_signs.append(symbol)
    return constraints_signs

def assemble_variables_constraints(variables:list, symbols:list = [], 
                                   standard_form:bool=False) -> list:
    
    """
    Monta restricoes do tipo "x1 >= 0", "x2 <= 0", "x3 irrestrito" "x4 = 0" a partir de variáveis e sinais.
    Difere de variable_constraint_to_std_form, pois nao trabalha com x1 = x'1 - x''1.\n
    e so monta restricoes realmente de >= 0 para as variaveis passadas
    Args:
        variables (list): lista de variáveis
        symbols (list): lista de sinais das restrições
        is_vars_on_standard_form (bool): se True, as variáveis estão na forma padrão( >=0 )
    Returns:
        list: lista de restricoes do tipo "x1 >= 0", "x2 <= 0", "x3 irrestrito" "x4 = 0"
    """
    variables_constraints = []
    if not standard_form:
        for variable, symbol in zip(variables, symbols):
            if symbol == "irrestrito":
                variables_constraints.append(f"{standard_display_variable(1, variable, first_var=True)} {symbol}")
            else:
                variables_constraints.append(f"{standard_display_variable(1, variable, first_var=True)} {symbol} 0")
    else:
        for variable in variables:
            variables_constraints.append(f"{standard_display_variable(1, variable, first_var=True)} >= 0")


    return variables_constraints

def str_problem_to_standard_form(f_obj:str, constraints:list, detailed:bool = False, decimal:bool = False) -> str:
    """ 
    Transforma um problema de programação linear em sua forma padrão.
    Args:
        problem (str): problema a ser transformado
        detailed (bool): se True, retorna a função com termos em 0
        decimal (bool): se True, retorna a função com termos em 0
    Returns:
        Tuple:
            - str: Funcao objetivo padrão
            - list: restrições padrão
    """
    restricoes = constraints
    
    # Transforma as restricoes primeiro
    # tipo_funcao, funcao_objetivo, constantes, variaveis = extrai_f_obj(problem[0])
    # constantes_e_variaveis = dict(zip(variaveis, constantes))
    tipo_funcao, _, constantes, variaveis = extrai_f_obj(f_obj)
    constantes_e_variaveis_fobj = dict(zip(variaveis, constantes))
    slack_var = 1
    new_restricoes = []
    variaveis_nao_descritas_na_f_obj = []
    variaveis_alteradas = []
    variaveis_irrestritas_modificadas = []
    
    for restricao in restricoes:
        constantes_lhs, variaveis_lhs, simbolo, valor_rhs = extrai_restricao(restricao)
        constantes_e_variaveis_lhs = dict(zip(variaveis_lhs, constantes_lhs))
        forma_padrao_restricao, change_var = monta_restricao(constantes_e_variaveis_lhs, simbolo, valor_rhs, standard_form=True,
                                                             detailed=detailed, decimal=decimal, slack_var="s" + str(slack_var))
        
        # Variaveis que nao estao na funcao objetivo
        for variable in variaveis_lhs:
            if variable not in constantes_e_variaveis_fobj:
                variaveis_nao_descritas_na_f_obj.append(variable)
        
        # Variaveis de folga
        if change_var == VARIAVEL_ADICIONADA:
            constantes_e_variaveis_fobj["s" + str(slack_var)] = 0
            slack_var += 1
        
        # Variaveis com sinal alterados
        elif change_var == VARIAVEL_ALTERADA:
            for variavel, constante in constantes_e_variaveis_lhs.items():
                if constante != 0:
                    variaveis_alteradas.append(variavel)
                # Trocando o sinal da variavel na funcao objetivo
                    # constantes_e_variaveis_fobj[variavel] = -constantes_e_variaveis_fobj[variavel]
                # Trocando sinal da variavel nas equacoes atuasi
                    # change_variable_sign_in_constraints(variavel, new_restricoes)
        
        # Variaveis irrestritas modificadas
        elif change_var == VARIAVEL_IRRESTRITA_MODIFICADA:
            for variavel, constante in constantes_e_variaveis_lhs.items():
                if constante != 0:
                # Trocando o sinal da variavel na funcao objetivo
                    variaveis_irrestritas_modificadas.append(variavel)
                    # constantes_e_variaveis_fobj[variavel] = -constantes_e_variaveis_fobj[variavel]
                # Trocando sinal da variavel nas equacoes
                    # change_variable_sign_in_constraints(variavel, new_restricoes)
            continue
        
        new_restricoes.append(forma_padrao_restricao)
        
        if VERBOSE:
            logger.debug(f"forma_padrao_restricao {restricao.lstrip()} SE TRANSFORMA EIN {forma_padrao_restricao}")
    #monta_f_obj(tipo_funcao, constantes_e_variaveis, standard_form=True, detailed=detailed, decimal=decimal)
    
    # Adicionando 0's nas variaveis que nao aparecem nas restricoes, mas estao na funcao objetivo
    if variaveis_nao_descritas_na_f_obj:
        for variavel in variaveis_nao_descritas_na_f_obj:
            constantes_e_variaveis_fobj[variavel] = 0
    
    # Modificando a f_objetivo e a lista de restricoes para variaveis marcadas como alteradas
    if variaveis_alteradas:
        for variavel in variaveis_alteradas:
            constantes_e_variaveis_fobj[variavel] = -constantes_e_variaveis_fobj[variavel]
            # _, _, constants, variables = extrai_f_obj(new_f)
            # constantes_e_variaveis_fobj = dict(zip(variables, constants))
            change_variable_sign_in_constraints(variavel, new_restricoes, detailed=detailed, standard_form=True)
    
    # Modificando a f_objetivo e a lista de restricoes para variaveis marcadas como irrestrita_modificadas
    if variaveis_irrestritas_modificadas:
        for variavel in variaveis_irrestritas_modificadas:
            current_f = monta_f_obj(tipo_funcao, constantes_e_variaveis_fobj, standard_form=False, detailed=True, decimal=decimal)
            new_f = change_unbounded_variable_format_in_f_obj(variavel, current_f, detailed=True)
            _, _, constants, variables = extrai_f_obj(new_f)
            constantes_e_variaveis_fobj = dict(zip(variables, constants))
            change_unbounded_variable_format_in_constraint(variavel, new_restricoes, detailed=detailed)
            #assemble_variables_constraints(variaveis_irrestritas_modificadas, standard_form=True)
    
    if detailed:
        restricoes = new_restricoes
        new_restricoes = []
        for restricao in restricoes:
            constantes_lhs, variaveis_lhs, simbolo, valor_rhs = extrai_restricao(str(restricao))
            constantes_e_variaveis_lhs = dict(zip(variaveis_lhs, constantes_lhs))
            constantes_e_variaveis_lhs = adding_zero_vars_on_constants_variables_dict(constantes_e_variaveis_fobj, constantes_e_variaveis_lhs)
            forma_padrao_restricao_detailed, _ = monta_restricao(constantes_e_variaveis_lhs, simbolo, valor_rhs, standard_form=True,
                                                                 detailed=detailed, decimal=decimal) 
            new_restricoes.append(forma_padrao_restricao_detailed)
            if VERBOSE:
                logger.debug(f"forma_padrao_restricao_detailed {restricao.lstrip()} SE TRANSFORMA EIN {forma_padrao_restricao_detailed}")
    f_obj = monta_f_obj(tipo_funcao, constantes_e_variaveis_fobj, standard_form=True, detailed=detailed, decimal=decimal)
    #logger.debug(f"Função objetivo antiga: \n{problem}")
    #logger.debug(f"\nFunção objetivo padrão: \n{std_problem}")
    return f_obj, new_restricoes

def str_problem_to_std_form_matrix (f_obj:str, constraints:list, standard_form:bool=False, 
                                    decimal:bool = False) -> tuple: 
    """ 
        Função para transformar um problema de programação linear em sua de matriz.
        Args:
            problem (str): problema a ser transformado
            decimal (bool): se True, retorna a função com termos em 0
        Returns:
            Tuple:
                - A (list): matriz de coeficientes
                - b (list): vetor de constantes
                - c (list): vetor de coeficientes da função objetivo
                - x (list): lista de variáveis
    """
    if standard_form:
        f_obj, constraints = str_problem_to_standard_form(f_obj, constraints, detailed=True, decimal=decimal)
    func_type, obj_func, constants, variables = extrai_f_obj(f_obj)
    # Matriz de coeficientes c
    c = constants
    relevant_constraints = constraints.copy()
    remove_ge_le_constraints(relevant_constraints)
    all_variables = extract_variables_problem(f_obj, constraints)
    all_constants = [0 for _ in range(len(all_variables))]
    all_constants_and_variables = dict(zip(all_variables, all_constants))
            
    # Matriz de coeficientes A
    A = []
    b = []
    for constraint in relevant_constraints:
        if standard_form:
            constants_lhs, variables_lhs, symbol, value_rhs = extrai_restricao(constraint)
        else:
            constants_lhs, variables_lhs, symbol, value_rhs = extrai_restricao(constraint)
            constants_and_variables_lhs = adding_zero_vars_on_constants_variables_dict(all_constants_and_variables, dict(zip(variables_lhs, constants_lhs)))
            constants_lhs = list(constants_and_variables_lhs.values())
        A.append(constants_lhs)
        b.append(value_rhs)
    # Mostra bonitinho no console
    # Mostra bonitinho no log (como string)
    logger.debug(f"Problema em matriz {display_matrix_f_obj(A, b, c, all_variables)}")
    return A, b, c, all_variables

def std_matrix_to_str_problem(A:list, b:list, c:list, x:list, tipo_funcao:str = "min", 
                              standard_form = False, restricoes_simbolos:list = None, 
                              decimal:bool = False, detailed:bool = False) -> str:
    """
    Converte uma matriz de programação linear de A,b,c e x para string.
    Espera-se todas as variáveis no vetor x, exceto se for exigido forma padrao
    Args:
        A (list): matriz de coeficientes
        b (list): vetor de constantes
        c (list): vetor de coeficientes da função objetivo
        x (list): lista de variáveis
        tipo_funcao (str): "max" ou "min"
        tipo_variaveis (list): lista o tipo 
        decimal (bool): se True, retorna a função com termos em 0
    Returns:
        str: f_obj,
        list: restrições
    """
    #assert tipo_funcao.lower() == "min", "Tipo de função inválido, deve ser 'min'"
    assert A, "Matriz A está vazia"
    assert b, "Vetor b está vazio"
    assert c, "Vetor c está vazio"
    
    forma_padrao = standard_form
    if forma_padrao:
        if tipo_funcao.lower() == "max":
            tipo_funcao = "min"
            c = [-i for i in c]
    
    constantes_e_variaveis_f_obj = dict(zip(x, c))
    f_obj = monta_f_obj(tipo_funcao, constantes_e_variaveis_f_obj, standard_form=forma_padrao, detailed=detailed, decimal=decimal)
    
    # Monta as restrições
    restricoes = []
    slack_var = 1
    for i in range(len(A)):
        constantes_lhs = A[i]
        variaveis_lhs = x
        
        if restricoes_simbolos:
            simbolo = restricoes_simbolos[i]
        
        else:
            simbolo = "="
        
        valor_rhs = b[i]
        constantes_e_variaveis_lhs = dict(zip(variaveis_lhs, constantes_lhs))
        forma_padrao_restricao, change_var = monta_restricao(constantes_e_variaveis_lhs, simbolo, valor_rhs, standard_form=forma_padrao,
                                                     detailed=detailed, decimal=decimal, slack_var="s" + str(slack_var))
        
        if change_var == VARIAVEL_ADICIONADA:
            constantes_e_variaveis_f_obj["s" + str(slack_var)] = 0
            slack_var += 1
            # Adicionando variaveis de folga
        
        if change_var == VARIAVEL_ALTERADA:
            for variavel, constante in constantes_e_variaveis_lhs.items():
                if constante != 0:
                    # Trocando o sinal da variavel na funcao objetivo
                    constantes_e_variaveis_f_obj[variavel] = -constantes_e_variaveis_f_obj[variavel]
                    # Trocando sinal de todas variaveis lhs
                    change_variable_sign_in_constraints(variavel, restricoes)
        
        restricoes.append(forma_padrao_restricao)
    
    # Monta o problema final
    if detailed:
        f_obj = monta_f_obj(tipo_funcao, constantes_e_variaveis_f_obj, standard_form=True, detailed=detailed, decimal=decimal)
        std_problem = f_obj + "\n"
        aux_restricoes = restricoes
        for restricao in restricoes:
            constantes_lhs, variaveis_lhs, simbolo, valor_rhs = extrai_restricao(str(restricao))
            constantes_e_variaveis_lhs = dict(zip(variaveis_lhs, constantes_lhs))
            constantes_e_variaveis_lhs = adding_zero_vars_on_constants_variables_dict(constantes_e_variaveis_f_obj, constantes_e_variaveis_lhs)
            forma_padrao_restricao_detailed, _ = monta_restricao(constantes_e_variaveis_lhs, simbolo, valor_rhs, standard_form=(standard_form, "s1"),
                                                                 detailed=detailed, decimal=decimal) 
            
            aux_restricoes.append(forma_padrao_restricao_detailed)
        restricoes = aux_restricoes
  
    return f_obj, restricoes
    logger.debug(f"Função objetivo antiga: {display_matrix_f_obj(A, b, c, x)}")
    logger.debug(f"Função objetivo de matriz para padrão: \n{std_problem.strip()}")
    return std_problem.strip()


""" Parte de testes """


def bateria_testes_utilitarios(test_standard_display_variable:bool=False,
                                test_check_ge_zero:bool=False,
                                test_check_le_zero:bool=False,
                                test_check_variable_constraint:bool=False,
                                test_check_variable_Constraint_std_form:bool=False,
                                test_extract_non_var_constraints:bool=False,
                                test_extract_variable_constraints:bool=False,
                                test_remove_variable_constraints:bool=False,
                                test_variable_constraints:bool=False,
                                test_remove_ge_le_constraints:bool=False,
                                test_change_variable_sign_in_f_obj:bool=False,
                                test_change_variable_sign_in_constraints:bool=False,
                                test_adding_zero_vars_on_constants_variables_dict:bool=False,
                                test_extract_variables_problem:bool=False,
                                test_extract_ge_le_constraints:bool=False,
                               ):
    logger.debug(f"Testes utilitarios")
    # Testes para standard_display_variable
    if test_standard_display_variable:
        t1 = {
            "constante":0, 
            "variavel":"x1", 
            "first_var":True, 
            "show_zero":True, 
            "decimal":False, 
            "result":"0x1"}
        t2 = {
            "constante": 0,
            "variavel": "x1",
            "first_var": False,
            "show_zero": True,
            "decimal": False,
            "result": "+ 0x1",
        }
        t3 = {
            "constante": 0,
            "variavel": "x1",
            "first_var": True,
            "show_zero": False,
            "decimal": False,
            "result": "",
        }
        t4 = {
            "constante": 0,
            "variavel": "x1",
            "first_var": False,
            "show_zero": False,
            "decimal": False,
            "result": "",
        }
        t5 = {
            "constante": 1,
            "variavel": "x1",
            "first_var": True,
            "show_zero": False,
            "decimal": False,
            "result": "x1",
        }
        t6 = {
            "constante": -1,
            "variavel": "x1",
            "first_var": True,
            "show_zero": False,
            "decimal": False,
            "result": "-x1",
        }
        t7 = {
            "constante": -2.5,
            "variavel": "x1",
            "first_var": True,
            "show_zero": False,
            "decimal": True,
            "result": "-2.5x1",
        }
        t8 = {
            "constante": Fraction(-5, 2),
            "variavel": "x1",
            "first_var": True,
            "show_zero": False,
            "decimal": True,
            "result": "-2.5x1",
        }
        t9 = {
            "constante": Fraction(5, 2),
            "variavel": "x1",
            "first_var": True,
            "show_zero": False,
            "decimal": True,
            "result": "2.5x1",
        }
        t10 = {
            "constante": Fraction(5, 2),
            "variavel": "x1",
            "first_var": False,
            "show_zero": False,
            "decimal": True,
            "result": "+ 2.5x1",
        }
        t11 = {
            "constante": -1,
            "variavel": "x1",
            "first_var": False,
            "show_zero": False,
            "decimal": True,
            "result": "- x1",
        }
        t12 = {
            "constante": 1,
            "variavel": "x1",
            "first_var": False,
            "show_zero": False,
            "decimal": True,
            "result": "+ x1",
        }
        t13 = {
            "constante": -1,
            "variavel": "x1",
            "first_var": False,
            "show_zero": False,
            "decimal": True,
            "result": "- x1",
        }
        t14 = {
            "constante": 2,
            "variavel": "x1",
            "first_var": False,
            "show_zero": False,
            "decimal": True,
            "result": "+ 2x1",
        }
        t15 = {
            "constante": -2,
            "variavel": "x1",
            "first_var": False,
            "show_zero": False,
            "decimal": True,
            "result": "- 2x1",
        }

        tests = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15]
        #tests = [t14]
        
        if VERBOSE:
            logger.info(f"Iniciando testes para standard_display_variable")

        for i, test in enumerate(tests):
            constante = test["constante"]
            variavel = test["variavel"]
            first_var = test["first_var"]
            show_zero = test["show_zero"]
            decimal = test["decimal"]
            result = test["result"]
            calculated = standard_display_variable(constante, variavel, first_var, show_zero, decimal)
            try:
                assert calculated == result
            except AssertionError as e:
                logger.error(f"Erro no teste {i+1}")
                logger.error(f"\nvalor calculado: {calculated}\nvalor  esperado: {result}")
                raise e

    # Testes para check_ge_zero
    if test_check_ge_zero:
        t1 = {"constraint":"2x1 + π2 + 3x4 ≥ 2/3", "result":False}
        t2 = {"constraint":"π1 + 2x2 ≤ 5.2", "result":False}
        t3 = {"constraint":"-x1 + p2 + s3 = -2", "result":False}
        t4 = {"constraint":"x1 irrestrito", "result":False}
        t5 = {"constraint":"x1 + x2 + x3 >= 0", "result":False}
        t6 = {"constraint":"x1 <= 0", "result":False}
        t7 = {"constraint":"x1 >= 0", "result":True}
        t8 = {"constraint":"0x1 >= 0", "result":False}
        t9 = {"constraint":"0x1 + x2 >= 0", "result":True}
        t10 = {"constraint":"-x1 >= 0", "result":True}
        t11 = {"constraint":"24x1 >= 0", "result":True}
        tests = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11]
        if VERBOSE:
            logger.info(f"Iniciando testes para check_ge_zero")

        for test in tests:
            constraint = test["constraint"]
            result = test["result"]
            constants, _, symbol, value_rhs = extrai_restricao(constraint)
            logger.debug(f"teste: {test}")
            assert check_ge_zero(constantes_lhs=constants, valor_rhs=value_rhs, simbolo=symbol) == result, f"Erro: {constraint} != {result}, teste: {test}"

    # Testes para check_le_zero
    if test_check_le_zero:
        t1 = {"constraint":"2x1 + π2 + 3x4 ≤ 2/3", "result":False}
        t2 = {"constraint":"π1 + 2x2 ≥ 5.2", "result":False}
        t3 = {"constraint":"-x1 + p2 + s3 = -2", "result":False}
        t4 = {"constraint":"x1 irrestrito", "result":False}
        t5 = {"constraint":"x1 + x2 + x3 <= 0", "result":False}
        t6 = {"constraint":"x1 >= 0", "result":False}
        t7 = {"constraint":"x1 <= 0", "result":True}
        t8 = {"constraint":"0x1 <= 0", "result":False}
        t9 = {"constraint":"0x1 + x2 <= 0", "result":True}
        t10 = {"constraint":"-x1 <= 0", "result":True}
        t11 = {"constraint":"24x1 <= 0", "result":True}
        tests = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11]

        if VERBOSE:
            logger.info(f"Iniciando testes para check_le_zero")

        for test in tests:
            constraint = test["constraint"]
            result = test["result"]
            constants, _, symbol, value_rhs = extrai_restricao(constraint)
            assert check_le_zero(constants_lhs=constants, value_rhs=value_rhs, symbol=symbol) == result, f"Erro: {constraint} != {result}"

    if test_check_variable_constraint:
        pass
    
    if test_check_variable_Constraint_std_form:
        pass
    
    if test_variable_constraints:   
        pass
    
    if test_remove_variable_constraints:
        pass
    
    if test_extract_variable_constraints:
        pass
    
    if test_extract_non_var_constraints:
        pass
    
    # Testes para remove_ge_le_constraints
    if test_remove_ge_le_constraints:
        t1 = "2x1 + π2 + 3x4 ≥ 2/3"
        t2 = "π1 + 2x2 ≤ 5.2"
        t3 = "-x1 + p2 + s3 = -2"
        t4 = "x1 irrestrito"
        t5 = "x1 + x2 + x3 >= 0"
        t6 = "x1 <= 0"
        t7 = "x1 >= 0"
        constraints = [t1, t2, t3, t4, t5, t6, t7]
        if VERBOSE:
            logger.info(f"Iniciando testes para remove_ge_le_constraints")
        remove_ge_le_constraints(constraints)
        for constraint in constraints:
            constants, _, symbol, value_rhs = extrai_restricao(constraint)
            if check_ge_zero(constants, value_rhs, symbol) or check_le_zero(constants, value_rhs, symbol):
                raise AssertionError(f"Erro: {constraint} não foi removida")

    # Testes para change_variable_sign_in_f_obj
    if test_change_variable_sign_in_f_obj:
        t1 = {"f_obj":"max 2x1 + π2 + 3x4", "variable":"x1", "result":"max -2x1 + π2 + 3x4"}
        t2 = {"f_obj":"max 2x1 + π2 + 3x4", "variable":"π2", "result":"max 2x1 - π2 + 3x4"}
        t3 = {"f_obj":"max 2x1 + π2 + 3x4", "variable":"x4", "result":"max 2x1 + π2 - 3x4"}
        t4 = {"f_obj":"MAX -π1 + 2x2", "variable":"π1", "result":"MAX π1 + 2x2"}
        t5 = {"f_obj":"MIN -π1 + 0x2", "variable":"x2", "result":"MIN -π1 + 0x2"}
        tests = [t1, t2, t3, t4, t5]
        if VERBOSE:
            logger.info(f"Iniciando testes para change_variable_sign_in_f_obj")
        for test in tests:
            f_obj = test["f_obj"]
            variable = test["variable"]
            result = test["result"]
            new_f = change_variable_sign_in_f_obj(variable, f_obj, detailed=True)
            assert new_f == result, f"Erro: {new_f} != {result}, teste: {test}"

    # Testes para change_variable_sign_in_constraints
    if test_change_variable_sign_in_constraints:
        t1 = {"constraints":["2x1 + π2 + 3x4 ≥ 2/3", 
                              "x1 >= 24", 
                              "-5x1 + π2 + 23x4 ≥ 2/3"],
                "variable":"x1", 
                "result":["-2x1 + π2 + 3x4 ≥ 2/3", 
                          "-x1 >= 24", 
                          "5x1 + π2 + 23x4 ≥ 2/3"]}
        tests = [t1]
        if VERBOSE:
            logger.info(f"Iniciando testes para change_variable_sign_in_constraints")
        for test in tests:
            constraints = test["constraints"]
            variable = test["variable"]
            result = test["result"]
            change_variable_sign_in_constraints(variable, constraints)
            for i in range(len(constraints)):
                assert constraints[i] == result[i], f"Erro: {constraints[i]} != {result[i]} teste: {test}"

    # Testes para adding_zero_vars_on_constants_variables_dict
    if test_adding_zero_vars_on_constants_variables_dict:
        # Implementar
        pass

    # Testes para extract_variables_problem
    if test_extract_variables_problem:
        t1 = {"f_obj":"max 2x1 + π2 + 3x4", 
              "constraints":["2x1 + π2 + 3x4 ≥ 2/3", "x1 >= 24"], 
              "result":["x1", "π2", "x4"]
              }

        t2 = {"f_obj":"max 2x1 + π2 + 3x4",
              "constraints":[], 
              "result":["x1", "π2", "x4",]
              }

        t3 = {"f_obj":"",
              "constraints":["2x1 + π2 + 3x4 ≥ 2/3", "x1 >= 24", "x5 = 0"], 
              "result":["x1", "π2", "x4", "x5"]
              }
        tests = [t1, t2, t3]
        if VERBOSE:
            logger.info(f"Iniciando testes para extract_variables_problem")
        for test in tests:
            f_obj = test["f_obj"]
            constraints = test["constraints"]
            result = test["result"]
            variables = extract_variables_problem(f_obj, constraints)
            try:
                assert variables == result
            except AssertionError:
                logger.error(f"Erro: {variables} != {result}, teste: {test}")
                logger.error(f"\nvalor calculado:{variables}\nvalor  esperado:{result}")
                raise

    # Testes para extrair_ge_le_constraints
    if test_extract_ge_le_constraints:
        t1 = {"constraints":[
            "2x1 + π2 + 3x4 ≥ 2/3",
            "π1 + 2x2 ≤ 5.2",
            "x3 <= 0",
            "x1 + x2 + x3 >= 0",
            "-x1 <= 0",
            "p2 >= 0"
            ],
            "positive_lhs":False,
            "result":[
                "x3 <= 0",
                "-x1 <= 0",
                "p2 >= 0"
            ]
              }
        t2 = {"constraints":[
            "2x1 + π2 + 3x4 ≥ 2/3",
            "π1 + 2x2 ≤ 5.2",
            "-p2 >= 0",
            "x1 + x2 + x3 >= 0",
            "-x3 >= 0",
            "-x1 <= 0"
            ],
            "positive_lhs":True,
            "result":[
                "p2 <= 0",
                "x3 <= 0",
                "x1 >= 0"
            ]
              }
        tests = [t1, t2]

        if VERBOSE:
            logger.info(f"Iniciando testes para extrair_ge_le_constraints")
        for test in tests:
            constraints = test["constraints"]
            positive_lhs = test["positive_lhs"]
            result = test["result"]
            ge_le_constraints = extract_ge_le_constraints(constraints, positive_lhs=positive_lhs)
            for i in range(len(ge_le_constraints)):
                try:
                    assert ge_le_constraints[i] == result[i]
                except AssertionError:
                    logger.error(f"Erro: {ge_le_constraints[i]} != {result[i]}, teste: {test}")
                    logger.error(f"\nvalor calculado:{ge_le_constraints}\nvalor  esperado:{result}")
                    raise

def bateria_testes_str_padrao_problema(test_extrai_f_obj:bool = False, 
                   test_extrai_restricao:bool = False,
                   test_monta_f_obj:bool = False, 
                   test_monta_restricao:bool = False, 
                   test_extract_variables_problem:bool = False, 
                   test_forma_padrao:bool = False, 
                   test_problema_padrao_matriz:bool = False, 
                   test_matriz_para_problema_padrao:bool = False,
                   test_extract_constraints_signs:bool = False,
                   test_assemble_variables_constraints:bool = False,
                   ):
    
    # Testes para extrair_f_obj
    if test_extrai_f_obj:
        logger.info(f"Iniciando testes para extrair_f_obj")
        teste1 = "max 3/2π1 + 2y2"
        teste2 = "MIN 3/2Φ1 + 2Φ2 + 3Φ3"
        teste3 = "MIN 1.5x1"
        teste4 = "MIN -1.5x1 + 0x2 + 0x3"
        teste5 = "MAX - 1.5x1 + 0x2 + 0x3"
        teste6 = "max -3/2x1 - 2x2 + 0x3"
        teste7 = "max 2.1x''1 + 3.2x'2 + 2.4Φ''3"

        assert extrai_f_obj(teste1) == ("max", "3/2π1 + 2y2", [Fraction(3, 2), Fraction(2)], ["π1", "y2"])
        assert extrai_f_obj(teste2) == ("MIN", "3/2Φ1 + 2Φ2 + 3Φ3", [Fraction(3, 2), Fraction(2), Fraction(3)], ["Φ1", "Φ2", "Φ3"])
        assert extrai_f_obj(teste3) == ("MIN", "1.5x1", [Fraction(3, 2)], ["x1"])
        assert extrai_f_obj(teste4) == ("MIN", "-1.5x1 + 0x2 + 0x3", [Fraction(-3, 2), Fraction(0, 1), Fraction(0, 1)], ["x1", "x2", "x3"])
        assert extrai_f_obj(teste5) == ("MAX", "- 1.5x1 + 0x2 + 0x3", [Fraction(-3, 2), Fraction(0, 1), Fraction(0, 1)], ["x1", "x2", "x3"])
        assert extrai_f_obj(teste6) == ("max", "-3/2x1 - 2x2 + 0x3", [Fraction(-3, 2), Fraction(-2), Fraction(0, 1)], ["x1", "x2", "x3"])
        assert extrai_f_obj(teste7) == ("max", "2.1x''1 + 3.2x'2 + 2.4Φ''3", [Fraction(21, 10), Fraction(16, 5), Fraction(12, 5)], ["x''1", "x'2", "Φ''3"])

    # Testes para extrair_restricao
    if test_extrai_restricao:
        logger.info(f"Iniciando testes para extrair_restricao")
        t1 = "2x1 + π2 + 3x4 ≥ 2/3"
        t2 = "π1 + 2x2 ≤ 5.2"
        t3 = "-x1 + p2 + s3 = -2"
        t4 = "x1 irrestrito"
        t5 = "x''1 + x'2 + x3 ≥ 0"

        # print(extrai_restricao(t5))
        assert extrai_restricao(t1) == ([Fraction(2, 1), Fraction(1, 1), Fraction(3, 1)], ["x1", "π2", "x4"], "≥", 
                                        Fraction(2, 3))
        assert extrai_restricao(t2) == ([Fraction(1, 1), Fraction(2, 1)], ["π1", "x2"], "≤", Fraction(26, 5))
        assert extrai_restricao(t3) == ([Fraction(-1, 1), Fraction(1, 1), Fraction(1, 1)], ["x1", "p2", "s3"], "=", Fraction(-2, 1))
        assert extrai_restricao(t4) == ([1],["x1"], "irrestrito", 0)
        assert extrai_restricao(t5) == ([Fraction(1, 1), Fraction(1, 1), Fraction(1, 1)], ["x''1", "x'2", "x3"], "≥", Fraction(0, 1))

    # Testes para monta_f_obj
    if test_monta_f_obj:
        logger.info("Iniciando testes para monta_f_obj")
        t1 = {
                "tipo_funcao": "max",
                "constantes_e_variaveis": {
                    "π1": Fraction(3, 2),
                    "y2": Fraction(2)
                    },
                "standard_form": False,
                "detailed": True, 
                "decimal": False,
                "result": "max 3/2π1 + 2y2"
            }
        t2 = {
                "tipo_funcao": "min",
                "constantes_e_variaveis": {
                    "x1": Fraction(-3, 2),
                    "x2": Fraction(2),
                    "x3": Fraction(0, 1)
                    },
                "standard_form": True,
                "detailed": True,
                "decimal": False,
                "result": "min -3/2x1 + 2x2 + 0x3"
            }
        t3 = {
                "tipo_funcao": "min",
                "constantes_e_variaveis": {
                    "x1": Fraction(-3, 2),
                    "x2": Fraction(2),
                    "x3": Fraction(0, 1)
                    },
                "standard_form": True,
                "detailed": False,
                "decimal": False,
                "result": "min -3/2x1 + 2x2"
            }
        t4 = {
                "tipo_funcao": "max",
                "constantes_e_variaveis": {
                    "x1": -4.1
                    },
                "standard_form": False,
                "detailed": True,
                "decimal": True,
                "result": "max -4.1x1"
            }
        t5 = {
                "tipo_funcao": "max",
                "constantes_e_variaveis": {
                    "x1": -1.0
                    },
                "standard_form": False,
                "detailed": True,
                "decimal": True,
                "result": "max -x1"
            }
        t6 = {
                "tipo_funcao": "max",
                "constantes_e_variaveis": {
                    "x1": 1.0,
                    },
                "standard_form": False,
                "detailed": True,
                "decimal": True,
                "result": "max x1" 
            }
        t7 = {
                "tipo_funcao": "max",
                "constantes_e_variaveis": {
                    "x1": 1.0,
                    "x2": Fraction(3,2),
                    "x3": -1.0
                    },
                "standard_form": True,
                "detailed": True,
                "decimal": True,
                "result": "min -x1 - 1.5x2 + x3" 
            }
        t8 = {
                "tipo_funcao": "max",
                "constantes_e_variaveis": {
                    "x'1": 1.0,
                    "x''2": Fraction(3,2),
                    "x3": -1.0
                    },
                "standard_form": True,
                "detailed": True,
                "decimal": True,
                "result": "min -x'1 - 1.5x''2 + x3" 
            }

        tests = [t1, t2, t3, t4, t5, t6, t7, t8]
        for i, test in enumerate(tests):
            tipo_funcao = test["tipo_funcao"]
            constantes_e_variaveis = test["constantes_e_variaveis"]
            standard_form = test["standard_form"]
            detailed = test["detailed"]
            decimal = test["decimal"]
            result = test["result"]
            try:
                calculated = monta_f_obj(tipo_funcao, constantes_e_variaveis, standard_form=standard_form, detailed=detailed, decimal=decimal)
                assert calculated == result
            except AssertionError as e:
                logger.error(f"Erro no teste: {i + 1}")
                logger.error(f"\nvalor calculado: {calculated}\nvalor  esperado: {result}")
                raise e

    # Testes para extract_variables_problem
    if test_extract_variables_problem:
        logger.info("Iniciando testes para extract_variables_problem")
        t1 = {
            "f_obj": "min x1 + 2x2",
            "constraints": [
                "2x1 + x2 ≥ 2/3",
                "x1 + x2 ≥ 1",
                "x2 = 2",
                "x1 >= 0",
                "x2 <= 0"
            ],
            "result": ["x1", "x2"]
        }

        t2 = {
            "f_obj": "max π1 + 2π2",
            "constraints": [
                "2π1 + π2 ≥ 4",
                "7π1 + π2 <= 1",
                "-π2 = 2",
                "π1 <= 0",
                "π2 >= 0"
            ],
            "result": ["π1", "π2"]
        }

        t3 = {
            "f_obj": "max π1 + 2π2",
            "constraints": [
                "2π1 + π2 ≥ 4",
                "7π1 + π2 <= 1",
                "-π2 = 2",
                "π1 <= 0",
                "π2 irrestrito"
            ],
            "result": ["π1", "π2"]
        }
        tests = [t1, t2, t3]
        for test in tests:
            f_obj = test["f_obj"]
            constraints = test["constraints"]
            result = test["result"]
            variables = extract_variables_problem(f_obj, constraints)
            try:
                assert variables == result
            except AssertionError as e:
                logger.error(f"Erro no teste: {test}, valor calculado: {variables}, valor esperado: {result}")
                raise e

    # Testes para monta_restricao
    if test_monta_restricao:
        logger.info("Iniciando testes para monta_restricao")

        t1 = {
            "expr": "2x1 + π2 + 3x4 ≥ 2/3",
            "standard_form": False,
            "detailed": True,
            "decimal": False,
            "slack_var": "s1",
            "result": ("2x1 + π2 + 3x4 ≥ 2/3", 0)
        }
        t2 = {
            "expr": "2x1 + π2 + 3x4 ≤ 2/3",
            "standard_form": False,
            "detailed": True,
            "decimal": False,
            "slack_var": "s1",
            "result": ("2x1 + π2 + 3x4 ≤ 2/3", 0)
        }
        t3 = {
            "expr": "2x1 + π2 + 3x4 = 2/3",
            "standard_form": False,
            "detailed": True,
            "decimal": False,
            "slack_var": "s1",
            "result": ("2x1 + π2 + 3x4 = 2/3", 0)
        }
        t4 = {
            "expr": "2x1 + π2 + 3x4 ≥ 2/3",
            "standard_form": True,
            "detailed": True,
            "decimal": False,
            "slack_var": "s1",
            "result": ("-2x1 - π2 - 3x4 + s1 = -2/3", VARIAVEL_ADICIONADA)
        }
        t5 = {
            "expr": "x1 >= 0",
            "standard_form": False,
            "detailed": False,
            "decimal": False,
            "slack_var": "s1",
            "result": ("x1 >= 0", 0)
        }
        t6 = {
            "expr": "x1 <= 0",
            "standard_form": False,
            "detailed": False,
            "decimal": False,
            "slack_var": "s1",
            "result": ("x1 <= 0", 0)
        }
        t7 = {
            "expr": "x1 <= 0",
            "standard_form": True,
            "detailed": False,
            "decimal": False,
            "slack_var": "s1",
            "result": ("x1 >= 0", VARIAVEL_ALTERADA)
        }
        t8 = {
            "expr": "x1 = 0",
            "standard_form": False,
            "detailed": False,
            "decimal": False,
            "slack_var": "s1",
            "result": ("x1 = 0", 0)
        }
        t9 = {
            "expr": "π2 irrestrito",
            "standard_form": False,
            "detailed": False,
            "decimal": False,
            "slack_var": "s1",
            "result": ("π2 irrestrito", 0)
        }
        t10 = {
            "expr": "π2 irrestrito",
            "standard_form": False,
            "detailed": False,
            "decimal": False,
            "slack_var": "s1",
            "result": ("π2 = π''2 - π'1", VARIAVEL_IRRESTRITA_MODIFICADA)
        }

        testes = [
            t1, t2, t3, t4, t5, t6, t7, t8, t9
        ]

        for i, test in enumerate(testes):
            # TODO: PAREI AQUI CARAMBA
            expr = test["expr"]
            constants_lhs, variables_lhs, symbol, value_rhs = extrai_restricao(expr)
            constants_and_variables_lhs = dict(zip(variables_lhs, constants_lhs))
            standard_form = test["standard_form"]
            detailed = test["detailed"]
            decimal = test["decimal"]
            slack_var = test["slack_var"]
            result = test["result"]

            calculated_expr, s1_value = monta_restricao(constants_and_variables_lhs,
                                                        simbolo=symbol,
                                                        valor_rhs=value_rhs, 
                                                        standard_form=standard_form, 
                                                        detailed=detailed, 
                                                        decimal=decimal,
                                                        slack_var=slack_var)

            try:
                assert calculated_expr == result[0]
                assert s1_value == result[1]
            except AssertionError as e:
                logger.error(f"Erro no teste: {i + 1}")
                logger.error(f"\nvalor calculado: {calculated_expr}\nvalor esperado: {result[0]}")
                logger.error(f"s1_value calculado: {s1_value},\n s1_value esperado: {result[1]}")
                raise e

    # Testes para str_problem_to_standard_form
    if test_forma_padrao:
        logger.info("Iniciando testes para str_problem_to_standard_form")
        # Teste 1
        t1 = {
                "f_obj":"min x1 + 2x2",
                "constraints":["2x1 + x2 ≥ 2/3",
                    "x1 + x2 ≥ 1",
                    "x2 = 2",
                    "x1 >= 0",
                    "x2 <= 0"], 
                "detailed": False, 
                "decimal": False,
                "result":{
                        "f_obj":"min x1 - 2x2",
                        "constraints":["-2x1 + x2 + s1 = -2/3",
                                        "-x1 + x2 + s2 = -1",
                                        "-x2 = 2",
                                        "x1 >= 0",
                                        "x2 >= 0"]}
                }
        t2 = {
            "f_obj": "max π1 + 2π2",
            "constraints": [
                "2π1 + π2 ≥ 4",
                "7π1 + π2 <= 1",
                "-π2 = 2",
                "π1 <= 0",
                "π2 >= 0"
            ],
            "detailed": True,
            "decimal": False,
            "result": {
                "f_obj": "min π1 - 2π2 + 0s1 + 0s2",
                "constraints": [
                    "2π1 - π2 + s1 + 0s2 = -4",
                    "-7π1 + π2 + 0s1 + s2 = 1",
                    "0π1 - π2 + 0s1 + 0s2 = 2",
                    "π1 >= 0",
                    "π2 >= 0"
                ]
            }
        }
        t3 = {
            "f_obj": "Min 2x1 + x2 + 3x4",
            "constraints": [
                "2x1 + x2 + x4 <= 2/3",
                "x1 + x2 + x3 <= 1",
                "x1 >= 0",
                "x2 <= 0",
                "x3 >= 0",
                "x4 irrestrito",
            ],
            "detailed": True,
            "decimal": False,
            "result": {
                "f_obj": "Min 2x1 - x2 + 0x3 + 3x'4 - 3x''4 + 0s1 + 0s2",
                "constraints": [
                    "2x1 - x2 + 0x3 + x'4 - x''4 + s1 + 0s2 = 2/3",
                    "x1 - x2 + x3 + 0x'4 + 0x''4 + 0s1 + s2 = 1",
                    "x1 >= 0",
                    "x2 >= 0",
                    "x3 >= 0", 
                    "x'4 >= 0",
                    "x''4 >= 0",
                ],
            },
        }
        t4 = {
            "f_obj": "Max 2x1 + x2 + 3x4",
            "constraints": [
                "2x1 + x2 + x4 <= 2/3",
                "x1 - x2 + x3 <= 1",
                "x1 >= 0",
                "x2 irrestrito",
                "x3 >= 0",
                "x4 >= 0 ",
            ],
            "detailed": False,
            "decimal": False,
            "result": {
                "f_obj": "min -2x1 - x'2 + x''2 - 3x4",
                "constraints": [
                    "2x1 + x'2 - x''2 + x4 + s1 = 2/3",
                    "x1 - x'2 + x''2 + x3 + s2 = 1",
                    "x1 >= 0",
                    "x3 >= 0",
                    "x4 >= 0",
                    "x'2 >= 0",
                    "x''2 >= 0",
                ],
            },
        }
        # Implementar esse caso t5 se for necessário
        t5 = {
            "f_obj": "Max 2x1 + x2 + 3x4",
            "constraints": [
                "2x1 + x2 + x4 <= 2/3",
                "x1 - x2 + x3 <= 1",
                "x1 >= 0",
                "x2 = x'2 - x''2",
                "x3 >= 0",
                "x4 >= 0 ",
            ],
            "detailed": False,
            "decimal": False,
            "result": {
                "f_obj": "Min -2x1 - x'2 + x''2 + 3x4",
                "constraints": [
                    "2x1 + x'2 - x''2 + x4 + s1 = 2/3",
                    "x1 - x'2 + x''2 + x3 + s2 = 1",
                    "x1 >= 0",
                    "x'2 >= 0",
                    "x''2 >= 0",
                    "x3 >= 0",
                    "x4 >= 0",
                ],
            },
        }

        # TODO:
       
        tests = [t1, t2, t3, t4]
        # problems = [problem2]
        for i, test in enumerate(tests):
            f_obj = test["f_obj"]
            constraints = test["constraints"]
            detailed = test["detailed"]
            decimal = test["decimal"]
            result = test["result"]

            f_obj, constraints = str_problem_to_standard_form(f_obj, constraints, detailed=detailed, decimal=decimal)

            try:
                assert f_obj == result["f_obj"]
                assert constraints == result["constraints"]
            except AssertionError as e:
                logger.error(f"Erro no teste: {i + 1}")
                logger.error(f"\nvalor calculado: {f_obj.replace(' ','_')} \nvalor esperado: {result['f_obj'].replace(' ','_')}")
                # logger.error(f"\nf_obj calculada:{f_obj.replace(' ',"_")} ")
                
                for i in range(len(constraints)):
                    logger.error(f"\nrestrição calculada: {constraints[i].replace(' ','_')} \nrestrição esperada : {result['constraints'][i].replace(' ','_')}")
                    if constraints[i] != result["constraints"][i]:
                        logger.error(f"\n*******Erro na restrição {i + 1}*****\n {constraints[i]} != {result['constraints'][i]}")
                raise e

    # Testes para str_problem_to_std_form_matrix
    if test_problema_padrao_matriz:
        logger.info("Iniciando testes para std_matrix_to_str_problem")
        # Teste 1
        t1 = {
            "f_obj": "min x1 + 2x2",
            "constraints": [
                "2x1 + x2 ≥ 2/3",
                "x1 + x2 ≥ 1",
                "x2 = 2",
                "x1 >= 0",
                "x2 >= 0"
            ],
            "standard_form": True,
            "decimal": False,
            "result": {
                "c": [1, 2, 0, 0],
                "b": [Fraction(-2, 3), -1, 2],
                "A": [[-2, -1, 1, 0],
                    [-1, -1, 0, 1],
                    [0, 1, 0, 0]],
                "x": ["x1", "x2", "s1", "s2"]
            }
        }
        t2 = {
            "f_obj": "max π1 + 2π2",
            "constraints": [
                "2π1 + π2 ≥ 4",
                "7π1 + π2 <= 1",
                "-π2 = 2",
                "π1 >= 0",
                "π2 >= 0"
            ],
            "standard_form": False,
            "decimal": False,
            "result": {
                "c": [Fraction(1, 1), Fraction(2, 1)],
                "b": [Fraction(4, 1), Fraction(1, 1), Fraction(2, 1)],
                "A": [[2, 1],
                     [7, 1],
                     [0, -1]],
                "x": ["π1", "π2"]
            }
        }

        tests = [t1, t2]

        for test in tests:
            f_obj = test["f_obj"]
            constraints = test["constraints"]
            standard_form = test["standard_form"]
            decimal = test["decimal"]
            result = test["result"]

            A, b, c, x = str_problem_to_std_form_matrix(f_obj, constraints, standard_form,decimal)

            try:
                assert c == result["c"]
                assert b == result["b"]
                assert A == result["A"]
                assert x == result["x"]
            except AssertionError as e:
                logger.error(f"Erro no teste: {test['f_obj']}")
                logger.error(f"\nvalor calculado: {c}\nvalor  esperado: {result['c']}")
                logger.error(f"\nrestrições calculadas: {b}\nrestrições  esperadas: {result['b']}")
                raise e

    # Testes para std_matrix_to_str_problem
    if test_matriz_para_problema_padrao:
        logger.info("Iniciando testes para std_matrix_to_str_problem")

        """ problem_structure (
            [A],
            [b],
            [c],
            [x],
            tipo_funcao,
            [restricoes_simbolos],
        )
            
             """
        t1 = {
            "A":[
                [-2, -1, 1, 0],
                [-1, -1, 0, 1],
                [0, 1, 0, 0]
            ],
            "b":[Fraction(-2, 3), -1, 2],
            "c":[1, 2, 0, 0],
            "x":["x1", "x2", "s1", "s2"],
            "standard_form": True,
            "tipo_funcao":"min",
            "restricoes_simbolo": [],
            "detailed": False,
            "decimal": False,
            "result": {
                "f_obj": "min x1 + 2x2",
                "constraints": [
                    "-2x1 - x2 + s1 = -2/3",
                    "-x1 - x2 + s2 = -1",
                    "x2 = 2"
                ]
            }
        }
        t2 ={ 
            "A":[
                [-2, -1, 1, 0],
                [-1, -1, 0, 1],
                [0, 1, 0, 0]
            ],
            "b":[Fraction(-2, 3), -1, 2],
            "c":[1, 2, 0, 0],
            "x":["x1", "x2", "x3", "x4"],
            "standard_form": True,
            "tipo_funcao":"max",
            "restricoes_simbolo":["<=", "<=", "<="],
            "detailed": False,
            "decimal": False,
            "result": {
                "f_obj": "min -x1 - 2x2",
                "constraints": [
                    "-2x1 - x2 + x3 + s1 = -2/3",
                    "-x1 - x2 + x4 + s2 = -1",
                    "x2 + s3 = 2"
                ]
            }
        }
        t3 ={ 
            "A":[
                [-2, -1, 1, 0],
                [-1, -1, 0, 1],
                [0, 1, 0, 0]
            ],
            "b":[Fraction(-2, 3), -1, 2],
            "c":[1, 2, 0, 0],
            "x":["x1", "x2", "x3", "x4"],
            "standard_form": False,
            "tipo_funcao":"max",
            "restricoes_simbolo":["<=", "<=", "<="],
            "detailed": False,
            "decimal": False,
            "result": {
                "f_obj": "max x1 + 2x2",
                "constraints": [
                    "-2x1 - x2 + x3 <= -2/3",
                    "-x1 - x2 + x4 <= -1",
                    "x2 <= 2"
                ]
            }
        }

        tests = [t2]

        for i, test in enumerate(tests):
            A = test["A"]
            b = test["b"]
            c = test["c"]
            x = test["x"]
            standard_form = test["standard_form"]
            tipo_funcao = test["tipo_funcao"]
            restricoes_simbolo = test["restricoes_simbolo"]
            detailed = test["detailed"]
            decimal = test["decimal"]

            f_obj, constraints = std_matrix_to_str_problem(A, b, c, x, tipo_funcao=tipo_funcao,
                                                           standard_form=standard_form,
                                                           restricoes_simbolos=restricoes_simbolo,
                                                           detailed=detailed, decimal=decimal)

            try:
                assert f_obj == test["result"]["f_obj"]
                assert constraints == test["result"]["constraints"]
            except AssertionError as e:
                logger.error(f"Erro no teste: {i + 1}")
                logger.error(f"valor calculado:\n {f_obj},\n valor esperado: {test['result']['f_obj']}")
                for i in range(len(constraints)):
                    logger.error(f"\nrestrição calculada: {constraints[i].replace(' ','_')},\nrestrição esperada : {test['result']['constraints'][i].replace(' ','_')}")
                    if constraints[i] != test["result"]["constraints"][i]:
                        logger.error(f"\n*******Erro na restrição {i + 1}*****\n {constraints[i]} != {test['result']['constraints'][i]}")
                raise e

    # Testes para extract_constraints_signs
    if test_extract_constraints_signs:
        logger.info("Iniciando testes para extract_constraints_signs")
        t1 = "2x1 + π2 + 3x4 ≥ 2/3"
        t2 = "π1 + 2x2 <= 5.2"
        t3 = "-x1 + p2 + s3 = -2"
        t4 = "x1 irrestrito"
        tests = [t1, t2, t3, t4]

        assert extract_constraints_signs(tests) == ["≥", "<=", "=", "irrestrito"]

    # Testes para assemble_variables_constraints
    if test_assemble_variables_constraints:
        logger.info("Iniciando testes para assemble_variables_constraints")
        t1 = {"variables":["x1","x2","x3","x4"] ,"symbols":["≥", "<=", "=", "irrestrito"], "is_vars_on_standard_form":False, 
              "result": ["x1 ≥ 0","x2 <= 0","x3 = 0","x4 irrestrito"]}
        t2 = {"variables":["π1","π2","π3","π4"] ,"symbols":["≤", ">=", "=", "irrestrito"], "is_vars_on_standard_form":False,
              "result": ["π1 ≤ 0","π2 >= 0","π3 = 0","π4 irrestrito"]}
        t3 = {"variables":["x1","x2","x3","x4"] ,"symbols":[], "is_vars_on_standard_form":True, 
              "result": ["x1 >= 0","x2 >= 0","x3 >= 0","x4 >= 0"]}
        tests = [t1, t2, t3]

        for test in tests:
            variables = test["variables"]
            symbols = test["symbols"]
            is_vars_on_standard_form = test["is_vars_on_standard_form"]
            result = test["result"]
            try:
                assert assemble_variables_constraints(variables, symbols, is_vars_on_standard_form) == result
            except AssertionError as e:
                logger.error(f"Erro no teste: {test}\nvalor calculado: {assemble_variables_constraints(variables, symbols)}\nvalor  esperado: {result}")
                raise e        

def check_health_status():
    logger.level = logging.INFO
    logger.info("Iniciando com os testes utilitarios ...")
    try:
        bateria_testes_utilitarios(True, True, True, True, True, True, True, True, True)
        logger.info("Testes utilitarios passaram com sucesso!")
        logger.info("Iniciando com os testes de str_padrao_problema ...")
        bateria_testes_str_padrao_problema(True, True, True, True, True, True, True, True, True, True)
        logger.info("Todos os testes passaram com sucesso!")
    except Exception as e:
        logger.error("Erro nos testes utilitarios ou de str_padrao_problema")
        logger.error(e)
        raise e

bateria_testes_utilitarios(
    test_check_variable_constraint=True,
    test_check_variable_Constraint_std_form=True,
    test_variable_constraints=True,
    test_remove_variable_constraints=True,
    test_extract_variable_constraints=True,
    test_extract_non_var_constraints=True,
)

# bateria_testes_str_padrao_problema(test_extrai_f_obj=True,
#                                    test_extrai_restricao=True,
#                                    test_monta_f_obj=True,
#                                    test_monta_restricao=True,)


# bateria_testes_str_padrao_problema(test_forma_padrao=True)


# check_health_status()
# print(f"f_ = {extrai_f_obj(f_obj['func'])}, restricoes = {extrai_restricao(f_obj['restricoes'][0])}")

# bateria_testes_str_padrao_problema(test_monta_f_obj=True)

# bateria_testes_utilitarios(test_standard_display_variable=True)

# bateria_testes_str_padrao_problema(teste_forma_padrao=True,teste_problema_padrao_matriz=True)

# check_health_status()


# print(extrai_restricao("pi1 irrestrito"))

# str_primal_to_dual("")

# std_matrix_to_str_problem([], [], [], [], tipo_funcao="max", decimal=False)

# str_problem_to_std_form_matrix("min 3/2x1 + 2x2\n2x1 + x2 + 3x4 ≥ 2/3")


# str_problem_to_standard_form("", detailed=True)

# bateria_testes_str_padrao_problema(True, True, True, True, True, True)


# bateria_testes()

# str_problem_to_matrix("")

""" 
NOVO PADRAO: para todas funcoes que devolvem constraints, 
devolver variable_constraints de todas variaveis
Plano de trabalho:
- Criar testes para todos auxiliares novos
- Refazer todos os testes existentes para o padrao mais atual
- Refazer os testes de str_padrao_problema

 """

# π, Φ, λ


#  Dois casos novos para restricoes : variavel irrestrita e variavel < 0
