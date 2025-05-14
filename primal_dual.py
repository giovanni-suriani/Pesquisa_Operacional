from fractions import Fraction
import logging
#from settings import LOGGING, VERBOSE_VAR
import settings
import re
import str_padrao_problema as spp
import sys, os
from sympy import Matrix, pprint, pretty

""" Módulo de transformação de problemas primais em problemas duais e vice-versa."""

#logging.config.dictConfig(settings.LOGGING)
logger = logging.getLogger("top_module.child")  

if not logger.hasHandlers() and __name__ == '__main__':
    logging.config.dictConfig(settings.LOGGING)
    logger = logging.getLogger("top_module")  # __main__
    print(f"sem handler, executando como top_module o arquivo {os.path.basename(__file__)}")

logger.debug("primal_dual.py")

import str_padrao_problema as spp


VERBOSE = settings.VERBOSE

explain = settings.PRECISO_EXPLICAR

#teste_gpt = spp.teste_gpt()


# Variaveis primais 
VARIAVEIS_PRIMAIS = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"]


#RELACAO_PRIMAL_DUAL = {"x1":"π1", "x2":"π2", "x3":"π3", "x4":"π4", "x5":"π5", "x6":"π6", "x7":"π7", "x8":"π8", "x9":"π9", "x10":"π10" }

VARIAVEIS_DUAIS = ["π1", "π2", "π3", "π4", "π5", "π6", "π7", "π8", "π9", "π10"]


def all_variables_positive(f_obj:str, constraints:list, neg_variables:list):
    """ 
        converte todas as variaveis do problema primmal em não-negativas.
        Args:
            f_obj (str): Função objetivo.
            constraints (list): Lista de restrições.
            neg_variables (list): Lista de variaveis negativas.
        Returns:
            f_obj (str): Função objetivo convertida.
            constraints (list): Lista de restrições convertidas.
    """
    non_variable_constraints = spp.extract_non_var_constraints(constraints)
    variable_constraints = spp.extract_ge_le_constraints(constraints, positive_lhs=True)
    new_vars_constraints = []
    new_constraints = []
    
    # Troca o sinal das f_obj e das restricoes que nao tem a ver com variaveis
    for variable in neg_variables:
        f_obj = spp.change_variable_sign_in_f_obj(variable, f_obj)
        spp.change_variable_sign_in_constraints(variable, non_variable_constraints)
            
    for constraint in variable_constraints:
        constants, variables, symbol, value_rhs = spp.extrai_restricao(constraint)
        if variable in variables:
            new_vars_constraints.append(spp.flip_symbol_in_constraint(constraint, flip_lhs=False))
    
    new_constraints = non_variable_constraints + new_vars_constraints
            
    return f_obj, new_constraints

def check_neg_variables(constraints:list):
    """ 
    Verifica se todas as variaveis são não-negativas.
    Args:
        restricoes (list): Lista de restrições.
    Returns:
        bool: True se todas as variaveis forem não-negativas, False caso contrário.
        list: Lista de variaveis negativas.
    """
    variable_constraints = spp.extract_ge_le_constraints(constraints, positive_lhs=True)
    neg_variables = []
    for i, constraint in enumerate(variable_constraints):
        constants, variables, symbol, value_rhs = spp.extrai_restricao(constraint)
        #constants_and_variables = dict(zip(variables, constants))
        #variables_and_constants = dict(zip(constants, variables))
        if spp.check_le_zero(constants, value_rhs, symbol):
            for constant, variable in zip(constants, variables):
                neg_variables.append(variable)
    return neg_variables
        
def str_primal_to_dual(f_obj:str, constraints:list, current_relation:str="primal", 
                       allow_neg_variables:bool=False, infer_var_signs:bool=True, decimal:bool=False):
    """ 
    Converte um problema primal, transformando todas as variaveis em NÃO-NEGATIVAS, exceto irrestritas, para o seu dual.
    A função retorna o tipo da função objetivo do dual, as restrições do dual e as variáveis do dual.
    """
    variaveis_primais = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"]
    variaveis_duais = ["π1", "π2", "π3", "π4", "π5", "π6", "π7", "π8", "π9", "π10"]
    new_relacao = ""
    logger.debug(f"Restricoes do tipo -x1 <= 0 serao convertidas para x1 >= 0 XD")
    
    
    
    # Verify if the variables are non-negative, if so, convert them to non-negative in whole problem
    if not allow_neg_variables:
        neg_variables = check_neg_variables(constraints)
        if neg_variables:
            f_obj, constraints = all_variables_positive(f_obj, constraints, neg_variables)
    
    # Extracting the function type, variables_constraints and non variable constraints
    func_type, _, _, _ = spp.extrai_f_obj(f_obj)
    all_variables = spp.extract_variables_problem(f_obj, constraints)
    variables_constraints = spp.extract_ge_le_constraints(constraints, positive_lhs=True)
    if infer_var_signs:
        variables_without_constraint = []
        if variables_constraints:
           variables_with_constraint = spp.extract_variables_problem(constraints=variables_constraints)
        for variable in all_variables:
            if variable not in variables_with_constraint:
                variables_without_constraint.append(variable)
        extra_variables_constraints =  spp.assemble_variables_constraints(variables_without_constraint, 
                                                                          is_vars_on_standard_form=True)
        constraints += extra_variables_constraints
        variables_constraints += extra_variables_constraints
    
    non_variable_constraints = spp.extract_non_var_constraints(constraints)
    non_variable_constraints_symbols = spp.extract_constraints_signs(non_variable_constraints) #
    
    if current_relation.lower() == "primal":
        new_vars_to_use = variaveis_duais
        new_relacao = "dual"
    elif current_relation.lower() == "dual":
        new_vars_to_use = variaveis_primais
        new_relacao = "primal"
    
    new_func_type = ""
    new_variables = []
    variable_constraints_symbols = []
    new_variables_constraints = []
    new_non_variable_constraints_symbols = []
    
    if func_type.lower() == "min":
        for variable_constraint in variables_constraints:
            _, _, symbol, _ = spp.extrai_restricao(variable_constraint)
            if symbol == "<=" or symbol == "≤":
                new_non_variable_constraints_symbols.append(f">=")
            elif symbol == ">=" or symbol == "≥":
                new_non_variable_constraints_symbols.append(f"<=")
            elif symbol == "irrestrito":
                new_non_variable_constraints_symbols.append(f"=")
            if explain:
                logger.info(f"Explicando: {variable_constraint} vira restricao {new_non_variable_constraints_symbols[-1]}" )
            
        for i, symbol in enumerate(non_variable_constraints_symbols):
            new_variables.append(new_vars_to_use[i])
            if symbol == "<=" or symbol == "≤":
                var_constraint = f"{new_vars_to_use[i]} <= 0"
                new_variables_constraints.append(var_constraint)
            elif symbol == ">=" or symbol == "≥":
                var_constraint = f"{new_vars_to_use[i]} >= 0"
                new_variables_constraints.append(var_constraint)
            elif symbol == "=":
                var_constraint = f"{new_vars_to_use[i]} irrestrito"
                new_variables_constraints.append(var_constraint)
            if explain:
                logger.info(f"Explicando: restricao de {symbol} vira variavel {new_variables_constraints[-1]}" )
        
        new_func_type = "max"
        if explain:
            logger.info(f"Explicando: funcao objetivo '{func_type}' vira '{new_func_type}'")
            
    elif func_type.lower() == "max":
        for i, symbol in enumerate(non_variable_constraints_symbols):
            new_variables.append(new_vars_to_use[i])
            if symbol == "<=" or symbol == "≤":
                var_constraint = f"{new_vars_to_use[i]} >= 0"
                new_variables_constraints.append(var_constraint)
            elif symbol == ">=" or symbol == "≥":
                var_constraint = f"{new_vars_to_use[i]} <= 0"
                new_variables_constraints.append(var_constraint)
            elif symbol == "=":
                var_constraint = f"{new_vars_to_use[i]} irrestrito"
                new_variables_constraints.append(var_constraint)
            if explain:
                logger.info(f"Explicando: restricao de {symbol} vira variavel {new_variables_constraints[-1]}" )
        
        for variable_constraint in variables_constraints:
            _, _, symbol, _ = spp.extrai_restricao(variable_constraint)
            if symbol == "<=" or symbol == "≤":
                new_non_variable_constraints_symbols.append(f"<=")
            elif symbol == ">=" or symbol == "≥":
                new_non_variable_constraints_symbols.append(f">=")
            elif symbol == "irrestrito":
                new_non_variable_constraints_symbols.append(f"=")
            if explain:
                logger.info(f"Explicando: {variable_constraint} vira restricao {new_non_variable_constraints_symbols[-1]}" )
        
        new_func_type = "min"
        if explain:
            logger.info(f"Explicando: funcao objetivo '{func_type}' vira '{new_func_type}'")
    
    # Assembling the dual problem
    A_input, b_input, c_input, x_input = spp.str_problem_to_std_form_matrix(f_obj, constraints)
    c_output = b_input
    A_output = (Matrix(A_input).T).tolist()
    b_output = c_input
    f_obj_output, constraints_output = spp.std_matrix_to_str_problem(A_output, b_output, 
                                                                     c_output, new_variables,
                                  tipo_funcao=new_func_type,
                                  restricoes_simbolos=new_non_variable_constraints_symbols,
                                  decimal=decimal,
                                  )
    constraints_output += new_variables_constraints
    if VERBOSE:
        logger.debug(f"f_obj_output: {f_obj_output}")
        logger.debug(f"constraints_output: {constraints_output}")
    
    
    return f_obj_output, constraints_output, new_relacao

def matrix_primal_to_dual(A, b, c, x, current_relation:str="primal", standard_form:bool=False):
    """ 
    Converte um problema primal ou dual na forma de matriz para o seu dual
    """
    pass
   
def bateria_de_testes_primal_dual(test_primal_to_dual:bool=False,
                                  test_dual_to_primal:bool=False):
    logger.info("Iniciando com os testes de primal dual ...")
    # Testes para a funcao str_primal_to_dual
    if test_primal_to_dual:
        logger.info("Iniciando testes para test_primal_to_dual")
        # Teste 1
        t1= {
            "f_obj": "min 2x1 + 3x2",
            "constraints": ["x1 + x2 <= 4", 
                            "3/2x1 + x2 <= 5", 
                            "x1 >= 0",
                            "x2 >= 0"],
            "current_relation": "primal",
            "allow_neg_variables": False,
            "infer_var_signs": False,
            "decimal": False,
            "result": {
                "f_obj": "max 4π1 + 5π2",
                "constraints": ["π1 + 3/2π2 <= 2", 
                                "π1 + π2 <= 3", 
                                "π1 <= 0", 
                                "π2 <= 0"],
                "current_relation": "dual",
            }
        }
        
        t2 = {
            "f_obj": "min 2x1 + 3x2",
            "constraints": ["x1 + x2 <= 4", 
                            "3/2x1 + x2 <= 5", 
                            "-x1 >= 0"],
            "current_relation": "primal",
            "allow_neg_variables": False,
            "infer_var_signs": True,
            "decimal": False,
            "result": {
                "f_obj": "max 4π1 + 5π2",
                "constraints": ["-π1 - 3/2π2 <= -2", 
                                "π1 + π2 <= 3", 
                                "π1 <= 0", 
                                "π2 <= 0"],
                "current_relation": "dual",
            }
        }
        
        t3 = {
            "f_obj": "min 2x1 + 3x2",
            "constraints": ["x1 + x2 <= 4", 
                            "3/2x1 + x2 >= 5", 
                            "-x1 >= 0",
                            "x2 >= 0"],
            "current_relation": "primal",
            "allow_neg_variables": True,
            "infer_var_signs": False,
            "decimal": False,
            "result": {
                "f_obj": "max 4π1 + 5π2",
                "constraints": ['π1 + 3/2π2 >= 2', 
                                'π1 + π2 <= 3', 
                                'π1 <= 0', 
                                'π2 >= 0'],
                "current_relation": "dual",
            }
        }
        
        t4 = {
            "f_obj": "max 2x1 - 3x2",
            "constraints": ["x1 + x2 <= 4", 
                            "3/2x1 + x2 >= 5", 
                            "-x1 >= 0",
                            "x2 >= 0"],
            "current_relation": "primal",
            "allow_neg_variables": False,
            "infer_var_signs": True,
            "decimal": False,
            "result": {
                "f_obj": "min 4π1 + 5π2",
                "constraints": ['-π1 - 3/2π2 >= -2', 
                                'π1 + π2 >= -3', 
                                'π1 >= 0', 
                                'π2 <= 0'],
                "current_relation": "dual",
            }
        }
        
        t5 = {
            "f_obj": "max 2π1 - π2",
            "constraints": 
                ["π1 + π2 <= 4", 
                "1.5π1 + π2 >= 5", 
                "π1 >= 0",
                "π2 >= 0"],
            "current_relation": "dual",
            "allow_neg_variables": False,
            "infer_var_signs": True,
            "decimal": True,
            "result": {
                "f_obj": "min 4x1 + 5x2",
                "constraints": ['x1 + 1.5x2 >= 2', 
                                'x1 + x2 >= -1', 
                                'x1 >= 0', 
                                'x2 <= 0'],
                "current_relation": "primal",
            }
        }
        
        tests = [t1, t2, t3, t4, t5]
        for i, teste in enumerate(tests):
            f_obj = teste["f_obj"]
            constraints = teste["constraints"]
            current_relation = teste["current_relation"]
            allow_neg_variables = teste["allow_neg_variables"]
            infer_var_signs = teste["infer_var_signs"]
            decimal = teste["decimal"]
            try:
                f_obj_output, constraints_output, relacao = str_primal_to_dual(f_obj, constraints, current_relation, allow_neg_variables, infer_var_signs, decimal)
                assert f_obj_output == teste["result"]["f_obj"]
                assert constraints_output == teste["result"]["constraints"]
                assert relacao == teste["result"]["current_relation"]
            except AssertionError as e:
                logger.error(f"Erro no teste {i+1}")
                logger.error(f"\nvalor calculado: {f_obj_output} \nvalor esperado: {teste['result']['f_obj']}")
                logger.error(f"\nvalor calculado: {constraints_output} \nvalor esperado: {teste['result']['constraints']}")
                logger.error(f"\nvalor calculado: {relacao} \nvalor esperado: {teste['result']['current_relation']}")
                raise e
            #str_primal_to_dual(f_obj, constraints, current_relation, allow_neg_variables, infer_var_signs)
    
    logger.info("Bateria de testes finalizada.")        
   
#bateria_de_testes_primal_dual(test_primal_to_dual=True)

def check_health_status():
    try:
        logger.setLevel(logging.INFO)
        explain = False
        bateria_de_testes_primal_dual(True)
        logger.info("Todos os testes passaram com sucesso!")
    except Exception as e:
        logger.error("Erro nos testes utilitarios ou de primal_dual")
        logger.error(e)
        raise e
    

# Checar caso primal -> dual -> primal  no test_primal_to_dual
# Implementar no str_padrao_problema o caso de variaveis negativas para monta_restricao forma padrao