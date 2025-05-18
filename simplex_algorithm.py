import os
import logging
import settings
from sympy import Matrix, pprint, pretty
from fractions import Fraction
import str_padrao_problema as spp
from collections import defaultdict


explain = settings.PRECISO_EXPLICAR
import os
import logging

VERBOSE = settings.VERBOSE

HALT = 0

CONTINUE = 1

logger = logging.getLogger("top_module.child")  

if not logger.hasHandlers() and __name__ == '__main__':
    logging.config.dictConfig(settings.LOGGING)
    logger = logging.getLogger("top_module")  # __main__
    print(f"sem handler, executando como top_module o arquivo {os.path.basename(__file__)}")

logger.debug(f"nome arquivo = {os.path.basename(__file__)}")    

def str_in_line_matrixes(matrixes:list):
    """
    Converte uma lista de matrizes em uma string formatada.
    Args:
        matrixes (list): Lista de matrizes a serem convertidas.
    Returns:
        str: String formatada com as matrizes.
    """
    result = ""
    for matrix in matrixes:
        result += f"{pretty(matrix)}\n"
    return result

from sympy import pretty, Matrix


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


def side_by_side_matrices_with_labels(matrices: list[Matrix], labels: list[str], sep: str = " │ ") -> str:
    """
    Exibe n matrizes lado a lado com rótulos centralizados.

    Args:
        matrices (list[Matrix]): Lista de objetos Matrix do sympy.
        labels (list[str]): Lista de rótulos, um para cada matriz.
        sep (str): Separador entre as colunas.

    Returns:
        str: Matrizes formatadas lado a lado com rótulos.
    """
    if len(matrices) != len(labels):
        raise ValueError("O número de matrizes deve ser igual ao número de rótulos.")

    # Transforma cada matriz em uma lista de strings (linhas)
    matrix_lines = [pretty(m).splitlines() for m in matrices]
    heights = [len(lines) for lines in matrix_lines]
    widths = [max(len(line) for line in lines) for lines in matrix_lines]

    max_height = max(heights)

    # Preenche as linhas de cada matriz com espaços para igualar alturas
    for i in range(len(matrix_lines)):
        matrix_lines[i] += [" " * widths[i]] * (max_height - heights[i])

    # Rótulos centralizados acima de cada coluna
    label_line = sep.join(label.center(width) for label, width in zip(labels, widths))

    # Junta as linhas horizontalmente
    body_lines = []
    for i in range(max_height):
        line = sep.join(matrix_lines[j][i].ljust(widths[j]) for j in range(len(matrices)))
        body_lines.append(line)

    return "\n".join([label_line] + body_lines)

def get_index_based_on_x(dummy_x:str, x:list):
    """
    Retorna os índices de uma lista de variáveis.
    Args:
        x (list): Lista de variáveis.
        indexes (list): Lista de índices a serem retornados.
    Returns:
        int: indice correspondente de dummy_x
    """
    for i, var in enumerate(x):
        if var[0] == dummy_x:
            return i
        
    raise ValueError(f"Variável {dummy_x} não encontrada na lista de variáveis.")

def concatenate_matrices_vertical(matrices:list[Matrix])->list:
    """
    Concatena as linhas de várias matrizes em uma única resultante
    Args:
        matrices (list): Lista de matrizes a serem concatenadas.
    Returns:
        list: matriz resultante em forma de list.
    """
    n_cols = matrices[0].cols
    for matrix in matrices:
        if matrix.cols != n_cols:
            raise ValueError("Todas as matrizes devem ter o mesmo número de colunas.")
    
    
    
    result_matrix = Matrix.vstack(*matrices)
    
    return result_matrix.tolist()

def concatenate_matrices_horizontal(matrices:list)->list:
    """
    Concatena as colunas de várias matrizes em uma única resultante
    Args:
        matrices (list): Lista de matrizes a serem concatenadas.
    Returns:
        list: matriz resultante.
    """
    n_rows = matrices[0].rows
    for matrix in matrices:
        if matrix.rows != n_rows:
            raise ValueError("Todas as matrizes devem ter o mesmo número de linhas.")
    
    result_matrix = Matrix.hstack(*matrices)
    
    return result_matrix.tolist()

def get_A_column(A:list, index:int=-1, dummy_x:str="", x:list=[]):
    """
    Retorna a coluna de uma matriz A correspondente a uma variável dummy_x ou um indice passado.
    Args:
        A (list): Matriz A.
        index (int): Índice da coluna a ser retornada.
        dummy_x (str): Nome da variável.
        x (list): Lista de variáveis.
    Returns:
        list: Coluna correspondente à variável dummy_x.
    """
    
    if index == -1:
        index = get_index_based_on_x(dummy_x, x)
    
    column = []
    for i in range(len(A)):
       column.append(A[i][index])
        
    return column

def get_A_basic(A:list, xb:list, x:list, indexes:list=None):
    """
    Retorna a matriz A básica correspondente a uma lista de variáveis básicas.
    Args:
        A (list): Matriz A.
        xb (list): Lista de variáveis básicas.
        indexes (list): Lista de índices a serem retornados.
    Returns:
        list: Matriz A básica correspondente às variáveis básicas.
    """
    
    if not indexes:
        indexes = []
        for var in xb:
            indexes.append(get_index_based_on_x(var[0], x))
        #indexes = [get_index_based_on_x(x[0], xb) for x in xb]
    
    #A_basic = []
    """ for column in indexes:
        a_column = get_A_column(A, column)
        print(f"coluna {column} = {a_column}")
        A_basic.append(get_A_column(A, column))
        
    A_basic = concatenate_matrices_horizontal(A_basic) """
    A = Matrix(A)
    # Extração de colunas por indexação de slice
    A_basic = A[:, indexes]
    return A_basic.tolist()

def get_A_non_basic(A:list, xb:list, x:list, indexes:list=None):
    """
    Retorna a matriz A não básica correspondente a uma lista de variáveis não básicas.
    Args:
        A (list): Matriz A.
        xb (list): Lista de variáveis não básicas.
        indexes (list): Lista de índices a serem retornados.
    Returns:
        list: Matriz A não básica correspondente às variáveis não básicas.
    """
    
    if not indexes:
        indexes = []
        for var in xb:
            indexes.append(get_index_based_on_x(var[0], x))
    
    A = Matrix(A)
    all_indexes = list(range(A.cols))
    non_basic_indexes = [i for i in all_indexes if i not in indexes]
      
    A_non_basic = A[:, non_basic_indexes]  
    return A_non_basic.tolist()

def get_c_basic(c:list, xb:list, x:list, indexes:list=None):
    """
    Retorna o vetor c básico correspondente a uma lista de variáveis básicas.
    Args:
        c (list): Vetor c.
        xb (list): Lista de variáveis básicas.
        indexes (list): Lista de índices a serem retornados.
    Returns:
        list: Vetor c básico correspondente às variáveis básicas.
    """
    
    if not indexes:
        indexes = []
        for var in xb:
            indexes.append(get_index_based_on_x(var[0], x))
    
    c = Matrix(c).T # A matriz vira coluna sem o .T
    # Extração de linhas por indexação de slice
    c_basic = c[:, indexes]
    return c_basic.tolist()

def get_c_non_basic(c:list, xb:list, x:list, indexes:list=None):
    """
    Retorna o vetor c não básico correspondente a uma lista de variáveis não básicas.
    Args:
        c (list): Vetor c.
        xb (list): Lista de variáveis não básicas.
        indexes (list): Lista de índices a serem retornados.
    Returns:
        list: Vetor c não básico correspondente às variáveis não básicas.
    """
    
    if not indexes:
        indexes = []
        for var in xb:
            indexes.append(get_index_based_on_x(var[0], x))
    
    c = Matrix(c).T 
    all_indexes = list(range(c.cols))
    non_basic_indexes = [i for i in all_indexes if i not in indexes]
      
    c_non_basic = c[:, non_basic_indexes]  
    return c_non_basic.tolist()

def get_x_non_basic (x:list, xb:list, indexes:list=None):
    """
    Retorna o vetor x não básico correspondente a uma lista de variáveis não básicas.
    Args:
        x (list): Vetor x.
        xb (list): Lista de variáveis não básicas.
        indexes (list): Lista de índices a serem retornados.
    Returns:
        list: Vetor x não básico correspondente às variáveis não básicas.
    """
    
    if not indexes:
        indexes = []
        for var in xb:
            indexes.append(get_index_based_on_x(var[0], x))
    
    x = Matrix(x)
    all_indexes = list(range(x.rows)) # linhas, pois x é coluna
    non_basic_indexes = [i for i in all_indexes if i not in indexes]
      
    x_non_basic = x[non_basic_indexes, :] # Seleciona linhas do non_basic_indexes
    
    x_non_basic = x_non_basic.tolist()
    
    for i in range(len(x_non_basic)):
        x_non_basic[i][0] = str(x_non_basic[i][0])
    
    return x_non_basic
    
def assemble_new_basis_variable(x:list, xb:list, new_basic_variable:str, leaving_variable:str):
    """
    Monta a nova base após a troca de variáveis.
    Args:
        x (list): Vetor x.
        xb (list): Lista de variáveis básicas.
        new_basic_variable (str): Variável que entra na base.
        leaving_variable (str): Variável que sai da base.
    Returns:
        list: Nova base.
    """
    new_basis = []
    for var in xb:
        if var[0] == leaving_variable:
            new_basis.append([new_basic_variable])
        else:
            new_basis.append(var)
    return new_basis

def calculate_basic_solution(A_basic:list, b:list, basic_c:list, method:str="sergio", basic_variables:list=None) -> list:
    """ 
    Calcula a solução básica do problema de programação linear.
    Args:
        A_basic (list): Matriz das variáveis básicas.
        b (list): Vetor de constantes.
        basic_c (list): Constantes associadas às variáveis básicas.
        method (str): Método de resolução do problema cruzeiro ou Sergio
    Returns:
        list: Solução básica.
    """
    mat_A_basic = Matrix(A_basic)
    mat_b = Matrix(b)
    mat_basic_c = Matrix(basic_c)
    mat_basic_variables = Matrix(basic_variables)
    if method == "sergio":
        basic_solution = mat_A_basic.inv() * mat_b
        if explain:
            logger.info(f"calculando solução básica (xᵢ):\n{side_by_side_matrices_with_labels([mat_A_basic.inv(), mat_b], ['A⁻¹', 'b'])}")
            logger.debug(f"\nSolução básica (xᵦ):\n{pretty(basic_solution)}")
    elif method == "cruzeiro":
        basic_solution = mat_A_basic.LUsolve(mat_b)
        if explain:
            logger.info(f"calculando solução básica (xᵢ):\n{side_by_side_matrices_with_labels([mat_A_basic, mat_basic_variables, mat_b], ['B', 'xᵦ', 'b'])}")
            logger.debug(f"\nSolução básica (xᵦ):\n{pretty(basic_solution)}")
    else:
        raise ValueError("Método inválido. Use 'sergio' ou 'cruzeiro'.")
    return basic_solution.tolist()

def calculate_simplex_multiplier_vector(A_basic:list, basic_c:list, method:str="sergio") -> list: 
    mat_A_basic = Matrix(A_basic)
    mat_basic_c = Matrix(basic_c)
    if method == "sergio":
        # Multiplicadores de Sérgio
        simplex_multiplier_vector = mat_basic_c * mat_A_basic.inv()
        if explain:
            explanation_matrix = side_by_side_matrices_with_labels([mat_basic_c, mat_A_basic.inv()], ['cᵢ', 'A⁻¹'])
            logger.info(f"calculando vetor multiplicador simplex (π):\n{explanation_matrix}")
            logger.debug(f"\nMultiplicador simplex (π):\n{pretty(simplex_multiplier_vector)}")
    elif method == "cruzeiro":
        # Multiplicadores do Cruzeiro
        lambda_vector = [["λ" for _ in range(len(A_basic))]]
        mat_lambda = Matrix(lambda_vector).T
        mat_A_basic_t = mat_A_basic.T
        simplex_multiplier_vector = mat_A_basic_t.LUsolve(mat_basic_c.T).T
        if explain:
            explanation_matrix = side_by_side_matrices_with_labels([mat_A_basic_t, mat_lambda, mat_basic_c.T], ['Bᵗ', 'λ','cᵦ'])
            logger.info(f"calculando vetor multiplicador simplex (λ):\n{explanation_matrix}")
            logger.debug(f"\nMultiplicador simplex (λ):\n{pretty(simplex_multiplier_vector.T)}")
    else:
        raise ValueError("Método inválido. Use 'sergio' ou 'cruzeiro'.")
    return simplex_multiplier_vector.tolist()

def calculate_reduced_costs(simplex_multiplier_vector:list, A_non_basic:list, non_basic_c:list, method:str="sergio") -> list:
    """
    Calcula os custos reduzidos para cada variável não básica.
    Args:
       simplex_multiplier_vector (list): Vetor multiplicador simplex.
        A_non_basic (list): Matriz das variáveis não básicas.
        non_basic_c (list): Vetor de custos não básicos.
        method (str): Método de resolução do problema cruzeiro ou Sergio
    Returns:
        list: Custos reduzidos.
    """
    #n_non_basic_vars = len(A_non_basic[0])
    mat_simplex_multiplier = Matrix(simplex_multiplier_vector)
    mat_A_non_basic = Matrix(A_non_basic)
    mat_non_basic_c = Matrix(non_basic_c)
    if method == "sergio":
        reduced_costs = mat_simplex_multiplier * mat_A_non_basic - mat_non_basic_c
        if explain:
            logger.info(f"calculando custos reduzidos (πAⱼ - cⱼ):\n{side_by_side_matrices_with_labels([mat_simplex_multiplier, mat_A_non_basic, mat_non_basic_c], ['π', 'Aⱼ', 'cⱼ'])}")
            logger.debug(f"\nCustos reduzidos (πAⱼ - cⱼ):\n{pretty(reduced_costs)}")
    elif method == "cruzeiro":
        reduced_costs = mat_non_basic_c - mat_simplex_multiplier * mat_A_non_basic
        if explain:
            logger.info(f"calculando custos relativos (ĉₙ):\n{side_by_side_matrices_with_labels([mat_non_basic_c.T, mat_simplex_multiplier, mat_A_non_basic], ['cₙ', 'λᵗ', 'aₙ'])}")
            logger.debug(f"\nCustos relativos (cₙ - λᵗaₙ):\n{pretty(reduced_costs.T)}")
    else:
        raise ValueError("Método inválido. Use 'sergio' ou 'cruzeiro'.")
    return reduced_costs.tolist()

def calculate_new_basis_variable_index(reduced_costs:list, type_func:str="min", method:str="sergio",
                                 start_index_from_zero:bool=False) -> int:
    """
    Calcula o indice a ser introduzida na base.
    Args:
        reduced_costs (list): Custos reduzidos.
        type_func (str): Tipo de função objetivo ('max' ou 'min').
        method (str): Método de resolução do problema cruzeiro ou Sergio
    Returns:
        int: Índice da nova variável básica., -1 => não há nova variável
    """
    
    # Verifica se há custos reduzidos negativos
    new_basic_index = -1
    reduced_costs = reduced_costs[0]
    if type_func == "min":
        if method == "sergio":
            if not all(cost <= 0 for cost in reduced_costs):
                logger.debug(f"buscando o maior custo reduzido, metodo do sergio minimizacao")
                new_basic_index  = reduced_costs.index(max(reduced_costs))
            else:
                if explain:
                    logger.info("Todos os custos relativos são ≤ 0. Solucao ótima encontrada.")
        elif method == "cruzeiro":
            if not all(cost >= 0 for cost in reduced_costs):
                logger.debug(f"buscando o menor custo reduzido, metodo do cruzeiro minimizacao")
                new_basic_index = reduced_costs.index(min(reduced_costs))
            else:
                if explain:
                    logger.info("Todos os custos relativos são ≥ 0. Solucao ótima encontrada.")
        else:
            raise ValueError("Método inválido. Use 'sergio' ou 'cruzeiro'.")
        
    elif type_func == "max":
        if method == "sergio":
            if not all(cost >= 0 for cost in reduced_costs):
                logger.debug(f"buscando o menor custo reduzido, no metodo do sergio maximizacao")
                new_basic_index = reduced_costs.index(min(reduced_costs))
            else:
                if explain:
                    logger.info("Todos os custos relativos são ≥ 0. Solucao ótima encontrada.")
        elif method == "cruzeiro":
            if not all(cost <= 0 for cost in reduced_costs):
                logger.debug(f"buscando o maior custo reduzido, no metodo do cruzeiro maximizacao")
                new_basic_index = reduced_costs.index(max(reduced_costs))
            else:
                if explain:
                    logger.info("Todos os custos relativos são ≤ 0. Solucao ótima encontrada.")
        else:
            raise ValueError("Método inválido. Use 'sergio' ou 'cruzeiro'.")
    else:
        raise ValueError("Tipo de função inválido. Use 'max' ou 'min'.")
    
    if new_basic_index == -1:
        return -1
    elif start_index_from_zero:
        if explain:
            logger.info(f"índice k: {new_basic_index}")
        return new_basic_index
    else:
        # Adiciona 1 ao índice para corresponder à contagem a partir de 1
        new_basic_index += 1
        if explain:
            logger.info(f"índice k: {new_basic_index}")
        
    return new_basic_index

def calculate_search_direction_vector(A_basic:list, A_non_basic:list, k_index:int, method:str="sergio")-> list:
    """
    Calcula o vetor de direção de busca.
    Args:
        A_basic (list): Matriz das variáveis básicas.
        k_index (int): Índice da variável básica a ser removida.
    Returns:
        list: Vetor de direcao de busca
    """
    mat_A_basic = Matrix(A_basic)
    mat_A_non_basic = Matrix(A_non_basic)
    mat_A_k = mat_A_non_basic.col(k_index)
    
    if method == "sergio":
        # Multiplicadores de Sérgio
        search_direction_vector = mat_A_basic.inv() * mat_A_k
        if explain:
            logger.info(f"calculando vetor de direção de busca (yₖ):\n{side_by_side_matrices_with_labels([mat_A_basic.inv(), mat_A_k], ['A⁻¹', 'Aₖ'])}")
            logger.debug(f"\nVetor de direção de busca (yₖ):\n{pretty(search_direction_vector)}")
    
    elif method == "cruzeiro":
        # Multiplicadores do Cruzeiro
        y_vector = [["y" for _ in range(len(A_basic))]]
        y_vector = Matrix(y_vector).T
        search_direction_vector = mat_A_basic.LUsolve(mat_A_k)
        if explain:
            logger.info(f"calculando vetor de direção de busca (yₖ):\n{side_by_side_matrices_with_labels([mat_A_basic,y_vector ,mat_A_k], ['B','y','aₖ'])}")
            logger.debug(f"\nVetor de direção de busca (yₖ):\n{pretty(search_direction_vector)}")
    
    else:
        raise ValueError("Método inválido. Use 'sergio' ou 'cruzeiro'.")
    return search_direction_vector.tolist()
    
    
    #return search_direction_vector.tolist()

def calculate_leaving_basis_variable_index(search_direction_vector:list, 
                                     basic_solution:list, 
                                     method:str="sergio",
                                     start_from_zero:bool=False) -> int:
    """
    Calcula o indice da variável básica que sairá da base.
    Args:
        search_direction_vector (list): Vetor de direção de busca.
        basic_solution (list): Solução básica.
        method (str): Método de resolução do problema cruzeiro ou Sergio
    Returns:
        int: Índice da variável básica que sairá da base, -1 se não houver variável a sair.
    """
    leaving_variable_index = -1
    r = {}
    for i, (yi, xb) in enumerate(zip(search_direction_vector, basic_solution)):
        if yi[0] > 0:
            r[i] = Fraction(xb[0],yi[0]) #xb[0] / yi[0]
        else:
            r[i] = float("inf")
    
    if all(value == float("inf") for value in r.values()):
        if explain:
            logger.info("Todas variaveis no vetor direcao são ≤ 0. Solução pode crescer indefinidamente.")
        return -1
    
    leaving_variable_index = min(r, key=r.get)
    if not start_from_zero:
        leaving_variable_index += 1
        
        
    if explain:
        r_list = list(r.values())
        r_list_str = [str(r.numerator) + "/" + str(r.denominator) if r != float("inf") else "∞" for r in r_list]
        if method == "sergio":
            logger.info(f"conjunto r {{{r_list_str}}}")
            logger.info(f"menor valor = {min(r_list)}, índice que sai: {leaving_variable_index}")
        elif method == "cruzeiro":
            logger.info(f"conjunto ε {{{r_list_str}}}")
            logger.info(f"menor valor = {min(r_list)}, índice que sai: {leaving_variable_index}")
        
    return leaving_variable_index

def get_trivial_basis(A:list, b:list, x:list):
    """
    Verifica se existe solução trivial para o problema de programação linear.
    Args:
        A (list): Matriz dos coeficientes.
        b (list): Vetor de constantes.
        c (list): Vetor de custos.
    Returns:
        bool: True se existe solução trivial, False caso contrário.
    """
    # Exemplo de solucao trivial
    # A = [[1, 0, 1], [0, 1, 1]]
    # b = [[1], [1]]
    # c = [[1], [1]]
    # xb = [x2, x3]
    n_cols = len(A[0])
    n_rows = len(A)
    xb = defaultdict(list)
    #xb_indexes = []
    
    for i in range(0, n_rows):
        for j in range(n_cols):
            if (A[i][j] == 1 and b[i][0] >= 0) or (A[i][j] == -1 and b[i][0] <= 0):
                aux_x = x[j][0]
                xb[i].append(aux_x) # xb[linha] = variavel
     
    for variavel_linha0 in xb[0]:
        solucao = [variavel_linha0]
        for i in range(1, n_rows):
            for variavel in xb[i]:
                if variavel not in solucao:
                    solucao.append(variavel)
                    break
        if len(solucao) == n_rows:
            logger.debug(f"Foi encontrada solucao trivial, {solucao}")
            return solucao
        
    logger.debug(f"Nao foi encontrada solucao trivial, {dict(xb)}")
    return [] # Nenhuma solucao trivial encontrada

def find_initial_basis(A:list, b:list, c:list, x:list):
    trivial_basis = get_trivial_basis(A, b, x)
    if trivial_basis:
        # Se existe solução trivial, retorna a solução trivial
        return trivial_basis
    else:
        # Caso contrário, verifica se existe uma solução básica viável
        # Se não existir, retorna None ou levanta uma exceção
        # Aqui você pode implementar a lógica para encontrar uma solução básica viável
        pass
    pass

def simplex_sergio_iteration(A:list, b:list, c:list, x:list, basis:list, type_func:str="min"):
    """ 
    Realiza uma iteracao completa do simplex sergio.
    Args:
        A (list): Matriz A.
        b (list): Vetor b.
        c (list): Vetor c.
        x (list): Vetor x.
        basis (list): Lista de variáveis básicas.
        type_func (str): Tipo de função objetivo ('max' ou 'min').
        it (int): Iteração atual.
    Returns:
        tuple: Nova base, solução básica e parada.
    """
    A_basic = get_A_basic(A, basis, x)
    A_non_basic = get_A_non_basic(A, basis, x)
    c_basic = get_c_basic(c, basis, x)
    c_non_basic = get_c_non_basic(c, basis, x)
    x_non_basic = get_x_non_basic(x, basis)
    
    list_matrixes = [
        Matrix(A_basic),
        Matrix(A_non_basic),
        Matrix(c_basic).T,
        Matrix(c_non_basic).T]
    
    #logger.debug(f"\n{side_by_side_matrices_with_labels(list_matrixes, ['Ab', 'An', 'cᵦ', 'cₙ'])}")
    
    if explain:
        logger.info(f"Base = {basis}, nao-base = {x_non_basic}")
        logger.info(f"Fase 1:\n {side_by_side_matrices_with_labels(list_matrixes, ['Ab', 'An', 'cᵦ', 'cₙ'])}")
    
    if explain:
        logger.info(f"Fase2 Passo1 (calculo da solucao basica):")
    basic_solution = calculate_basic_solution(A_basic, b, c, "sergio", basis)
    
    if explain:
        logger.info(f"Passo2.1 Vetor Multiplicador simplex:")
    simplex_multiplier_vector = calculate_simplex_multiplier_vector(A_basic, c_basic, "sergio")

    if explain:
        logger.info(f"Passo2.2 custos relativos:")
    reduced_costs = calculate_reduced_costs(simplex_multiplier_vector, A_non_basic, c_non_basic, "sergio")

    if explain:
        logger.info(f"Passo2.3 e 3 determinacao variavel a entrar na base:")
    
    new_basis_index = calculate_new_basis_variable_index(reduced_costs, type_func, "sergio")
    
    # Otimo pela falta de variaveis a entrar na base
    if new_basis_index == -1:
        return basis, basic_solution, HALT
    
    new_basis_variable = x_non_basic[new_basis_index-1][0]
    logger.debug(f"variavel a entrar na base: {new_basis_variable}, devido ao custo relativo {reduced_costs[0][new_basis_index-1]}")
    
    if explain:
        logger.info(f"Passo4 determinacao variavel a sair da base:")
    search_direction_vector = calculate_search_direction_vector(A_basic, A_non_basic, new_basis_index-1, "sergio")
    leaving_basis_variable_index = calculate_leaving_basis_variable_index(search_direction_vector, basic_solution, "sergio")
    
    # Solucao pode crescer indefinidamente pelas condicoes do vetor de direcao de busca
    if leaving_basis_variable_index == -1:
        return basis, basic_solution, HALT
    
    leaving_basis_variable = basis[leaving_basis_variable_index-1][0]
    logger.debug(f"variavel a sair da base: {leaving_basis_variable}, devido ao vetor de direcao de busca {search_direction_vector[leaving_basis_variable_index-1][0]}")
    
    new_basis_index = get_index_based_on_x(leaving_basis_variable, x)
    new_basis = assemble_new_basis_variable(x, basis, new_basis_variable, leaving_basis_variable)
    return new_basis, basic_solution, CONTINUE

def simplex_cruzeiro_iteration(A:list, b:list, c:list, x:list, basis:list, type_func:str="min", it:int=1):
    """ 
    Realiza uma iteracao completa do simplex cruzeiro.
    Args:
        A (list): Matriz A.
        b (list): Vetor b.
        c (list): Vetor c.
        x (list): Vetor x.
        basis (list): Lista de variáveis básicas.
        type_func (str): Tipo de função objetivo ('max' ou 'min').
        it (int): Iteração atual.
    Returns:
        tuple: Nova base, solução básica e parada.
    """
    
    A_basic = get_A_basic(A, basis, x)
    A_non_basic = get_A_non_basic(A, basis, x)
    c_basic = get_c_basic(c, basis, x)
    c_non_basic = get_c_non_basic(c, basis, x)
    x_non_basic = get_x_non_basic(x, basis)
    
    list_matrixes = [
        Matrix(A_basic),
        Matrix(A_non_basic),
        Matrix(c_basic).T,
        Matrix(c_non_basic).T]
    
    logger.debug(f"\n{side_by_side_matrices_with_labels(list_matrixes, ['Ab', 'An', 'cᵦ', 'cₙ'])}")
    
    if explain:
        logger.info(f"Base = {basis}, nao-base = {x_non_basic}")
        logger.info(f"Fase 1:\n {side_by_side_matrices_with_labels(list_matrixes, ['Ab', 'An', 'cᵦ', 'cₙ'])}")
    
    if explain:
        logger.info(f"Fase2 Passo1 (calculo da solucao basica):")
    basic_solution = calculate_basic_solution(A_basic, b, c, "cruzeiro", basis)
    
    if explain:
        logger.info(f"Passo2.1 Vetor Multiplicador simplex:")
    simplex_multiplier_vector = calculate_simplex_multiplier_vector(A_basic, c_basic, "cruzeiro")

    if explain:
        logger.info(f"Passo2.2 custos relativos:")
    reduced_costs = calculate_reduced_costs(simplex_multiplier_vector, A_non_basic, c_non_basic, "cruzeiro")

    if explain:
        logger.info(f"Passo2.3 e 3 determinacao variavel a entrar na base:")
    
    new_basis_index = calculate_new_basis_variable_index(reduced_costs, type_func, "cruzeiro")
    
    # Otimo pela falta de variaveis a entrar na base
    if new_basis_index == -1:
        return basis, basic_solution, HALT
    
    new_basis_variable = x_non_basic[new_basis_index-1][0]
    logger.debug(f"variavel a entrar na base: {new_basis_variable}, devido ao custo relativo {reduced_costs[0][new_basis_index-1]}")
    
    if explain:
        logger.info(f"Passo4 determinacao variavel a sair da base:")
    search_direction_vector = calculate_search_direction_vector(A_basic, A_non_basic, new_basis_index-1, "cruzeiro")
    leaving_basis_variable_index = calculate_leaving_basis_variable_index(search_direction_vector, basic_solution, "cruzeiro")
    
    # Solucao pode crescer indefinidamente pelas condicoes do vetor de direcao de busca
    if leaving_basis_variable_index == -1:
        return basis, basic_solution, HALT
    
    leaving_basis_variable = basis[leaving_basis_variable_index-1][0]
    logger.debug(f"variavel a sair da base: {leaving_basis_variable}, devido ao vetor de direcao de busca {search_direction_vector[leaving_basis_variable_index-1][0]}")
    
    new_basis = assemble_new_basis_variable(x, basis, new_basis_variable, leaving_basis_variable)
    return new_basis, basic_solution, CONTINUE

def solve_simplex_sergio(A:list, b:list, c:list, x:list, initial_basis:list, type_func:str="min"):
    if not initial_basis:
        initial_basis = find_initial_basis(A, b, c, x)
        if not initial_basis:
            raise ValueError("Não foi possível encontrar uma base inicial viável.")
    basis = initial_basis
    iteration = 1
    tested_basis = [initial_basis]
    
    while True:
        if not basis or iteration >= 5:
            break
        logger.info(f"///////////////iteracao {iteration} comecando/////////////////")
        basis, basic_sol, parada = simplex_sergio_iteration(A, b, c, x, basis, type_func)
        if parada == HALT:
            logger.info(f"Fim do algoritmo simplex")
            break
        if basis in tested_basis:
            logger.info(f"LOOP detectado, variaveis basicas iguais, sistema sem solucao otima")
            break
        tested_basis.append(basis)
        iteration += 1
    logger.info(f"base final: {basis}, solucao = {basic_sol}, iteracao {iteration}")

def solve_simplex_cruzeiro(A:list, b:list, c:list, x:list, initial_basis:list=[], type_func:str="min"):
    if not initial_basis:
        initial_basis = find_initial_basis(A, b, c, x)
        if not initial_basis:
            raise ValueError("Não foi possível encontrar uma base inicial viável.")
    basis = initial_basis
    iteration = 1
    tested_basis = [initial_basis]
    
    while True:
        if not basis or iteration >= 6:
            break
        logger.debug(f"///////////////iteracao {iteration} comecando/////////////////")
        basis, basic_sol, parada = simplex_cruzeiro_iteration(A, b, c, x, basis, type_func, iteration)
        if parada == HALT:
            logger.debug(f"Fim do algoritmo simplex")
            break
        if basis in tested_basis:
            logger.debug(f"LOOP detectado, variaveis basicas iguais, sistema sem solucao otima")
            break
        tested_basis.append(basis)
        iteration += 1
    logger.debug(f"base final: {basis}, solucao = {basic_sol}, iteracao {iteration}")
 
def bateria_de_testes_solve_simplex(test_calculate_reduced_costs:bool=False, 
                                           test_calculate_simplex_multiplier_vector:bool=False,
                                           test_calculate_basic_solution:bool=False,
                                           test_calculate_search_direction_vector:bool=False,
                                           test_calculate_new_basis_variable_index:bool=False,
                                           test_calculate_leaving_basis_variable:bool=False,
                                           test_get_trivial_basis:bool=False,
                                           test_get_A_column:bool=False,
                                           test_get_A_basic:bool=False,
                                           test_get_A_non_basic:bool=False,
                                           test_get_c_basic:bool=False,
                                           test_get_c_non_basic:bool=False,
                                           test_get_x_non_basic:bool=False,
                                           test_simplex_sergio_iteration:bool=False,
                                           test_simplex_cruzeiro_iteration:bool=False,
                                           test_solve_simplex_sergio:bool=False,
                                           test_solve_simplex_cruzeiro:bool=False):



    # Testes para test_calculate_reduced_costs
    if test_calculate_reduced_costs:
        logging.info("Testando calcular custos reduzidos")
        t1 = {
        "simplex_multiplier_vector": [[Fraction(-1, 2), 0, 0]],
        "A_non_basic": [[1, 1], [0, 2], [0, -1]],
        "non_basic_c": [[0, -1]],
        "method": "cruzeiro",
        "result": [[Fraction(-1, 2), Fraction(1, 2)]]
        #"simplex_multiplier_vector":  calculate_simplex_multiplier_vector(A_basic, basic_c, "cruzeiro")
        }

        t2 = {
        "simplex_multiplier_vector": [[Fraction(-1, 2), 0, 0]],
        "A_non_basic": [[1, 1], [0, 2], [0, -1]],
        "non_basic_c": [[0, -1]],
        "method": "sergio",
        "result": [[Fraction(-1, 2), Fraction(1, 2)]]
        #"simplex_multiplier_vector":  calculate_simplex_multiplier_vector(A_basic, basic_c, "cruzeiro")
        }

        tests = [t1, t2]
        for i, test in enumerate(tests):
            simplex_multiplier_vector = test["simplex_multiplier_vector"]
            A_non_basic = test["A_non_basic"]
            non_basic_c = test["non_basic_c"]
            method = test["method"]
            result = test["result"]
            calculated = calculate_reduced_costs(simplex_multiplier_vector, A_non_basic, non_basic_c, method)
            try:
                assert calculated == result
            except AssertionError as e:
                logger.error(f'Erro no teste {i + 1}')
                logger.error(f"valor calculado: {calculated}\nvalor  esperado: {result}")
                raise e

    # Teste para test_calculate_simplex_multiplier_vector
    if test_calculate_simplex_multiplier_vector:
        logging.info("Testando calcular custos reduzidos")
        t1 = {
            "A_basic": [[2, 0, 0], [1, 1, 0], [0, 0, 1]],
            "b": [[8], [7], [3]],
            "basic_c": [[-1, 0, 0]],
            "method": "sergio",
            "result": [[Fraction(-1, 2), 0, 0]]
        }

        t2 = {
            "A_basic": [[2, 0, 0], [1, 1, 0], [0, 0, 1]],
            "b": [[8], [7], [3]],
            "basic_c": [[-1, 0, 0]],
            "method": "cruzeiro",
            "result": [[Fraction(-1, 2), 0, 0]]
        }

        tests = [t1, t2]

        for i, test in enumerate(tests):
            A_basic = test["A_basic"]
            b = test["b"]
            basic_c = test["basic_c"]
            method = test["method"]
            result = test["result"]
            calculated = calculate_simplex_multiplier_vector(A_basic, basic_c, method)
            try:
                assert calculated == result
            except AssertionError as e:
                logger.error(f'Erro no teste {i + 1}')
                logger.error(f"valor calculado: {calculated}\nvalor  esperado: {result}")
                raise e

        # Teste para test_calculate_basic_solution

    # Teste para test_calculate_basic_solution
    if test_calculate_basic_solution:
        logging.info("Testando calcular solução básica")
        t1 = {
            "A_basic": [[2, 0, 0], [1, 1, 0], [0, 0, 1]],
            "b": [8, 7, 3],
            "basic_c": [[-1, 0, 0]],
            "method": "sergio",
            "basic_variables": [],
            "result": [[4], [3], [3]]
        }
        t2 = {
            "A_basic": [[2, 0, 0], [1, 1, 0], [0, 0, 1]],
            "b": [8, 7, 3],
            "basic_c": [[-1, 0, 0]],
            "method": "cruzeiro",
            "basic_variables": [["x1"], ["x2"], ["x3"]],
            "result": [[4], [3], [3]]
        }
        tests = [t1, t2]
        for i, test in enumerate(tests):
            A_basic = test["A_basic"]
            b = test["b"]
            basic_c = test["basic_c"]
            method = test["method"]
            result = test["result"]
            basic_variables = test["basic_variables"]
            calculated = calculate_basic_solution(A_basic, b, basic_c, method, basic_variables)
            try:
                assert calculated == result
            except AssertionError as e:
                logger.error(f'Erro no teste {i + 1}')
                logger.error(f"valor calculado: {calculated}\nvalor  esperado: {result}")
                raise e

    # Teste para test_calculate_search_direction_vector
    if test_calculate_search_direction_vector:
        logging.info("Testando calcular vetor de direção de busca")
        t1 = {
            "A_basic": [[2, 0, 0], [1, 1, 0], [0, 0, 1]],
            "k_index": 1,
            "method": "sergio",
            "result": [[0],[1],[0]]
        }
        t2 = {
            "A_basic": [[2, 0, 0], [1, 1, 0], [0, 0, 1]],
            "k_index": 1,
            "method": "cruzeiro",
            "result": [[0],[1],[0]]
        }
        tests = [t1,t2]
        for i, test in enumerate(tests):
            A_basic = test["A_basic"]
            k_index = test["k_index"]
            method = test["method"]
            result = test["result"]
            calculated = calculate_search_direction_vector(A_basic, k_index, method)
            try:
                assert calculated == result
            except AssertionError as e:
                logger.error(f'Erro no teste {i + 1}')
                logger.error(f"valor calculado: {calculated}\nvalor  esperado: {result}")
                raise e

    # Teste para test_calculate_new_basis_variable_index
    if test_calculate_new_basis_variable_index:
        logging.info("Testando calcular nova variável básica")
        t1 = {
            "reduced_costs": [[Fraction(-1, 2), Fraction(1, 2)]],
            "type_func": "min",
            "method": "sergio",
            "start_index_from_zero": False,
            "result": 2
        }

        t2 = {
            "reduced_costs": [[Fraction(-1, 2), Fraction(1, 2)]],
            "type_func": "min",
            "method": "cruzeiro",
            "start_index_from_zero": False,
            "result": 1
        }
        tests = [t1, t2]
        for i, test in enumerate(tests):
            reduced_costs = test["reduced_costs"]
            type_func = test["type_func"]
            method = test["method"]
            result = test["result"]
            start_index_from_zero = test["start_index_from_zero"]
            calculated = calculate_new_basis_variable_index(reduced_costs, type_func, method, start_index_from_zero)
            try:
                assert calculated == result
            except AssertionError as e:
                logger.error(f'Erro no teste {i + 1}')
                logger.error(f"valor calculado: {calculated}\nvalor  esperado: {result}")
                raise e

    # Teste para test_calculate_leaving_basis_variable
    if test_calculate_leaving_basis_variable:
        logging.info("Testando calcular variável básica que sairá da base")
        t1 = {
            "search_direction_vector": [[0], [1], [0]],
            "basic_solution": [[4], [3], [3]],
            "method": "sergio",
            "start_from_zero": False,
            "result": 1
        }

        t2 = {
            "search_direction_vector": [[0], [1], [0]],
            "basic_solution": [[4], [3], [3]],
            "method": "cruzeiro",
            "start_from_zero": False,
            "result": 1
        }
        t3 = {
            "search_direction_vector": [[0], [1], [0]],
            "basic_solution": [[4], [3], [3]],
            "method": "sergio",
            "start_from_zero": True,
            "result": 2
        }

        t4 = {
            "search_direction_vector": [[0], [1], [0]],
            "basic_solution": [[4], [3], [3]],
            "method": "cruzeiro",
            "start_from_zero": True,
            "result": 2
        }

        tests = [t1, t2, t3, t4]
        for i, test in enumerate(tests):
            search_direction_vector = test["search_direction_vector"]
            basic_solution = test["basic_solution"]
            method = test["method"]
            result = test["result"]
            start_from_zero = test["start_from_zero"]
            calculated = calculate_leaving_basis_variable_index(search_direction_vector, 
                                                                basic_solution, method, start_from_zero)
            try:
                assert calculated == result
            except AssertionError as e:
                logger.error(f'Erro no teste {i + 1}')
                logger.error(f"valor calculado: {calculated}\nvalor  esperado: {result}")
                raise e

    # Teste para test_get_trivial_basis
    if test_get_trivial_basis:
        logging.info("Testando calcular solução trivial")
        t1 = {
        "A": [[1, 0, 1], 
              [0, 1, 1]],
        "b": [[1], [1]],
        "x": [["x1"], ["x2"], ["x3"]],
        "expected_result": ["x1", "x2"]
        }

        t2 = {
        "A": [[2, 0, 1], [0, 1, 1]],
        "b": [[1], [1]],
        "x": [["x1"], ["x2"], ["x3"]],
        "expected_result": ["x3", "x2"]
        }

        # Test case 2: No trivial solution exists
        t3 = {
            "A": [[1, 0, 1], [0, 1, 1]],
            "b": [[-1], [1]],
            "x": [["x1"], ["x2"], ["x3"]],
            "expected_result": []
        }

        tests = [t1, t2, t3]
        for i, test in enumerate(tests):
            A = test["A"]
            b = test["b"]
            x = test["x"]
            expected_result = test["expected_result"]
            calculated_result = get_trivial_basis(A, b, x)
            try:
                assert calculated_result == expected_result
            except AssertionError as e:
                logger.error(f'Erro no teste {i + 1}')
                logger.error(f"\nvalor calculado: {calculated_result}\nvalor  esperado: {expected_result}")
                raise e

        """ for i, test in enumerate(tests):
            A = test["A"]
            b = test["b"]
            x = test["x"]
            expected_result = test["expected_result"]
            
            calculated_result = get_trivial_basis(A, b, x)
            
            try:
                assert calculated_result == expected_result
                logging.info(f"Teste {i + 1} passou com sucesso!")
            except AssertionError as e:
                logger.error(f"Erro no teste {i + 1}")
                logger.error(f"Resultado calculado: {calculated_result}\nResultado esperado: {expected_result}")
                raise e """

    # Teste para test_get_A_column
    if test_get_A_column:
        logging.info("Testando get_A_column")
        t1 = {
            "A": [[1, 2, 3], [4, 5, 6]],
            "index": 1,
            "dummy_x": "",
            "x": [],
            "result": [2, 5],
        }
        t2 = {
            "A": [[7, 8], [9, 10]],
            "index": 0,
            "dummy_x": "x2",
            "x": [["x1"], ["x2"]],
            "result": [7, 9],
        }
        tests = [t1, t2]
        for i, test in enumerate(tests, start=1):
            calc = get_A_column(test["A"], test["index"], test["dummy_x"], test["x"])
            try:
                assert calc == test["result"]
            except AssertionError:
                logger.error(f"Erro no get_A_column teste {i}")
                logger.error(f"valor calculado: {calc}\nvalor esperado: {test['result']}")
                raise

    # Teste para test_get_A_basic
    if test_get_A_basic:
        logging.info("Testando get_A_basic")
        # exemplo: pega colunas x1 (índice 0) e x3 (índice 2)
        A = [[1, 0, 2],
            [0, 1, 3]]
        x = [["x1"], ["x2"], ["x3"]]
        xb = [["x1"], ["x3"]]
        expected = [[1, 2],
                    [0, 3]]
        calc = get_A_basic(A, xb, x)
        try:
            assert calc == expected
        except AssertionError:
            logger.error("Erro no get_A_basic")
            logger.error(f"valor calculado: {calc}\nvalor esperado: {expected}")
            raise

    # Teste para test_get_A_non_basic
    if test_get_A_non_basic:
        logging.info("Testando get_A_non_basic")
        # exemplo: A com 4 colunas, básicas x2,x4 => não-básicas x1,x3
        A = [[1, 2, 3, 4],
            [5, 6, 7, 8]]
        x = [["x1"], ["x2"], ["x3"], ["x4"]]
        xb = [["x2"], ["x4"]]
        expected = [[1, 3],
                    [5, 7]]
        calc = get_A_non_basic(A, xb, x)
        try:
            assert calc == expected
        except AssertionError:
            logger.error("Erro no get_A_non_basic")
            logger.error(f"valor calculado: {calc}\nvalor esperado: {expected}")
            raise

    # Teste para test_get_c_basic
    if test_get_c_basic:
        logging.info("Testando get_c_basic")
        # exemplo: c = [10,20,30], básicas x2,x3 => c_basic = [20,30]
        c = [10, 20, 30]
        x = [["x1"], ["x2"], ["x3"]]
        xb = [["x2"], ["x3"]]
        expected = [[20, 30]]
        calc = get_c_basic(c, xb, x)
        try:
            assert calc == expected
        except AssertionError:
            logger.error("Erro no get_c_basic")
            logger.error(f"valor calculado: {calc}\nvalor esperado: {expected}")
            raise

    # Teste para test_get_c_non_basic
    if test_get_c_non_basic:
        logging.info("Testando get_c_non_basic")
        # mesmas variáveis: não-básicas = x1 => c_non_basic = [10]
        c = [10, 20, 30]
        x = [["x1"], ["x2"], ["x3"]]
        xb = [["x2"], ["x3"]]
        expected = [[10]]
        calc = get_c_non_basic(c, xb, x)
        try:
            assert calc == expected
        except AssertionError:
            logger.error("Erro no get_c_non_basic")
            logger.error(f"valor calculado: {calc}\nvalor esperado: {expected}")
            raise

    # Teste para test_get_x_non_basic
    if test_get_x_non_basic:
        logging.info("Testando get_x_non_basic")
        # exemplo: x = [x1,x2,x3,x4], básicas x2,x4 => não-básicas [x1,x3]
        x = [["x1"], ["x2"], ["x3"], ["x4"]]
        xb = [["x2"], ["x4"]]
        expected = [["x1"], ["x3"]]
        calc = get_x_non_basic(x, xb)
        try:
            assert calc == expected
        except AssertionError:
            logger.error("Erro no get_x_non_basic")
            logger.error(f"valor calculado: {calc}\nvalor esperado: {expected}")
            raise

# Exemplo de chamada para o simplex do cruzeiro

""" A = [[1,  1, 1, 0, 0],
    [1, -1, 0, 1, 0],
    [-1, 1, 0, 0, 1]]
b = [[6], [4], [4]]
c = [-1, -2, 0, 0, 0]
I0 = [["s1"], ["s2"], ["s3"]]
x = [["x1"], ["x2"], ["s1"], ["s2"], ["s3"]]

 """

""" A = [[1, -1, 1, 0],
     [-1, 1, 0, 1]]
b = [[4], [4]]
c = [-1, -1, 0, 0]
I0 = [["s1"], ["s2"]]
x = [["x1"], ["x2"], ["s1"], ["s2"]] """





A = [[2,  1, 1, 0, 0],
    [1, 2, 0, 1, 0],
    [0, 1, 0, 0, 1]]
b = [[8], [7], [3]]
c = [-1, -1, 0, 0, 0]
I0 = [["x3"], ["x4"], ["x5"]]
x = [["x1"], ["x2"], ["x3"], ["x4"], ["x5"]] 


#solve_simplex_cruzeiro(A, b, c, x, I0, type_func="min")

#print(get_x_non_basic(x=x, xb=I0))



#solve_simplex_sergio(A, b, c, x, I0, type_func="min")

# bateria_de_testes_solve_simplex(test_get_A_column=True, 
#                                 test_get_c_basic=True, 
#                                 test_get_c_non_basic=True, 
#                                 test_get_A_basic=True, 
#                                 test_get_A_non_basic=True, 
#                                 test_get_x_non_basic=True)

# bateria_de_testes_solve_simplex(test_)

# bateria_de_testes_solve_simplex(test_calculate_reduced_costs=True,)

# bateria_de_testes_solve_simplex(test_calculate_basic_solution=True)

# bateria_de_testes_solve_simplex(test_calculate_new_basis_variable_index=True,)

# bateria_de_testes_solve_simplex(test_calculate_leaving_basis_variable=True)

def check_health_status():
    logger.info("Iniciando com os testes de simplex_algorithm ...")
    try:
        bateria_de_testes_solve_simplex(True, True, True, True, True, True, True, True, True, True, True, True, True)
        logger.info("Todos os testes passaram com sucesso!")
    except Exception as e:
        logger.error("Erro nos testes utilitarios ou de str_padrao_problema")
        logger.error(e)
        raise e


""" 
No simplex do cruzeiro: 
    o vetor x eh coluna
    o vetor c eh coluna
    o vetor m_simplex e coluna (m_simplex.T eh linha)
    
No simplex do sergio: 
    o vetor x eh coluna
    o vetor c eh linha
    o vetor m_simplex e linha (m_simplex.T eh coluna)
"""

# check_health_status()

# π cᵢ A⁻¹ λ Bᵗ cᵦ xᵦ cⱼ, cₙ Aᵗ
