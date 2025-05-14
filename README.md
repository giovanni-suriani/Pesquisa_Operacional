## Se quiser usar o simplex do cruzeiro, va em simplex_algorithm.py as matrizes A, b e c ja na forma padrao e rode, os resultados apareceram em forma de log

**Tem um exemplo comentado do primeiro exercicio da lista 1**

```python 
A = [[2,  1, 1, 0, 0],
    [1, 2, 0, 1, 0],
    [0, 1, 0, 0, 1]]
b = [[8], [7], [3]]
c = [-1, -1, 0, 0, 0]
I0 = [["x3"], ["x4"], ["x5"]]
x = [["x1"], ["x2"], ["x3"], ["x4"], ["x5"]] 


solve_simplex_cruzeiro(A, b, c, x, I0, type_func="min")