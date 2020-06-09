import numpy as np
from func_first import *

# Выбран поиск опорного элемента по столбцу => матрица перестановки столбцов - диагональная, det(Q) = 1

# Размерность матрицы
size = 6


print("☰☰Задание 1☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰")
matrix = matrix_generation(size)
print("☰☰Матрица системы☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", matrix)
matrix_u, matrix_l, matrix_p, matrix_q, swaps = decompose_LU(matrix, size)
print("☰☰☰Матрица L☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", matrix_l)
print("☰☰☰Матрица U☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", matrix_u)
print("☰☰☰Матрица Q(перестановки столбцов)☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", matrix_q)
print("☰☰☰Матрица P(перестановки строк)☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", matrix_p)
print("☰☰☰Матрица PA☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", matrix_p.dot(matrix))
print("☰☰☰Матрица LU☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", matrix_l.dot(matrix_u))
print("☰☰☰Детерминант A☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", np.linalg.det(matrix))
print("☰☰☰Детерминант через диагональные элементы U☰☰☰☰☰☰☰☰☰☰☰" + "\n", det_U(matrix_u, size, swaps))
b = vector_generation(size)
x = system_solution_LU(matrix_l, matrix_u, size, b, matrix_p)
print("☰☰☰Система с коэффициентами B☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", matrix, "\n", b)
print("☰☰☰Решение системы☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", x)
print("☰☰☰Проверка решения произведением Ax☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", matrix.dot(x))
matrix_inv = inverse(matrix_l, matrix_u, matrix_p, size)
print("☰☰☰Обратная матрица внутренней функцией numpy☰☰☰☰☰☰☰☰☰☰" + "\n", np.linalg.inv(matrix))
print("☰☰☰Обратная матрица через U^(-1) * L^(-1) * P☰☰☰☰☰☰☰☰☰☰" + "\n",
      (np.linalg.inv(matrix_u).dot(np.linalg.inv(matrix_l))).dot(matrix_p))
print("☰☰☰Обратная матрица через LU разложение☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", matrix_inv)
print("☰☰☰A * A^(-1)☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", matrix.dot(matrix_inv))
print("☰Число обусловленности матрицы А☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n",
      np.linalg.norm(matrix_inv) * np.linalg.norm(matrix))


print("☰☰Задание 2☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰")
sing_m = singular_matrix_generation(size)
L, U, P, Q, swps = decompose_PAQ_LU(sing_m, size)
print("☰☰☰Вырожденнная матрица системы☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", sing_m)
print("☰☰☰Матрица L☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", L)
print("☰☰☰Матрица U☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", U)
print("☰☰☰Проверка правильности разложения (ч/з LU)☰☰☰☰☰☰☰☰☰☰☰" + "\n", L.dot(U))
print("☰☰☰Проверка правильности разложения (ч/з P*A*Q)☰☰☰☰☰☰☰☰☰" + "\n", P.dot(sing_m).dot(Q))
rank_U = rang(U, size)
print("☰☰☰Ранг матрицы☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", rank_U)
expan_matr = np.zeros((size, size + 1))  # Дорасширение матрицы столбцом коэффициентов
for i in range(size):
    expan_matr[:, i] = sing_m[:, i]
for i in range(size):
    expan_matr[i, size] = b[i, 0]

# Проверка на совместность по теореме Кроннекера — Капелли
if rank_U == np.linalg.matrix_rank(expan_matr):
    print("☰☰☰Система совместна☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰")
    if rank_U == size:
        x = system_solution_2(L, U, size, b, P, Q)
        print("☰☰☰Решение☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", x)
        print("☰☰☰Проверка ч/з Ax☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", sing_m.dot(x))
    else:
        x = system_solution_3(L, U, P, Q, b, rank_U, size)
        print("☰☰☰Частное решение☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", x)
        print("☰☰☰Проверка решения через Ax☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", sing_m.dot(x))
else:
    print("☰☰☰Система несовместна☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰")


print("☰☰Задание 3☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰")
matrix_Q, matrix_r = decompose_QR(matrix, size)
print("☰☰☰Матрица Q разложения QR☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", matrix_Q)
print("☰☰☰Матрица R разложения QR☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", matrix_r)
print("☰☰☰Матрица Q * R☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", matrix_Q.dot(matrix_r))
print("☰☰☰Решение X☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n",
      system_solution_QR(matrix_r, matrix_Q, b))


print("☰☰Задание 4☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰")
accuracy = 1e-12
matrix_diag = generation_matrix_diag_pred(size)
x, apr, iterations = zeidel(matrix_diag, b, accuracy, size)
print("☰☰☰Методы Зейделя и Якоби☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰")

print("☰☰☰Матрица с диагональным преобладанием☰☰☰☰☰☰☰☰☰☰☰☰☰")
print("☰☰☰Матрица А и столбец b☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", matrix_diag, "\n", b)
print("☰☰☰Решение системы встроенным методом☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", np.linalg.solve(matrix_diag, b))
print("☰☰☰Реализованный метод Зейделя☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", x)
print("☰☰☰Априорная оценка☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", apr)
print("☰☰☰Количество итераций☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", iterations)
x, apr, iterations = jacobi(matrix_diag, b, accuracy, size)
print("☰☰☰Метод Якоби☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", x)
print("☰☰☰Априорная оценка☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", apr)
print("☰☰☰Количество итераций☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", iterations)


matrix_positive = generation_matrix_positive(size)
x, apr, iterations = zeidel(matrix_positive, b, accuracy, size)
print("☰☰☰Положительно определенная матрица без диаг.преобладания☰☰☰")
print("☰☰☰Матрица А и столбец b☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", matrix_positive, "\n", b)
print("☰☰☰Решение системы встроенным методом☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", np.linalg.solve(matrix_positive, b))
print("☰☰☰Реализованный метод Зейделя☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", x)
print("☰☰☰Априорная оценка☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", apr)
print("☰☰☰Количество итераций☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", iterations)
x, apr, iterations = jacobi(matrix_positive, b, accuracy, size)
print("☰☰☰Метод Якоби☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", x)
print("☰☰☰Априорная оценка☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", apr)
print("☰☰☰Количество итераций☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰☰" + "\n", iterations)
