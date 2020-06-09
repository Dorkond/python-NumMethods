import numpy as np


# Генерация квадратной матрицы
def matrix_generation(n):
    return np.random.randint(0, 10, (n, n)).astype(np.float32)


# Генерация вырожденной квадратной матрицы
def singular_matrix_generation(n):
    matr = matrix_generation(n)
    matr[1, :] = 2 * matr[0, :]
    return matr


def generation_matrix_diag_pred(n):
    matrix_1 = np.random.uniform(0, 10, (n, n)) * (np.ones((n, n)) - np.eye(n))
    matrix_2 = np.random.uniform(10 * n, 10 * n + 10, (n, n)) * np.eye(n)
    return matrix_1 + matrix_2


def generation_matrix_positive(n):
    matrix = np.random.uniform(0, 10, (n, n))
    return np.transpose(matrix).dot(matrix)


# Генерация вектора длиной n
def vector_generation(n):
    return np.random.randint(0, 10, (n, 1)).astype(np.float32)


# Свап строчек k и n в матрице
def swap_rows(matrix, k, n):
    matr = np.copy(matrix)
    matr[n, :] = matr[k, :]
    matr[k, :] = matrix[n, :]
    return matr


# Свап столбцов k, n в матрице
def swap_columns(matrix, k, n):
    matr = np.copy(matrix)
    matr[:, n] = matr[:, k]
    matr[:, k] = matrix[:, n]
    return matr


# Свап столбцов через листы
def swap_columns2(your_list, pos1, pos2):
    for item in your_list:
        item[pos1], item[pos2] = item[pos2], item[pos1]


def swap_rows2(input_array, row1, row2):
    temp = np.copy(input_array[row1][:])
    input_array[row1][:] = input_array[row2][:]
    input_array[row2][:] = temp


# Det of a matrix
def det_U(matrix, size, swaps):
    det = (-1) ** swaps
    for i in range(size):
        det *= matrix[i, i]
    return det


# LU разложение квадратной матрицы
def decompose_LU(matrix, n):
    matrix_c = np.copy(matrix)

    swaps = 0
    matrix_p = np.eye(n)
    matrix_q = np.eye(n)
    matrix_l = np.zeros((n, n))

    for i in range(n):
        pivot = -1
        pivotValue = np.fabs(matrix_c[i, i])
        for row in range(i + 1, n):
            if np.fabs(matrix_c[row, i]) > pivotValue:
                pivotValue = np.fabs(matrix_c[row, i])
                pivot = row
        if pivot != -1:
            swaps += 1
            matrix_c = swap_rows(matrix_c, pivot, i)
            matrix_p = swap_rows(matrix_p, pivot, i)
            matrix_l = swap_rows(matrix_l, pivot, i)
        matrix_l[i, i] = 1
        for j in range(i + 1, n):
            coeff = matrix_c[j, i] / matrix_c[i, i]
            matrix_l[j, i] = coeff
            for k in range(i, n):
                matrix_c[j, k] -= coeff * matrix_c[i, k]

    matrix_u = matrix_c
    return matrix_u, matrix_l, matrix_p, matrix_q, swaps


# Решение системы LU разложения обратным ходом
def system_solution_LU(matrix_l, matrix_u, n, b, matrix_p):
    # Ax = b        =>           PAx = LUx = Pb
    res_x = np.zeros((n, 1))
    b = matrix_p.dot(b)
    y = []
    x = []
    for i in range(n):
        k = b[i]
        for j in range(n)[0:i]:
            k -= matrix_l[i, j] * y[j]
        y.append(k)
    for i in range(n):
        k = y[n - (i + 1)]
        for j in range(n)[0:i]:
            k -= matrix_u[n - (i + 1), n - (j + 1)] * x[j]
        x.append(k / matrix_u[n - (i + 1), n - (i + 1)])
    x.reverse()
    for i in range(n):
        res_x[i, 0] = x[i]
    return res_x


# Обращение матрицы
def inverse(matrix_l, matrix_u, matrix_p, n):
    matrix_inv = []
    eye = np.eye(n)
    for i in range(n):
        matrix_inv.append(system_solution_LU(matrix_l, matrix_u, n, eye[:, i], matrix_p))
    return np.transpose(matrix_inv)


# QR - разложение
def decompose_QR(matrix, n):
    matrix_q = np.zeros((n, n))
    matrix_r = np.zeros((n, n))
    for j in range(n):
        matrix_q[:, j] = matrix[:, j]
        for i in range(j - 2):
            matrix_r[i, j] = np.transpose(matrix_q[:, i]).dot(matrix[:, j])
            matrix_q[:, j] = matrix_q[:, j] - matrix_r[j, i] * matrix_q[:, i]
        matrix_r[j, j] = np.linalg.norm(matrix_q[:, j])
        if matrix_r[j, j] == 0:
            pass
        else:
            matrix_q[:, j] = matrix_q[:, j] / matrix_r[j, j]

    return matrix_q, matrix_r
    # QR - разложение


# Разрешение системы QR - разложением
def system_solution_QR(matrix_r, matrix_Q, b):
    # A = QR => Ax = b => QRx = b => Q^(-1)*QRx = Q^(-1)b => ERx = Q^(-1)b => Rx = Q^(-1)b
    return np.linalg.solve(matrix_r, (np.linalg.inv(matrix_Q).dot(b)))


# Решение системы методом Зейделя
def zeidel(matrix, b, accuracy, n):
    # matrix_D - диагональная матрица с элементами из оригинальной матрицы
    # matrix_L - нижнетреугольная матрица
    # matrix_U - верхнетреугольная матрица
    aprior = 0
    iters = 1
    matrix_D = np.zeros((n, n))
    for i in range(n):
        matrix_D[i, i] = matrix[i, i]
    matrix_L = np.tril(matrix) - matrix_D
    matrix_U = np.triu(matrix) - matrix_D
    x_previous = np.zeros((n, 1))
    # Введём вспомогательные матрицы В и g для оценки
    g = np.linalg.inv(np.tril(matrix)).dot(b)
    B = -1 * np.linalg.inv(np.tril(matrix)).dot(matrix_U)
    q = 1
    if np.linalg.norm(B) < 1:
        aprior = 1 + int((np.log(accuracy) + np.log(1 - np.linalg.norm(B)) - np.log(np.linalg.norm(g))) / np.log(np.linalg.norm(B)))
        q = np.linalg.norm(B) / (1 - np.linalg.norm(B))
    else:
        aprior = "Норма больше 1"

    x_current = ((np.linalg.inv(matrix_L + matrix_D)).dot(-1 * matrix_U.dot(x_previous))) + (
        np.linalg.inv(matrix_L + matrix_D).dot(b))
    norma = 10000
    while q * norma > accuracy:
        iters += 1
        x_current = ((np.linalg.inv(matrix_L + matrix_D)).dot(-1 * matrix_U.dot(x_previous))) + (
            np.linalg.inv(matrix_L + matrix_D).dot(b))
        norma = np.linalg.norm(x_current - x_previous)
        if norma > 10000:
            print("☰☰☰Метод Зейделя разошёлся☰☰☰")
            return None, None, None
        x_previous = np.copy(x_current)

    return x_current, aprior, iters


# Решение системы методом Якоби
def jacobi(matrix, b, accuracy, n):
    # matrix_D - диагональная матрица с элементами из оригинальной матрицы
    # matrix_L - нижнетреугольная матрица
    # matrix_R - верхнетреугольная матрица
    aprior = 0
    iters = 1
    matrix_D = np.zeros((n, n))
    for i in range(n):
        matrix_D[i, i] = matrix[i, i]
    matrix_L = np.tril(matrix) - matrix_D
    matrix_R = np.triu(matrix) - matrix_D
    x_previous = np.zeros((n, 1))
    # Введём вспомогательные матрицы В и g для оценки
    g = np.linalg.inv(matrix_D).dot(b)
    B = np.eye(n) - np.linalg.inv(matrix_D).dot(matrix)
    q = 1
    if np.linalg.norm(B) < 1:
        aprior = 1 + int(
            (np.log(accuracy) + np.log(1 - np.linalg.norm(B)) - np.log(np.linalg.norm(g))) / np.log(np.linalg.norm(B)))
        q = np.linalg.norm(B) / (1 - np.linalg.norm(B))
    else:
        aprior = "Норма больше 1"

    x_current = (-1 * (np.linalg.inv(matrix_D).dot(matrix_L + matrix_R))).dot(x_previous) + (np.linalg.inv(matrix_D).dot(b))
    norma = 10000
    while q * norma > accuracy:
        iters += 1
        x_current = (-1 * (np.linalg.inv(matrix_D).dot(matrix_L + matrix_R))).dot(x_previous) + (
            np.linalg.inv(matrix_D).dot(b))
        norma = np.linalg.norm(x_current - x_previous)
        if norma > 10000:
            print("☰☰☰Метод Якоби разошёлся☰☰☰")
            return None, None, None
        x_previous = np.copy(x_current)

    return x_current, aprior, iters


# Разложение PAQ(LU)
def decompose_PAQ_LU(matrix, n):
    swaps = 0
    matrix_c = np.copy(matrix)
    matrix_p = np.eye(n)
    matrix_q = np.eye(n)

    for i in range(n):
        pivotValue = 0
        pivot1 = -1
        pivot2 = -1
        for row in range(n)[i:n]:
            for column in range(n)[i:n]:
                if np.fabs(matrix_c[row][column]) > pivotValue:
                    pivotValue = np.fabs(matrix_c[row][column])
                    pivot1 = row
                    pivot2 = column
        if pivotValue != 0:
            if pivot1 != i:
                swaps += 1
                swap_rows2(matrix_p, pivot1, i)
                swap_rows2(matrix_c, pivot1, i)
            if pivot2 != i:
                swaps += 1
                swap_columns2(matrix_q, pivot2, i)
                swap_columns2(matrix_c, pivot2, i)
            for j in range(n)[i + 1:n]:
                matrix_c[j][i] /= matrix_c[i][i]
                for s in range(n)[i + 1:n]:
                    matrix_c[j][s] -= matrix_c[j][i] * matrix_c[i][s]

    matrix_l = np.tril(np.array(matrix_c), -1) + np.identity(n)
    matrix_u = np.triu(np.array(matrix_c), 0)
    return matrix_l, matrix_u, matrix_p, matrix_q, swaps


# Ранг матрицы
def rang(matrix, n):
    rank = n
    for row in range(n):
        if zeros(matrix[n - 1 - row], n):
            rank -= 1
        else:
            break
    return rank


# Нулевая ли строка
def zeros(row, n):
    for i in range(n):
        if abs(row[i]) - 0.0000001 > 0:
            return False
    return True


# Для решения системы с совместной матрицей
def system_solution_2(matrix_l, matrix_u, n, b, matrix_p, matrix_q):
    res_x = np.zeros((n, 1))
    b = matrix_p.dot(b)
    y = []
    x = []
    for i in range(n):
        k = b[i]
        for j in range(n)[0:i]:
            k -= matrix_l[i, j] * y[j]
        y.append(k)
    for i in range(n):
        k = y[n - (i + 1)]
        for j in range(n)[0:i]:
            k -= matrix_u[n - (i + 1), n - (j + 1)] * x[j]
        x.append(k / matrix_u[n - (i + 1), n - (i + 1)])
    x.reverse()
    for i in range(n):
        res_x[i, 0] = x[i]
    return matrix_q.dot(res_x)


# Для решения системы с вырожденной матрицей
def system_solution_3(L, U, P, Q, b, rank, n):
    cU = np.copy(U)
    g = np.linalg.inv(L).dot(P).dot(b)
    y = np.zeros(n)
    for i in range(rank)[::-1]:
        g[i] = g[i] / cU[i, i]
        cU[i, :] /= cU[i, i]
        for j in range(i):
            g[j] -= g[i] * cU[j, i]
            cU[j, :] -= cU[i, :] * cU[j, i]
        y[i] = g[i]
    x = Q.dot(y)
    x_res = np.zeros((n, 1))
    for i in range(n):
        x_res[i, 0] = x[i]
    return x_res
