import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. СОЗДАНИЕ И ОБРАБОТКА МАССИВОВ
# ============================================================

def create_vector() -> np.ndarray:
    """
        Создать массив от 0 до 9.
        Returns:
        numpy.ndarray: Массив чисел от 0 до 9 включительно
    """
    return np.arange(10)


def create_matrix() -> np.ndarray:
    """
        Создать матрицу 5x5 со случайными числами [0,1].
        Returns:
        numpy.ndarray: Матрица 5x5 со случайными значениями от 0 до 1
    """
    return np.random.rand(5,5)


def reshape_vector(vec: np.ndarray) -> np.ndarray:
    """
    Преобразовать (10,) -> (2,5)


    Args:
        vec (numpy.ndarray): Входной массив формы (10,)

    Returns:
        numpy.ndarray: Преобразованный массив формы (2, 5)
    """
    return vec.reshape(2,5)


def transpose_matrix(mat: np.ndarray) -> np.ndarray:
    """
    Транспонирование матрицы.

    Args:
        mat (numpy.ndarray): Входная матрица

    Returns:
        numpy.ndarray: Транспонированная матрица
    """
    return mat.T


# ============================================================
# 2. ВЕКТОРНЫЕ ОПЕРАЦИИ
# ============================================================

def vector_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Сложение векторов одинаковой длины.
    (Векторизация без циклов)

    Args:
        a (numpy.ndarray): Первый вектор
        b (numpy.ndarray): Второй вектор

    Returns:
        numpy.ndarray: Результат поэлементного сложения
    """
    return a + b



def scalar_multiply(vec: np.ndarray, scalar: float) -> np.ndarray:
    """
    Умножение вектора на число.

    Args:
        vec (numpy.ndarray): Входной вектор
        scalar (float/int): Число для умножения

    Returns:
        numpy.ndarray: Результат умножения вектора на скаляр
    """
    return vec * scalar



def elementwise_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Поэлементное умножение.

    Args:
        a (numpy.ndarray): Первый вектор/матрица
        b (numpy.ndarray): Второй вектор/матрица

    Returns:
        numpy.ndarray: Результат поэлементного умножения
    """
    return a * b

def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """
    Скалярное произведение

    Args:
        a (numpy.ndarray): Первый вектор
        b (numpy.ndarray): Второй вектор

    Returns:
        float: Скалярное произведение векторов
    """
    return float(np.dot(a, b))


# ============================================================
# 3. МАТРИЧНЫЕ ОПЕРАЦИИ
# ============================================================

def matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Умножение матриц.

    Args:
        a (numpy.ndarray): Первая матрица
        b (numpy.ndarray): Вторая матрица

    Returns:
        numpy.ndarray: Результат умножения матриц
    """
    return  a @ b


def matrix_determinant(a: np.ndarray) -> float:
    """
    Определитель матрицы.

    Args:
        a (numpy.ndarray): Квадратная матрица

    Returns:
        float: Определитель матрицы
    """
    return float(np.linalg.det(a))


def matrix_inverse(a: np.ndarray) -> np.ndarray:
    """
    Обратная матрица.


    Args:
        a (numpy.ndarray): Квадратная матрица

    Returns:
        numpy.ndarray: Обратная матрица
    """
    return np.linalg.inv(a)


def solve_linear_system(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Решить систему Ax = b

    Args:
        a (numpy.ndarray): Матрица коэффициентов A
        b (numpy.ndarray): Вектор свободных членов b

    Returns:
        numpy.ndarray: Решение системы x
    """
    return np.linalg.solve(a, b)


# ============================================================
# 4. СТАТИСТИЧЕСКИЙ АНАЛИЗ
# ============================================================

def load_dataset(path: str = "data/students_scores.csv") -> np.ndarray:
    """
    Загрузить CSV и вернуть NumPy массив.

    Args:
        path (str): Путь к CSV файлу

    Returns:
        numpy.ndarray: Загруженные данные в виде массива
    """
    df = pd.read_csv(path)
    return df.to_numpy()


def statistical_analysis(data: np.ndarray) -> dict:
    """
    Args:
        data (numpy.ndarray): Одномерный массив данных

    Returns:
        dict: Словарь со статистическими показателями
    """
    return {
        "mean": np.mean(data),
        "median": np.median(data),
        "std": np.std(data),
        "min": np.min(data),
        "max": np.max(data),
        "p25": np.percentile(data, 25),
        "p75": np.percentile(data, 75),
    }


def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Min-Max нормализация.

    Формула: (x - min) / (max - min)

    Args:
        data (numpy.ndarray): Входной массив данных

    Returns:
        numpy.ndarray: Нормализованный массив данных в диапазоне [0, 1]
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# ============================================================
# 5. ВИЗУАЛИЗАЦИЯ
# ============================================================

def plot_histogram(data: np.ndarray, save_path='plots/math_histogram.png') -> None:
    """
    Построить гистограмму распределения оценок по математике.

    Args:
        data (numpy.ndarray): Данные для гистограммы
        save_path (str): Путь для сохранения графика
    """
    plt.figure(figsize=(5, 5))
    plt.hist(data, bins=10, edgecolor="black", alpha=0.7, color="skyblue")

    plt.title("Распределение оценок по математике")
    plt.xlabel("Оценка")
    plt.ylabel("Количество студентов")
    plt.grid(True, linestyle="--", alpha=0.3)

    #  Создаем папку
    os.makedirs(os.path.dirname(save_path), exist_ok=True)


    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_heatmap(matrix: np.ndarray, save_path: str = "plots/heatmap.png") -> None:
    """
    Построить и сохранить тепловую карту корреляции предметов.

    Args:
        matrix (np.ndarray): Матрица корреляции
        save_path (str): Путь для сохранения графика
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Корреляция предметов")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_line(x: np.ndarray, y: np.ndarray) -> None:
    """
    Построить график зависимости: студент -> оценка по математике.
      Args:
        x (np.ndarray): номера студентов
        y (np.ndarray): оценки студентов
    """


    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker="o")

    plt.title("Зависимость оценки по математике от номера студента")
    plt.xlabel("Номер студента")
    plt.ylabel("Оценка по математике")

    plt.grid(True)

    os.makedirs("plots", exist_ok=True)

    plt.savefig("plots/students_scores_line.png")
    plt.close()

if __name__ == "__main__":
    # Пример данных
    math_scores = np.array([5, 4, 5, 3, 5, 4, 2, 3, 5, 4])
    students = np.arange(1, len(math_scores)+1)

    # Построение и сохранение графиков
    plot_histogram(math_scores)
    plot_line(students, math_scores)




