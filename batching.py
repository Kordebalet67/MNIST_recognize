# -*- coding: utf-8 -*-
"""
mnist_numpy_batch.py
====================
Самописная трёхслойная сеть (input -> hidden -> output), сигмоида,
обратное распространение ошибки. Версия: NumPy + МИНИ-БАТЧИ (CPU).

СОГЛАШЕНИЕ О РАЗМЕРНОСТЯХ (важно!):
    один пример = СТОЛБЕЦ матрицы.
    X    : (input_nodes,  batch)
    T    : (output_nodes, batch)
    W_ih : (hidden_nodes, input_nodes)
    W_ho : (output_nodes, hidden_nodes)

Формат весов совместим с исходной версией: joblib-словарь {'ih', 'ho'}.
"""

import os
import time

import joblib
import numpy as np
import scipy.special


class MNIST_Neural_Network:
    """Трёхслойная сеть с одним скрытым слоем и батчевым обучением."""

    def __init__(self,
                 input_nodes=784,
                 hidden_nodes=200,
                 output_nodes=10,
                 learning_rate=0.5,
                 epochs=10,
                 batch_size=64,
                 grad_reduction="mean",   # "mean" | "sum"
                 shuffle=True,
                 seed=None,
                 dtype=np.float64):
        """
        grad_reduction:
            "mean" — градиент усредняется по батчу (стандарт).
                     Шаг НЕ зависит от размера батча, но он "слабее"
                     поштучного, поэтому learning_rate нужно ПОДНЯТЬ
                     относительно старых 0.1 (обычно 0.3 ... 1.0).
            "sum"  — градиент суммируется. Один батч-шаг примерно равен
                     batch_size поштучным шагам со старым lr.
                     Удобно, чтобы воспроизвести поведение старой версии.
        """
        if grad_reduction not in ("mean", "sum"):
            raise ValueError("grad_reduction должен быть 'mean' или 'sum'")
        if batch_size < 1:
            raise ValueError("batch_size должен быть >= 1")

        self.input_nodes = int(input_nodes)
        self.hidden_nodes = int(hidden_nodes)
        self.output_nodes = int(output_nodes)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.grad_reduction = grad_reduction
        self.shuffle = bool(shuffle)
        self.dtype = np.dtype(dtype)

        self._rng = np.random.default_rng(seed)

        # инициализация весов: std = 1/sqrt(кол-во входов узла)
        self.weights_input_hidden = self._rng.normal(
            0.0, pow(self.input_nodes, -0.5),
            (self.hidden_nodes, self.input_nodes)
        ).astype(self.dtype)

        self.weights_hidden_output = self._rng.normal(
            0.0, pow(self.hidden_nodes, -0.5),
            (self.output_nodes, self.hidden_nodes)
        ).astype(self.dtype)

        # НЕ lambda: обычная ссылка на функцию, чтобы объект можно было
        # сохранить целиком через joblib/pickle
        self.activation_function = scipy.special.expit

    # ------------------------------------------------------------------ #
    #  Работа с данными
    # ------------------------------------------------------------------ #
    def _asarray(self, x):
        return np.asarray(x, dtype=self.dtype)

    def _prepare_arrays(self, data):
        """data: (N, 1 + input_nodes), столбец 0 — метка. -> X, T (столбцы=примеры)."""
        data = np.asarray(data, dtype=self.dtype)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        expected = self.input_nodes + 1
        if data.shape[1] != expected:
            raise ValueError(
                f"Ожидалось {expected} столбцов (1 метка + {self.input_nodes} пикселей), "
                f"получено {data.shape[1]}"
            )

        labels = data[:, 0].astype(np.int64)
        if labels.min() < 0 or labels.max() >= self.output_nodes:
            raise ValueError("Метка выходит за диапазон output_nodes")

        # масштабирование входов: 0..255 -> 0.01..1.0
        X = (data[:, 1:] / 255.0 * 0.99) + 0.01

        # one-hot цели: 0.01 везде, 0.99 у правильного класса
        T = np.full((labels.size, self.output_nodes), 0.01, dtype=self.dtype)
        T[np.arange(labels.size), labels] = 0.99

        # транспонируем: примеры становятся СТОЛБЦАМИ
        X = np.ascontiguousarray(X.T, dtype=self.dtype)
        T = np.ascontiguousarray(T.T, dtype=self.dtype)
        return X, T

    def _load_dataset(self, path):
        data = np.loadtxt(path, delimiter=",", dtype=self.dtype)
        return self._prepare_arrays(data)

    # ------------------------------------------------------------------ #
    #  Обучение
    # ------------------------------------------------------------------ #
    def train_batch(self, X, T):
        """
        Один шаг обучения на батче.
        X: (input_nodes, batch), T: (output_nodes, batch).
        Возвращает MSE на батче (до обновления весов).
        """
        X = self._asarray(X)
        T = self._asarray(T)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if T.ndim == 1:
            T = T.reshape(-1, 1)

        if X.shape[0] != self.input_nodes or T.shape[0] != self.output_nodes:
            raise ValueError("Неверная форма X или T")
        if X.shape[1] != T.shape[1]:
            raise ValueError("В X и T разное количество примеров")

        n = X.shape[1]

        # ---- прямое распространение (сразу для всех примеров батча) ----
        H = self.activation_function(self.weights_input_hidden @ X)   # (hidden, n)
        O = self.activation_function(self.weights_hidden_output @ H)  # (output, n)

        # ---- ошибки ----
        E_out = T - O                                   # (output, n)
        E_hid = self.weights_hidden_output.T @ E_out    # (hidden, n) — по СТАРЫМ весам!

        # ---- градиенты и обновление ----
        scale = self.learning_rate / n if self.grad_reduction == "mean" else self.learning_rate

        self.weights_hidden_output += scale * ((E_out * O * (1.0 - O)) @ H.T)
        self.weights_input_hidden  += scale * ((E_hid * H * (1.0 - H)) @ X.T)

        return np.mean(E_out * E_out)

    def train(self, training_list_path="mnist_train_100.csv", verbose=True):
        """Обучение на CSV-файле. Возвращает пару матриц весов (как в исходной версии)."""
        X, T = self._load_dataset(training_list_path)
        n = X.shape[1]
        n_batches = (n + self.batch_size - 1) // self.batch_size

        if verbose:
            print(f"[CPU] примеров: {n}, батч: {self.batch_size}, "
                  f"батчей/эпоху: {n_batches}, эпох: {self.epochs}")

        t0 = time.perf_counter()

        for epoch in range(1, self.epochs + 1):
            # перемешиваем только индексы — сами данные не копируем
            order = self._rng.permutation(n) if self.shuffle else np.arange(n)

            epoch_loss = 0.0
            for b, start in enumerate(range(0, n, self.batch_size), start=1):
                idx = order[start:start + self.batch_size]
                loss = self.train_batch(X[:, idx], T[:, idx])
                epoch_loss += float(loss) * idx.size

                if verbose:
                    print(f"  epoch {epoch}/{self.epochs}  batch {b}/{n_batches}", end="\r")

            if verbose:
                print(f"  epoch {epoch}/{self.epochs}  "
                      f"MSE = {epoch_loss / n:.6f}  "
                      f"[{time.perf_counter() - t0:.2f} s]        ", end='\r')

        if verbose:
            print(f"[CPU] обучение завершено за {time.perf_counter() - t0:.3f} c")

        return self.weights_input_hidden, self.weights_hidden_output

    # ------------------------------------------------------------------ #
    #  Предсказания
    # ------------------------------------------------------------------ #
    def query(self, inputs_list):
        """Один пример: плоский список/вектор (784,) -> столбец (10, 1)."""
        X = self._asarray(inputs_list)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        H = self.activation_function(self.weights_input_hidden @ X)
        return self.activation_function(self.weights_hidden_output @ H)

    def query_batch(self, X):
        """Батч: X (input_nodes, batch) -> (output_nodes, batch)."""
        X = self._asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        H = self.activation_function(self.weights_input_hidden @ X)
        return self.activation_function(self.weights_hidden_output @ H)

    def predict(self, samples):
        """Удобная обёртка: samples (N, 784) строками -> метки (N,)."""
        S = self._asarray(samples)
        if S.ndim == 1:
            S = S.reshape(1, -1)
        return np.argmax(self.query_batch(S.T), axis=0)

    def return_answers(self, input_data):
        """-> (метка, уверенность в процентах). Совместимо с исходной версией."""
        out = np.asarray(self.query(input_data)).ravel()
        idx = int(np.argmax(out))
        return idx, float(out[idx]) * 100.0

    # ------------------------------------------------------------------ #
    #  Оценка качества
    # ------------------------------------------------------------------ #
    def score_info(self, test_data="mnist_test_10.csv", verbose=True):
        """test_data: путь к CSV или список строк CSV. Возвращает accuracy (0..1)."""
        if isinstance(test_data, (str, os.PathLike)):
            X, T = self._load_dataset(test_data)
        else:
            rows = [[float(v) for v in rec.split(",")] for rec in test_data]
            X, T = self._prepare_arrays(rows)

        labels = np.argmax(T, axis=0)
        n = X.shape[1]
        correct = 0

        for start in range(0, n, self.batch_size):
            sl = slice(start, start + self.batch_size)
            pred = np.argmax(self.query_batch(X[:, sl]), axis=0)
            correct += int(np.count_nonzero(pred == labels[sl]))

        acc = correct / n
        if verbose:
            print(f"performance = {acc}  ({correct}/{n})")
        return acc

    # ------------------------------------------------------------------ #
    #  Сохранение / загрузка весов
    # ------------------------------------------------------------------ #
    def save_weights(self, file_name: str):
        joblib.dump({'ih': np.asarray(self.weights_input_hidden),
                     'ho': np.asarray(self.weights_hidden_output)}, file_name)

    def load_weights(self, file_name: str):
        w = joblib.load(file_name)
        ih = np.asarray(w['ih'], dtype=self.dtype)
        ho = np.asarray(w['ho'], dtype=self.dtype)

        if ih.shape != (self.hidden_nodes, self.input_nodes):
            raise ValueError(f"W_ih: ожидалось {(self.hidden_nodes, self.input_nodes)}, получено {ih.shape}")
        if ho.shape != (self.output_nodes, self.hidden_nodes):
            raise ValueError(f"W_ho: ожидалось {(self.output_nodes, self.hidden_nodes)}, получено {ho.shape}")

        self.weights_input_hidden = ih
        self.weights_hidden_output = ho

    # ------------------------------------------------------------------ #
    #  Картинка -> ответ
    # ------------------------------------------------------------------ #
    def input_from_picture(self, picture_path, invert=True):
        """
        JPG/PNG с цифрой. invert=True — если цифра тёмная на светлом фоне
        (в MNIST наоборот: светлая цифра на тёмном фоне).
        """
        from PIL import Image, ImageOps  # pip install pillow

        side = int(round(self.input_nodes ** 0.5))
        img = Image.open(picture_path).convert("L")
        if img.size != (side, side):
            img = img.resize((side, side), Image.LANCZOS)
        if invert:
            img = ImageOps.invert(img)

        pixels = np.asarray(img.getdata(), dtype=self.dtype)  # getdata(), не get_flattened_data()
        pixels = (pixels / 255.0 * 0.99) + 0.01
        return self.return_answers(pixels)


# ====================================================================== #
#  Демонстрация
# ====================================================================== #
if __name__ == "__main__":
    TRAIN_FILE = "mnist_train.csv"
    TEST_FILE = "mnist_test.csv"
    WEIGHTS_FILE = "weights_numpy_batch.joblib"

    # Для 100 строк: маленький батч + побольше эпох.
    # Для полного MNIST (60000): batch_size=64, epochs=10, learning_rate=0.5
    net = MNIST_Neural_Network(
        input_nodes=784, hidden_nodes=200, output_nodes=10,
        learning_rate=0.25,      # для mean-градиента lr выше, чем старые 0.1
        epochs=50,
        batch_size=64,
        grad_reduction="mean",
        seed=42,
    )

    if os.path.exists(TRAIN_FILE):
        net.train(TRAIN_FILE)
        net.save_weights(WEIGHTS_FILE)
        print(f"веса сохранены -> {WEIGHTS_FILE}")
    else:
        print(f"нет файла {TRAIN_FILE} — обучение пропущено")
        if os.path.exists(WEIGHTS_FILE):
            net.load_weights(WEIGHTS_FILE)
            print(f"веса загружены <- {WEIGHTS_FILE}")

    if os.path.exists(TEST_FILE):
        net.score_info(TEST_FILE)