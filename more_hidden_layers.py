# -*- coding: utf-8 -*-
"""
ИИ-сгенерированный файл на основе нашей трехслойной модели 
============================
Самописная сеть: input -> hidden1 -> hidden2 -> output.
Сигмоида, ручное обратное распространение, МИНИ-БАТЧИ, NumPy (CPU).

РАЗМЕРНОСТИ (один пример = СТОЛБЕЦ):
    X  : (input_nodes,   batch)
    T  : (output_nodes,  batch)
    W1 : (hidden1_nodes, input_nodes)     input   -> hidden1
    W2 : (hidden2_nodes, hidden1_nodes)   hidden1 -> hidden2
    W3 : (output_nodes,  hidden2_nodes)   hidden2 -> output

Формат файла весов: joblib-словарь {'w1', 'w2', 'w3', 'arch'}.
ВНИМАНИЕ: веса старой 3-слойной версии ({'ih','ho'}) НЕ подходят.
"""

import os
import time

import joblib
import numpy as np
import scipy.special


class MNIST_Neural_Network:
    """Четырёхслойная сеть (два скрытых слоя) с батчевым обучением."""

    def __init__(self,
                 input_nodes=784,
                 hidden1_nodes=200,      # НОВОЕ: первый скрытый слой
                 hidden2_nodes=100,      # НОВОЕ: второй скрытый слой
                 output_nodes=10,
                 learning_rate=0.5,
                 epochs=10,
                 batch_size=64,
                 grad_reduction="mean",  # "mean" | "sum"
                 shuffle=True,
                 seed=None,
                 dtype=np.float64):
        """
        grad_reduction:
            "mean" — градиент усредняется по батчу. Шаг не зависит от размера
                     батча, но слабее поштучного -> learning_rate выше.
            "sum"  — градиент суммируется. Один батч-шаг ~ batch_size
                     поштучных шагов со старым lr.

        Про глубину: три сигмоиды подряд ослабляют градиент (произведение
        трёх s'(z) <= 0.25^3). Поэтому 4-слойная сеть учится медленнее
        3-слойной: нужен lr побольше и эпох побольше.
        """
        if grad_reduction not in ("mean", "sum"):
            raise ValueError("grad_reduction должен быть 'mean' или 'sum'")
        if batch_size < 1:
            raise ValueError("batch_size должен быть >= 1")

        self.input_nodes = int(input_nodes)
        self.hidden1_nodes = int(hidden1_nodes)
        self.hidden2_nodes = int(hidden2_nodes)
        self.output_nodes = int(output_nodes)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.grad_reduction = grad_reduction
        self.shuffle = bool(shuffle)
        self.dtype = np.dtype(dtype)

        self._rng = np.random.default_rng(seed)

        # инициализация: std = 1/sqrt(кол-во входов узла)
        self.weights_input_hidden1 = self._rng.normal(
            0.0, pow(self.input_nodes, -0.5),
            (self.hidden1_nodes, self.input_nodes)).astype(self.dtype)

        self.weights_hidden1_hidden2 = self._rng.normal(
            0.0, pow(self.hidden1_nodes, -0.5),
            (self.hidden2_nodes, self.hidden1_nodes)).astype(self.dtype)

        self.weights_hidden2_output = self._rng.normal(
            0.0, pow(self.hidden2_nodes, -0.5),
            (self.output_nodes, self.hidden2_nodes)).astype(self.dtype)

        # не lambda — чтобы объект сериализовался через joblib/pickle
        self.activation_function = scipy.special.expit

    # ------------------------------------------------------------------ #
    #  Служебное
    # ------------------------------------------------------------------ #
    @property
    def architecture(self):
        return (self.input_nodes, self.hidden1_nodes,
                self.hidden2_nodes, self.output_nodes)

    def describe(self):
        return (f"{self.input_nodes} -> {self.hidden1_nodes} -> "
                f"{self.hidden2_nodes} -> {self.output_nodes}")

    def _asarray(self, x):
        return np.asarray(x, dtype=self.dtype)

    # ------------------------------------------------------------------ #
    #  Данные
    # ------------------------------------------------------------------ #
    def _prepare_arrays(self, data):
        """data (N, 1+input_nodes), столбец 0 — метка. -> X, T (столбцы=примеры)."""
        data = np.asarray(data, dtype=self.dtype)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        expected = self.input_nodes + 1
        if data.shape[1] != expected:
            raise ValueError(
                f"Ожидалось {expected} столбцов (1 метка + {self.input_nodes} пикселей), "
                f"получено {data.shape[1]}")

        labels = data[:, 0].astype(np.int64)
        if labels.min() < 0 or labels.max() >= self.output_nodes:
            raise ValueError("Метка выходит за диапазон output_nodes")

        X = (data[:, 1:] / 255.0 * 0.99) + 0.01

        T = np.full((labels.size, self.output_nodes), 0.01, dtype=self.dtype)
        T[np.arange(labels.size), labels] = 0.99

        X = np.ascontiguousarray(X.T, dtype=self.dtype)
        T = np.ascontiguousarray(T.T, dtype=self.dtype)
        return X, T

    def _load_dataset(self, path):
        return self._prepare_arrays(np.loadtxt(path, delimiter=",", dtype=self.dtype))

    # ------------------------------------------------------------------ #
    #  Обучение
    # ------------------------------------------------------------------ #
    def train_batch(self, X, T):
        """Один шаг обучения на батче. Возвращает MSE (до обновления весов)."""
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
        act = self.activation_function

        # ---- прямое распространение: 3 умножения вместо 2 ----
        H1 = act(self.weights_input_hidden1 @ X)      # (hidden1, n)
        H2 = act(self.weights_hidden1_hidden2 @ H1)   # (hidden2, n)
        O  = act(self.weights_hidden2_output @ H2)    # (output,  n)

        # ---- ошибки: ВСЕ считаем ДО обновления весов! ----
        E_out = T - O                                        # (output,  n)
        E_h2  = self.weights_hidden2_output.T @ E_out         # (hidden2, n)
        E_h1  = self.weights_hidden1_hidden2.T @ E_h2         # (hidden1, n)

        # ---- градиенты и обновление (от выхода к входу) ----
        scale = self.learning_rate / n if self.grad_reduction == "mean" else self.learning_rate

        self.weights_hidden2_output += scale * ((E_out * O * (1.0 - O)) @ H2.T)
        self.weights_hidden1_hidden2 += scale * ((E_h2 * H2 * (1.0 - H2)) @ H1.T)
        self.weights_input_hidden1  += scale * ((E_h1 * H1 * (1.0 - H1)) @ X.T)

        return np.mean(E_out * E_out)

    def train(self, training_list_path="mnist_train_100.csv", verbose=True):
        """Обучение на CSV. Возвращает тройку матриц весов."""
        X, T = self._load_dataset(training_list_path)
        n = X.shape[1]
        n_batches = (n + self.batch_size - 1) // self.batch_size

        if verbose:
            print(f"[CPU] {self.describe()} | примеров: {n} | батч: {self.batch_size} | "
                  f"батчей/эпоху: {n_batches} | эпох: {self.epochs}")

        t0 = time.perf_counter()

        for epoch in range(1, self.epochs + 1):
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
                      f"[{time.perf_counter() - t0:.2f} s]        ")

        if verbose:
            print(f"[CPU] обучение завершено за {time.perf_counter() - t0:.3f} c")

        return (self.weights_input_hidden1,
                self.weights_hidden1_hidden2,
                self.weights_hidden2_output)

    # ------------------------------------------------------------------ #
    #  Предсказания
    # ------------------------------------------------------------------ #
    def query(self, inputs_list):
        """Один пример: (784,) -> (10, 1)."""
        X = self._asarray(inputs_list)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.query_batch(X)

    def query_batch(self, X):
        """Батч: X (input_nodes, batch) -> (output_nodes, batch)."""
        X = self._asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        act = self.activation_function
        H1 = act(self.weights_input_hidden1 @ X)
        H2 = act(self.weights_hidden1_hidden2 @ H1)
        return act(self.weights_hidden2_output @ H2)

    def predict(self, samples):
        """samples (N, 784) строками -> метки (N,)."""
        S = self._asarray(samples)
        if S.ndim == 1:
            S = S.reshape(1, -1)
        return np.argmax(self.query_batch(S.T), axis=0)

    def return_answers(self, input_data):
        """-> (метка, уверенность в процентах)."""
        out = np.asarray(self.query(input_data)).ravel()
        idx = int(np.argmax(out))
        return idx, float(out[idx]) * 100.0

    # ------------------------------------------------------------------ #
    #  Оценка качества
    # ------------------------------------------------------------------ #
    def score_info(self, test_data="mnist_test_10.csv", verbose=True):
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
    #  Сохранение / загрузка весов (НОВЫЙ формат: 3 матрицы + архитектура)
    # ------------------------------------------------------------------ #
    def save_weights(self, file_name: str):
        joblib.dump({'w1': np.asarray(self.weights_input_hidden1),
                     'w2': np.asarray(self.weights_hidden1_hidden2),
                     'w3': np.asarray(self.weights_hidden2_output),
                     'arch': self.architecture}, file_name)

    def load_weights(self, file_name: str):
        w = joblib.load(file_name)

        if 'w1' not in w:
            raise ValueError(
                "Похоже, это веса старой 3-слойной сети ({'ih','ho'}). "
                "Для 4-слойной сети нужны 3 матрицы — переобучи модель "
                "и сохрани заново.")

        expected = {
            'w1': (self.hidden1_nodes, self.input_nodes),
            'w2': (self.hidden2_nodes, self.hidden1_nodes),
            'w3': (self.output_nodes, self.hidden2_nodes),
        }
        for key, shape in expected.items():
            if w[key].shape != shape:
                raise ValueError(
                    f"{key}: ожидалось {shape}, получено {w[key].shape}. "
                    f"Текущая архитектура: {self.describe()}")

        self.weights_input_hidden1 = np.asarray(w['w1'], dtype=self.dtype)
        self.weights_hidden1_hidden2 = np.asarray(w['w2'], dtype=self.dtype)
        self.weights_hidden2_output = np.asarray(w['w3'], dtype=self.dtype)

    # ------------------------------------------------------------------ #
    #  Картинка -> ответ
    # ------------------------------------------------------------------ #
    def input_from_picture(self, picture_path, invert=True):
        from PIL import Image, ImageOps  # pip install pillow

        side = int(round(self.input_nodes ** 0.5))
        img = Image.open(picture_path).convert("L")
        if img.size != (side, side):
            img = img.resize((side, side), Image.LANCZOS)
        if invert:
            img = ImageOps.invert(img)

        pixels = np.asarray(img.getdata(), dtype=self.dtype)
        pixels = (pixels / 255.0 * 0.99) + 0.01
        return self.return_answers(pixels)


# ====================================================================== #
#  Демонстрация
# ====================================================================== #
if __name__ == "__main__":
    TRAIN_FILE = "mnist_train_100.csv"
    TEST_FILE = "mnist_test_10.csv"
    WEIGHTS_FILE = "weights_numpy_2hidden.joblib"

    # Для 100 строк: маленький батч + много эпох + lr повыше (градиент глубже).
    # Для полного MNIST (60000): batch_size=64, epochs=15, learning_rate=0.5
    net = MNIST_Neural_Network(
        input_nodes=784,
        hidden1_nodes=200,
        hidden2_nodes=100,
        output_nodes=10,
        learning_rate=1.0,
        epochs=100,
        batch_size=32,
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