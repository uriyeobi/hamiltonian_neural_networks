"""Hamiltonian Neural Networks."""
from scipy.integrate import solve_ivp
from typing import Any, List, Union
import tensorflow as tf
import numpy.typing as npt
from functools import cached_property
import numpy as np
from spec import TrainSpec
import os

TypeVec = Union[npt.NDArray, tf.Tensor]


class HNN(tf.keras.Model):
    """Hamiltonian Neural Networks."""

    def __init__(self, input_dim, hidden_dims, **kwargs):
        super().__init__(**kwargs)
        self.feature_extractor = tf.keras.Sequential(
            [tf.keras.Input(shape=(input_dim,))]
            + [
                tf.keras.layers.Dense(hidden_dim, activation="tanh")
                for hidden_dim in hidden_dims
            ]
        )

        self.last_layer = tf.keras.layers.Dense(1)
        self.input_dim = input_dim

    @cached_property
    def M(self) -> npt.NDArray[Any]:
        """Permutation matrix (assuming canonical coordinates)."""
        M = np.eye(self.input_dim)
        M = np.concatenate(
            (M[self.input_dim // 2 :], -M[: self.input_dim // 2]), axis=0
        )
        return tf.constant(M, dtype="double")

    def call(self, x):
        """Call."""
        features = self.feature_extractor(x)
        outputs = self.last_layer(features)
        return outputs

    def forward(self, x):
        """Forward."""
        with tf.GradientTape() as tape:
            features = self.feature_extractor(x)
            outputs = self.last_layer(features)
        return (tape.gradient(outputs, x)) @ self.M


def calculate_single_pendulum_hamiltonian(mlg: List[float], x: TypeVec):
    """Calculate single pendulum hamiltonian."""
    [m, L, g] = mlg
    [q, p] = x
    return -m * g * L * np.cos(q) + p**2 / (2 * m * L**2)


def estimate_single_pendulum_hamiltonian(mlg: List[float], x: TypeVec, dx: TypeVec):
    """Estimate single pendulum hamiltonian."""
    [m, L, _] = mlg
    [q, _] = x
    [dqdt, _] = dx

    p_est = dqdt * m * L**2

    q_est = q

    return calculate_single_pendulum_hamiltonian(mlg, [q_est, p_est])


def estimate_double_pendulum_hamiltonian(
    mlg: List[float], x: TypeVec, dx: TypeVec
) -> float:
    """Estimate double pendulum hamiltonian."""
    [m1, L1, m2, L2, _] = mlg
    [q1, q2, _, _] = x
    [dq1dt, dq2dt, _, _] = dx

    p1_est = (m1 + m2) * L1**2 * dq1dt + m2 * L1 * L2 * dq2dt * np.cos(q1 - q2)
    p2_est = m2 * L2**2 * dq2dt + m2 * L1 * L2 * dq1dt * np.cos(q1 - q2)

    q1_est = q1
    q2_est = q2

    return calculate_double_pendulum_hamiltonian(mlg, [q1_est, q2_est, p1_est, p2_est])


def calculate_double_pendulum_hamiltonian(mlg: List[float], x: TypeVec) -> float:
    """Calculate double pendulum hamiltonian."""
    [m1, L1, m2, L2, g] = mlg
    [q1, q2, p1, p2] = x
    return (
        (
            m2 * L2**2 * p1**2
            + (m1 + m2) * L1**2 * p2**2
            - 2 * m2 * L1 * L2 * p1 * p2 * np.cos(q1 - q2)
        )
        / (2 * m2 * L1**2 * L2**2 * (m1 + m2 * (np.sin(q1 - q2)) ** 2))
        - (m1 + m2) * g * L1 * np.cos(q1)
        - m2 * g * L2 * np.cos(q2)
    )


def get_hamiltonian(
    x: TypeVec, mlg: List[float], hamiltonian_method: str, predictions=None
) -> float:
    """Get Hamiltonian."""
    if len(mlg) == 3:
        if hamiltonian_method == "curr":
            return calculate_single_pendulum_hamiltonian(mlg, x.T)
        elif hamiltonian_method == "next":
            return estimate_single_pendulum_hamiltonian(
                mlg, x.T, tf.transpose(predictions)
            )
    elif len(mlg) == 5:
        if hamiltonian_method == "curr":
            return calculate_double_pendulum_hamiltonian(
                mlg,
                x.T,
            )
        elif hamiltonian_method == "next":
            return estimate_double_pendulum_hamiltonian(
                mlg, x.T, tf.transpose(predictions)
            )


def get_loss(
    model: HNN,
    x: TypeVec,
    y: TypeVec,
    ham0: float,
    mlg: List[float],
    hamiltonian_method: str,
    penalty_lamb: float,
):
    """Get loss."""
    predictions = model.forward(tf.Variable(tf.stack(x)))
    ham_new = get_hamiltonian(
        x=x, mlg=mlg, hamiltonian_method=hamiltonian_method, predictions=predictions
    )

    physics_embedded_penalty = tf.reduce_mean(tf.square(ham0 - ham_new))

    return (
        tf.reduce_mean(tf.square(predictions - tf.Variable(tf.stack(y))))
        + penalty_lamb * physics_embedded_penalty
    )


def get_grad(
    model: HNN,
    optimizer: Any,
    x: TypeVec,
    y: TypeVec,
    ham0: float,
    mlg: List[float],
    hamiltonian_method: str,
    penalty_lamb: float,
):
    """Get gradient for each step for HNN."""
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        loss = get_loss(model, x, y, ham0, mlg, hamiltonian_method, penalty_lamb)
    gradients = tape.gradient(
        loss,
        model.trainable_variables,
        unconnected_gradients=tf.UnconnectedGradients.ZERO,
    )
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, optimizer


def integrate_hnn(model: HNN, t_span: npt.NDArray[Any], y0: npt.NDArray[Any], **kwargs):
    """Integrate HNN."""

    def fun(t, np_x):
        np_x = tf.Variable(tf.reshape(np_x, (1, len(y0))), dtype="double")
        dx = model.forward(np_x)
        return dx

    return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)


def init_seed() -> None:
    """Initialize seed."""
    os.environ["PYTHONHASHSEED"] = str(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.keras.utils.set_random_seed(42)


def train_hnn(
    x: npt.NDArray[Any],
    y: npt.NDArray[Any],
    train_spec: TrainSpec,
) -> HNN:
    """Train HNN."""

    init_seed()

    ham0 = get_hamiltonian(x[0], mlg=train_spec.mlg, hamiltonian_method="curr")
    print(f"{ham0=}")

    model = HNN(input_dim=x.shape[1], hidden_dims=train_spec.hidden_dims)
    optimizer = tf.keras.optimizers.Adam(learning_rate=train_spec.learning_rate)

    for itr in range(1, train_spec.epochs + 1):
        loss, optimizer = get_grad(
            model=model,
            optimizer=optimizer,
            x=x,
            y=y,
            ham0=ham0,
            mlg=train_spec.mlg,
            hamiltonian_method=train_spec.hamiltonian_method,
            penalty_lamb=train_spec.penalty_lamb,
        )
        if itr % 100 == 0:
            print(f"{itr=}, loss={loss.numpy()}")
    return model
