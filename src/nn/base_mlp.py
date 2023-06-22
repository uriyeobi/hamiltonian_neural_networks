"""Base MLP(Multi Layer Perceptrons)."""
from scipy.integrate import solve_ivp
from typing import Any
import tensorflow as tf
import numpy.typing as npt
from spec import TrainSpec
import numpy as np
import os


class BaseMLP(tf.keras.Model):
    """Base Multilayer Perceptrons."""

    def __init__(
        self, hidden_dims, input_dim, output_dim, hidden_activation="tanh", **kwargs
    ):
        super().__init__(**kwargs)

        self.feature_extractor = tf.keras.Sequential(
            [tf.keras.Input(shape=(input_dim,))]
            + [
                tf.keras.layers.Dense(hidden_dim, activation=hidden_activation)
                for hidden_dim in hidden_dims
            ]
        )
        self.last_layer = tf.keras.layers.Dense(output_dim, activation="linear")

    def call(self, x):
        features = self.feature_extractor(x)
        outputs = self.last_layer(features)
        return outputs


def init_seed() -> None:
    """Initialize seed."""

    os.environ["PYTHONHASHSEED"] = str(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.keras.utils.set_random_seed(42)


def train_base_mlp(
    x: npt.NDArray[Any],
    y: npt.NDArray[Any],
    train_spec: TrainSpec,
) -> BaseMLP:
    """Train base MLP."""
    init_seed()

    base_mlp = BaseMLP(
        hidden_dims=train_spec.hidden_dims,
        input_dim=x.shape[1],
        output_dim=x.shape[1],
    )
    base_mlp.compile(
        loss="mean_squared_error",
        optimizer=tf.keras.optimizers.Adam(train_spec.learning_rate),
    )
    batch_size = x.shape[0]
    base_mlp.fit(
        x=x,
        y=y,
        batch_size=batch_size,
        epochs=train_spec.epochs,
        verbose=0,
    )
    return base_mlp


def integrate_base_mlp(
    model: BaseMLP, t_span: npt.NDArray[Any], y0: npt.NDArray[Any], **kwargs
):
    """Integrate Base MLP."""

    def fun(t, np_x):
        np_x = np_x.reshape((1, len(y0)))
        dx = model.predict(np_x, verbose=0)
        return dx

    return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)
