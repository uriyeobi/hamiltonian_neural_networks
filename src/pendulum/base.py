import numpy as np
import pandas as pd
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
from pydantic import BaseModel, BaseConfig
from typing import List, Any, Optional
from dataclasses import dataclass
import yaml
import pathlib
from abc import abstractmethod


@dataclass
class TimeCoordinate:
    """Time Coordinates."""

    T: float  # time span
    N_t: int  # number of grid for time span

    @property
    def t_grid(self) -> npt.NDArray[Any]:
        """Time grid."""
        return np.linspace(0, self.T, self.N_t)

    @property
    def dt(self) -> float:
        """delta t."""
        return self.t_grid[1] - self.t_grid[0]


@dataclass
class InitialCondition:
    """Initial Condition"""

    theta_vec: Optional[List[float]]

    @property
    def u0(self) -> List[float]:
        """Initial values of angle and angular velocity."""
        for i in range(len(self.theta_vec)):
            if i % 2 == 0:
                self.theta_vec[i] *= np.pi / 180
        return self.theta_vec


class BasePendulum(BaseModel):
    """Base Pendulum."""

    init_cond: Optional[InitialCondition]
    time_coord: Optional[TimeCoordinate]
    m1: Optional[float]
    L1: Optional[float]
    g: Optional[float]

    @classmethod
    def from_yaml(
        cls,
        path: str,
    ) -> "BasePendulum":
        cfg = yaml.safe_load(pathlib.Path(path).read_text())
        obj = cls.parse_obj(cfg)
        return obj

    @abstractmethod
    def energy_kinetic(self, theta1, theta1_dot, theta2, theta2_dot) -> float:
        """Kinetic energy."""
        raise NotImplementedError

    @abstractmethod
    def energy_potential(self, theta1, theta2) -> float:
        """Potential energy."""
        raise NotImplementedError

    @abstractmethod
    def equation_motion(self, u, t) -> List[float]:
        """Equations of motion."""
        raise NotImplementedError

    def solve_ode(self) -> npt.NDArray[Any]:
        """Solve ODE."""
        sol = odeint(
            func=self.equation_motion,
            y0=self.init_cond.u0,
            t=self.time_coord.t_grid,
        )
        return sol

    @abstractmethod
    def gen_sol_df(self) -> pd.DataFrame:
        """Generate solution dataframe."""
        raise NotImplementedError

    @abstractmethod
    def create_generalized_coord_momenta(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create generalized coordinates and generalized momenta."""
        raise NotImplementedError


class BaseAnimator(BaseModel):
    """Base Animator."""

    df: pd.DataFrame
    size: float
    fig: Optional[Any]
    ax: Optional[Any]
    line1: Optional[Any]
    line2: Optional[Any]
    line_orig: Optional[Any]
    time_str: Optional[str]
    kinetic_energy_str: Optional[str]
    potential_energy_str: Optional[str]
    total_energy_str: Optional[str]

    class Config(BaseConfig):
        """Config for pydantic model."""

        arbitrary_types_allowed: bool = True

    @abstractmethod
    def init_canvas(self) -> None:
        plt.ioff()
        """Initial canvas."""
        raise NotImplementedError

    @abstractmethod
    def animate(self, i) -> None:
        """Animate."""
        raise NotImplementedError

    @abstractmethod
    def init_func(self) -> None:
        """Initial animation."""
        raise NotImplementedError

    def run(self, frames: int, interval: float) -> FuncAnimation:
        """Run animation."""
        return FuncAnimation(
            self.fig,
            self.animate,
            init_func=self.init_func,
            frames=frames,
            interval=interval,
        )

    def save(self, anim: FuncAnimation, fps: int, gif_file: str) -> None:
        """Save animation."""
        anim.save(gif_file, fps=fps, writer="imagemagick")
