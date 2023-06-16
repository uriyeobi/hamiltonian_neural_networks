"""Double Pendulum."""

from base import BasePendulum, BaseAnimator
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional


class DoublePendulum(BasePendulum):
    """Double Pendulum."""

    m2: Optional[float]
    L2: Optional[float]

    def energy_kinetic(self, theta1, theta1_dot, theta2, theta2_dot) -> float:
        """Kinetic energy."""
        return 0.5 * self.m1 * self.L1**2 * theta1_dot**2 + 0.5 * self.m2 * (
            self.L1**2 * theta1_dot**2
            + self.L2**2 * theta2_dot**2
            + 2 * self.L1 * self.L2 * theta1_dot * theta2_dot * np.cos(theta1 - theta2)
        )

    def energy_potential(self, theta1, theta2) -> float:
        """Potential energy."""
        return (
            self.m1 * (self.L1 * (1.0 - np.cos(theta1)) + self.L2)
            + self.m2
            * (self.L1 * (1.0 - np.cos(theta1)) + self.L2 * (1.0 - np.cos(theta2)))
        ) * self.g

    def equation_motion(self, u, t) -> List[float]:
        """Equations of motion."""

        c = np.cos(u[0] - u[2])
        s = np.sin(u[0] - u[2])

        return [
            u[1],
            (
                self.m2 * self.g * np.sin(u[2]) * c
                - self.m2 * s * (self.L1 * c * u[1] ** 2 + self.L2 * u[3] ** 2)
                - (self.m1 + self.m2) * self.g * np.sin(u[0])
            )
            / (self.L1 * (self.m1 + self.m2 * s**2)),
            u[3],
            (
                (self.m1 + self.m2)
                * (
                    self.L1 * u[1] ** 2 * s
                    - self.g * np.sin(u[2])
                    + self.g * np.sin(u[0]) * c
                )
                + self.m2 * self.L2 * u[3] ** 2 * s * c
            )
            / (self.L2 * (self.m1 + self.m2 * s**2)),
        ]

    def gen_sol_df(self) -> pd.DataFrame:
        """Generate solution dataframe."""

        sol = self.solve_ode()
        df = (
            pd.DataFrame(sol, columns=["theta1", "theta1_dot", "theta2", "theta2_dot"])
            .assign(x1=lambda df_: np.sin(df_["theta1"]) * self.L1)
            .assign(y1=lambda df_: -np.cos(df_["theta1"]) * self.L1)
            .assign(x2=lambda df_: df_.x1 + np.sin(df_["theta2"]) * self.L2)
            .assign(y2=lambda df_: df_.y1 - np.cos(df_["theta2"]) * self.L2)
            .assign(time_step=self.time_coord.t_grid)
            .assign(
                energy_kinetic=lambda df_: self.energy_kinetic(
                    df_["theta1"], df_["theta1_dot"], df_["theta2"], df_["theta2_dot"]
                )
            )
            .assign(
                energy_potential=lambda df_: self.energy_potential(
                    df_["theta1"], df_["theta2"]
                )
            )
            .assign(
                energy_total=lambda df_: df_["energy_kinetic"] + df_["energy_potential"]
            )
        )
        return df

    def create_generalized_coord_momenta(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create generalized coordinates and generalized momenta."""
        m1 = self.m1
        m2 = self.m2
        L1 = self.L1
        L2 = self.L2
        g = self.g
        theta1 = df["theta1"]
        theta2 = df["theta2"]
        theta1_dot = df["theta1_dot"]
        theta2_dot = df["theta2_dot"]
        c = np.cos(theta1 - theta2)
        s = np.sin(theta1 - theta2)
        df = (
            df.assign(q1=theta1)
            .assign(q2=theta2)
            .assign(p1=(m1 + m2) * L1**2 * theta1_dot + m2 * L1 * L2 * theta2_dot * c)
            .assign(p2=m2 * L2**2 * theta2_dot + m2 * L1 * L2 * theta1_dot * c)
        )

        p1 = df["p1"]
        p2 = df["p2"]
        h1 = p1 * p2 * s / (L1 * L2 * (m1 + m2 * s**2))
        h2 = (
            m2 * L2**2 * p1**2
            + (m1 + m2) * L1**2 * p2**2
            - 2 * m2 * L1 * L2 * p1 * p2 * c
        ) / (2 * (L1 * L2 * (m1 + m2 * s**2)) ** 2)
        df = (
            df.assign(
                dq1dt=(L2 * p1 - L1 * p2 * c) / (L1**2 * L2 * (m1 + m2 * s**2))
            )
            .assign(
                dq2dt=(-m2 * L2 * p1 * c + (m1 + m2) * L1 * p2)
                / (m2 * L1 * L2**2 * (m1 + m2 * s**2))
            )
            .assign(
                dp1dt=-(m1 + m2) * g * L1 * np.sin(theta1)
                - h1
                + h2 * np.sin(2 * (theta1 - theta2))
            )
            .assign(
                dp2dt=-m2 * g * L2 * np.sin(theta2)
                + h1
                - h2 * np.sin(2 * (theta1 - theta2))
            )
        )
        return df


cmap = sns.color_palette("tab20")


class DoublePendulumAnimator(BaseAnimator):
    """Double Pendulum Animator."""

    def init_canvas(self) -> None:
        """Initial canvas."""
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        (self.line2,) = self.ax.plot(
            [],
            [],
            "-o",
            color="k",
            markersize=12,
            lw=2,
            markerfacecolor=cmap[0],
            markeredgecolor="k",
        )
        (self.line1,) = self.ax.plot(
            [],
            [],
            "-o",
            color="k",
            markersize=12,
            lw=2,
            markerfacecolor=cmap[2],
            markeredgecolor="k",
        )
        (self.line_orig,) = self.ax.plot(
            [],
            [],
            "o",
            color=cmap[4],
            markersize=12,
            markerfacecolor=cmap[4],
            markeredgecolor="k",
        )
        self.time_str = self.ax.text(0.05, 0.9, "", transform=self.ax.transAxes)
        self.kinetic_energy_str = self.ax.text(
            0.05, 0.85, "", transform=self.ax.transAxes
        )
        self.potential_energy_str = self.ax.text(
            0.05, 0.8, "", transform=self.ax.transAxes
        )
        self.total_energy_str = self.ax.text(
            0.05, 0.75, "", transform=self.ax.transAxes
        )
        xlim = ylim = self.size * 1.5
        self.ax.set_ylim([-ylim, ylim])
        self.ax.set_xlim([-xlim, xlim])
        self.ax.set_title("Double Pendulum")
        self.fig.tight_layout()

    def animate(self, i) -> None:
        """Animate."""
        self.line2.set_data(
            [self.df.x1[i], self.df.x2[i]], [self.df.y1[i], self.df.y2[i]]
        )
        self.line1.set_data([self.df.x1[i], 0], [self.df.y1[i], 0])

        self.line_orig.set_data([0, 0], [0, 0])
        self.time_str.set_text("Time: %.1f s" % (self.df.time_step[i]))
        self.potential_energy_str.set_text(
            "Potential Energy: %.2f" % self.df.energy_potential[i]
        )
        self.kinetic_energy_str.set_text(
            "Kinetic Energy: %.2f" % self.df.energy_kinetic[i]
        )
        self.total_energy_str.set_text("Total Energy: %.2f" % self.df.energy_total[i])

    def init_func(self) -> None:
        """Initial animation."""
        self.line1.set_data([], [])
        self.line2.set_data([], [])
        self.line_orig.set_data([], [])
        self.time_str.set_text("")
        self.potential_energy_str.set_text("")
        self.kinetic_energy_str.set_text("")
        self.total_energy_str.set_text("")
