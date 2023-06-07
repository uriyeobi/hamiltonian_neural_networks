"""Single Pendulum."""

from base import BasePendulum, BaseAnimator
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List


class SinglePendulum(BasePendulum):
    """Single Pendulum."""

    def energy_kinetic(self, theta_dot) -> float:
        """Kinetic energy."""
        return 0.5 * self.m1 * self.L1**2 * theta_dot**2

    def energy_potential(self, theta) -> float:
        """Potential energy."""
        return self.m1 * self.L1 * self.g * (1.0 - np.cos(theta))

    def equation_motion(self, u, t) -> List[float]:
        """Equations of motion."""
        [theta, theta_dot] = u
        return [theta_dot, -self.g / self.L1 * np.sin(theta)]

    def gen_sol_df(self) -> pd.DataFrame:
        """Generate solution dataframe."""

        sol = self.solve_ode()
        df = (
            pd.DataFrame(sol, columns=["theta", "theta_dot"])
            .assign(x=lambda df_: np.sin(df_["theta"]) * self.L1)
            .assign(y=lambda df_: -np.cos(df_["theta"] * self.L1))
            .assign(time_step=self.time_coord.t_grid)
            .assign(energy_kinetic=lambda df_: self.energy_kinetic(df_["theta_dot"]))
            .assign(energy_potential=lambda df_: self.energy_potential(df_["theta"]))
            .assign(
                energy_total=lambda df_: df_["energy_kinetic"] + df_["energy_potential"]
            )
        )
        return df


cmap = sns.color_palette("tab20")


class SinglePendulumAnimator(BaseAnimator):
    """Single Pendulum Animator."""

    def init_canvas(self) -> None:
        """Initial canvas."""
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
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
        self.ax.set_title("Single Pendulum")
        self.fig.tight_layout()

    def animate(self, i) -> None:
        """Animate."""
        self.line1.set_data([self.df.x[i], 0], [self.df.y[i], 0])
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
        self.line_orig.set_data([], [])
        self.time_str.set_text("")
        self.potential_energy_str.set_text("")
        self.kinetic_energy_str.set_text("")
        self.total_energy_str.set_text("")
