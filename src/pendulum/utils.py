"""Utilities for Pendulum."""

from single import SinglePendulum, SinglePendulumAnimator
from double import DoublePendulum, DoublePendulumAnimator
import pandas as pd
from pathlib import Path
import os
import numpy as np


def generate_data(config_dir: Path) -> pd.DataFrame:
    """Generating time-series data of pendulum dynamics."""
    p1 = SinglePendulum().from_yaml(str(config_dir / "single.yaml"))
    p2 = DoublePendulum().from_yaml(str(config_dir / "double.yaml"))
    df1 = p1.gen_sol_df()
    df1 = p1.create_generalized_coord_momenta(df1)
    df2 = p2.gen_sol_df()
    df2 = p2.create_generalized_coord_momenta(df2)

    df = df1.merge(df2, how="inner", on="time_step", suffixes=["_single", "_double"])

    return df
