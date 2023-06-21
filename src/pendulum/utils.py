"""Utilities for Pendulum."""

from single import SinglePendulum
from double import DoublePendulum
from pathlib import Path
from typing import List, Any


def generate_data(config_dir: Path) -> List[Any]:
    """Generating time-series data of pendulum dynamics."""
    p1 = SinglePendulum().from_yaml(str(config_dir / "single.yaml"))
    p2 = DoublePendulum().from_yaml(str(config_dir / "double.yaml"))
    df1 = p1.gen_sol_df()
    df1 = p1.create_generalized_coord_momenta(df1)
    df2 = p2.gen_sol_df()
    df2 = p2.create_generalized_coord_momenta(df2)
    df = df1.merge(df2, how="outer", on="time_step", suffixes=["_single", "_double"])
    mlg = {"single": [p1.m1, p1.L1, p1.g], "double": [p2.m1, p2.L1, p2.m2, p2.L2, p2.g]}

    return [df, mlg]
