from pydantic import BaseModel
from typing import Optional, List


class TrainSpec(BaseModel):
    mlg: Optional[List[float]]
    hamiltonian_method: Optional[str]
    hidden_dims: List[int] = [100, 100]
    learning_rate: float = 0.002
    epochs: int = 1000
    penalty_lamb: Optional[float] = 0
