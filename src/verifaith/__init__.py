"""VeriFaith: claim-level faithfulness verification for RAG answers."""

from verifaith.pipeline import Verifier
from verifaith.schemas import ClaimVerdict, EvalResult, Label

__version__ = "0.1.0"
__all__ = ["ClaimVerdict", "EvalResult", "Label", "Verifier", "__version__"]
