import importlib
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
except ImportError:  # Local fallback for environments without torch.
    torch = None  # type: ignore[assignment]

    class _FallbackModule:
        def eval(self):
            return self

        def load_state_dict(self, state_dict, strict: bool = True):
            return None

    class _FallbackNN:
        Module = _FallbackModule

    nn = _FallbackNN()  # type: ignore[assignment]


class Model(nn.Module):
    """
    Template model for the leaderboard.

    Requirements:
    - Must be instantiable with no arguments (called by the evaluator).
    - Must implement `predict(batch)` which receives an iterable of inputs and
      returns a list of predictions (labels).
    - Must implement `eval()` to place the model in evaluation mode.
    - If you use PyTorch, submit a state_dict to be loaded via `load_state_dict`
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vocab_index: Dict[str, int] = {}
        self.log_prob_fox: List[float] = []
        self.log_prob_nbc: List[float] = []
        self.log_prior_fox: float = 0.5
        self.unk_log_prob_fox: float = -12.0
        self.unk_log_prob_nbc: float = -12.0

        self._maybe_autoload_weights()

    def eval(self) -> "Model":
        super().eval()
        return self

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Allow optional lightweight metadata while staying compatible with
        evaluator behavior that may call load_state_dict(torch.load(...)).
        """
        if isinstance(state_dict, dict):
            vocab = state_dict.get("vocab")
            log_prob_fox = state_dict.get("log_prob_fox")
            log_prob_nbc = state_dict.get("log_prob_nbc")
            log_prior_fox = state_dict.get("log_prior_fox")
            unk_log_prob_fox = state_dict.get("unk_log_prob_fox")
            unk_log_prob_nbc = state_dict.get("unk_log_prob_nbc")

            if (
                isinstance(vocab, list)
                and log_prob_fox is not None
                and log_prob_nbc is not None
                and log_prior_fox is not None
            ):
                if torch is not None:
                    if hasattr(log_prob_fox, "tolist"):
                        log_prob_fox = log_prob_fox.tolist()
                    if hasattr(log_prob_nbc, "tolist"):
                        log_prob_nbc = log_prob_nbc.tolist()
                    if hasattr(log_prior_fox, "item"):
                        log_prior_fox = log_prior_fox.item()
                    if hasattr(unk_log_prob_fox, "item"):
                        unk_log_prob_fox = unk_log_prob_fox.item()
                    if hasattr(unk_log_prob_nbc, "item"):
                        unk_log_prob_nbc = unk_log_prob_nbc.item()

                self.vocab_index = {str(tok): i for i, tok in enumerate(vocab)}
                self.log_prob_fox = [float(x) for x in log_prob_fox]
                self.log_prob_nbc = [float(x) for x in log_prob_nbc]
                self.log_prior_fox = float(log_prior_fox)
                if unk_log_prob_fox is not None:
                    self.unk_log_prob_fox = float(unk_log_prob_fox)
                if unk_log_prob_nbc is not None:
                    self.unk_log_prob_nbc = float(unk_log_prob_nbc)

                if torch is not None:
                    return torch.nn.modules.module._IncompatibleKeys([], [])
                return None

        return super().load_state_dict(state_dict, strict=strict)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z']+", (text or "").lower())

    def _maybe_autoload_weights(self) -> None:
        if torch is None:
            return

        model_path = Path(__file__).resolve().with_name("model.pt")
        if not model_path.exists():
            return

        loaded = torch.load(model_path, map_location="cpu")
        self.load_state_dict(loaded, strict=False)

    def _predict_one(self, text: Any) -> str:
        if not self.vocab_index:
            return "FoxNews"

        log_fox = math.log(max(self.log_prior_fox, 1e-9))
        log_nbc = math.log(max(1.0 - self.log_prior_fox, 1e-9))

        for token in self._tokenize(str(text)):
            idx = self.vocab_index.get(token)
            if idx is None:
                log_fox += self.unk_log_prob_fox
                log_nbc += self.unk_log_prob_nbc
            else:
                log_fox += self.log_prob_fox[idx]
                log_nbc += self.log_prob_nbc[idx]

        return "FoxNews" if log_fox >= log_nbc else "NBC"

    def predict(self, batch: Iterable[Any]) -> List[Any]:
        """
        Implement your inference here.
        Inputs:
            batch: Iterable of preprocessed inputs (as produced by your preprocess.py)
        Returns:
            A list of predictions with the same length as `batch`.
        """
        return [self._predict_one(item) for item in batch]


def get_model() -> Model:
    """
    Factory function required by the evaluator.
    Returns an uninitialized model instance. The evaluator may optionally load
    weights (if provided) before calling predict(...).
    """
    return Model()