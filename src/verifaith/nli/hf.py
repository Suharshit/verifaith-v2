"""Hugging Face sequence-classification NLI backend (e.g. DeBERTa NLI cross-encoders)."""

from __future__ import annotations

from verifaith.schemas import NLIScores

_LABELS = {"entailment", "neutral", "contradiction"}


class HFEntailmentModel:
    def __init__(self, model_name: str, batch_size: int = 16, device: str | None = None):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise ImportError("Install the NLI extra: pip install 'verifaith[nli]'") from e

        self._torch = torch
        self.name = model_name
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Read label order from the model config instead of guessing LABEL_0/1/2.
        id2label = {int(i): str(lbl).lower() for i, lbl in self.model.config.id2label.items()}
        self.index = {lbl: i for i, lbl in id2label.items() if lbl in _LABELS}
        if set(self.index) != _LABELS:
            raise ValueError(
                f"{model_name} labels {id2label} are not entailment/neutral/contradiction; "
                "use a 3-way NLI model or implement a custom EntailmentModel."
            )

    def predict(self, pairs: list[tuple[str, str]]) -> list[NLIScores]:
        out: list[NLIScores] = []
        torch = self._torch
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            enc = self.tokenizer(
                [p for p, _ in batch],  # premise = evidence
                [h for _, h in batch],  # hypothesis = claim (a real sentence pair, not "[SEP]")
                padding=True,
                truncation="only_first",
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(**enc).logits, dim=-1).cpu().tolist()
            for row in probs:
                out.append(
                    NLIScores(
                        entailment=row[self.index["entailment"]],
                        neutral=row[self.index["neutral"]],
                        contradiction=row[self.index["contradiction"]],
                    )
                )
        return out
