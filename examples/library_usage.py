"""Use VeriFaith in-process (no server). Needs: pip install 'verifaith[nli]' and a .env file."""

from verifaith import Verifier

verifier = Verifier.from_settings()  # reads VERIFAITH_* env vars / .env

contexts = [
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. "
    "It was constructed from 1887 to 1889. It stands 330 metres tall."
]
answer = "The Eiffel Tower was completed in 1889 and is 500 metres tall."

result = verifier.evaluate(answer, contexts)
print(result.verdict, result.faithfulness)
for v in result.claims:
    print(f"[{v.label.value:>12}] {v.claim.text}  <-  {v.evidence.text if v.evidence else '-'}")
