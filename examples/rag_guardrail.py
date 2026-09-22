"""Pattern: block or flag a RAG answer before it reaches the user, using the hosted service."""

from verifaith.client import VeriFaithClient, VeriFaithError

vf = VeriFaithClient("http://localhost:8000", api_key="dev-key")


def guarded_answer(question: str, retrieved_docs: list[str], generate) -> str:
    answer = generate(question, retrieved_docs)
    try:
        result = vf.evaluate(answer, retrieved_docs)
    except VeriFaithError:
        return answer  # fail open (or fail closed, your choice) if the checker is unavailable
    if result.counts["contradicted"]:
        bad = [v.claim.text for v in result.claims if v.label == "contradicted"]
        return "I couldn't verify parts of this answer against the sources: " + "; ".join(bad)
    if result.counts["conflicting"]:
        disputed = [v.claim.text for v in result.claims if v.conflicting_evidence]
        return answer + "\n\n(The sources disagree about: " + "; ".join(disputed) + ")"
    if result.verdict != "faithful":
        return answer + "\n\n(Some statements could not be verified against the sources.)"
    return answer
