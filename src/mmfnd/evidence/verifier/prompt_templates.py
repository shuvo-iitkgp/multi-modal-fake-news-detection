def t5_entailment_prompt(claim: str, evidence: str) -> str:
    # Keep it simple and deterministic
    claim = (claim or "").strip()
    evidence = (evidence or "").strip()
    return (
        "task: fact_verification\n"
        f"claim: {claim}\n"
        f"evidence: {evidence}\n"
        "label:"
    )
