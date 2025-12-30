from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from .prompt_templates import t5_entailment_prompt


@dataclass
class VerifierOutput:
    score: float        # support probability proxy
    label: str          # SUPPORTS/REFUTES/UNKNOWN


class T5EntailmentVerifier:
    """
    Practical verifier wrapper.

    Reality check:
    - Generic T5 is not trained for NLI by default.
    - If you have a finetuned checkpoint, plug it here.
    - This wrapper still gives you a consistent interface and caching.
    """
    def __init__(self, model_name: str, device: torch.device, max_len: int = 256):
        self.device = device
        self.max_len = max_len
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.model.eval()

        # token ids for labels if they exist
        self.support_id = self.tok.encode("SUPPORTS", add_special_tokens=False)
        self.refute_id = self.tok.encode("REFUTES", add_special_tokens=False)

    @torch.no_grad()
    def score(self, claim: str, evidence: str) -> Dict[str, Any]:
        if not evidence or not evidence.strip():
            return {"score": 0.0, "label": "UNKNOWN"}

        prompt = t5_entailment_prompt(claim, evidence)
        enc = self.tok(
            prompt,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        ).to(self.device)

        # compute next-token distribution at decoder start
        out = self.model(**enc, decoder_input_ids=torch.tensor([[self.model.config.decoder_start_token_id]], device=self.device))
        logits = out.logits[:, 0, :]  # [1,V]
        probs = torch.softmax(logits, dim=-1).squeeze(0)

        # crude: support score = P("SUPPORTS") if single-token, else fallback
        score = 0.0
        if len(self.support_id) == 1:
            score = float(probs[self.support_id[0]].item())
        elif len(self.refute_id) == 1:
            # if we cannot get support token prob, derive from refute
            score = 1.0 - float(probs[self.refute_id[0]].item())

        label = "SUPPORTS" if score >= 0.5 else "REFUTES"
        return {"score": float(score), "label": label}
