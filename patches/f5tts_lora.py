"""Monkey-patch F5TTS to add load_lora(path).

Upstream f5-tts (1.1.20 on PyPI as of 2026-04-20) ships full fine-tune CLI only; there is no
LoRA path on PyPI and no merged LoRA PR on SWivid/F5-TTS. We train via PEFT in the Colab
notebook (see notebooks/coherence_dial_spike.ipynb Cell 5) and load the resulting adapter at
inference here.

PeftModel.from_pretrained accepts a local directory OR an HF model repo id, so this works
for both Colab-trained adapters and the published akshan-main/glossolalia-dial-lora.

Decision provenance: DECISIONS.md "F5-TTS LoRA path = DIY PEFT" + F5-TTS LoRA path workflow.
"""

from f5_tts.api import F5TTS


def _load_lora(self, path):
    """Wrap the DiT (cfm.transformer) with a PEFT adapter loaded from path-or-repo-id.

    Wrapping cfm.transformer (the DiT), NOT cfm itself — wrapping the whole CFM with
    PeftModel would clobber its forward(mel, text=, lens=) signature.
    """
    from peft import PeftModel
    target = self.ema_model.transformer
    self.ema_model.transformer = PeftModel.from_pretrained(target, path).to(self.device)
    self.ema_model.transformer.eval()
    self._lora_path = path
    return self


def install_load_lora():
    if not hasattr(F5TTS, "load_lora"):
        F5TTS.load_lora = _load_lora
