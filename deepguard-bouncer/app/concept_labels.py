"""
Deep-Guard -- the Content Registry's "namer": says what an area of an image
looks like ("face", "hands", "landscape"...), so a registry match can be
explained as a list of THINGS the two images share, not just a score.

The registry's own matcher (SSCD, see content_registry.py) describes an
image as 512 numbers, none of which individually means anything a person
could read -- the meaning is spread across all of them at once. Naming
content needs a model trained to connect pictures with words, which is
exactly what CLIP is (Radford et al., "Learning Transferable Visual Models
From Natural Language Supervision", ICML 2021): it maps images and short
texts into one shared space, so "which of these names sits closest to this
picture" is a zero-shot classifier over any vocabulary.

Only CLIP's IMAGE half runs here. tools/build_concept_labeler.py ran the
TEXT half once over a fixed, curated vocabulary and saved the results, so
the server needs nothing beyond torch (the same torch.jit.load path SSCD
already uses). Model: OpenCLIP ViT-B/32, DataComp-XL weights, MIT licence.

A name is a guess, never evidence: CLIP can be wrong, especially on small
or abstract areas. Callers show it as "looks like", next to the measured
numbers (matching points, share of the similarity) that actually support
a match. If the two model files are missing, load_concept_labeler()
returns None and the registry falls back to naming areas by position --
additive, like every other optional model in this app.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


@dataclass
class ConceptLabeler:
    encoder: torch.jit.ScriptModule
    thing_labels: list
    thing_embeddings: torch.Tensor   # [n_things, dim], unit-length rows
    medium_labels: list
    medium_embeddings: torch.Tensor  # [n_mediums, dim], unit-length rows
    logit_scale: float
    preprocess: transforms.Compose
    model_card: str
    vocab_hash: str

    def embed(self, images: list) -> torch.Tensor:
        """Unit-length CLIP image embeddings, one row per image."""
        batch = torch.stack([self.preprocess(im.convert("RGB")) for im in images])
        with torch.no_grad():
            out = self.encoder(batch).float()
        return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def _rank(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor,
              labels: list, top_k: int) -> list:
        probs = (self.logit_scale * image_embeddings @ text_embeddings.T).softmax(dim=-1)
        ranked = []
        for row in probs:
            values, indices = row.topk(min(top_k, len(labels)))
            ranked.append([(labels[i], round(float(v), 4)) for v, i in zip(values, indices)])
        return ranked

    def name_things(self, images: list, top_k: int = 3) -> list:
        """For each image, its top_k (name, probability) guesses from the
        'things' vocabulary. Probabilities are a softmax over the whole
        vocabulary, so near-synonyms (tree / trees / forest) share mass --
        read them as a ranking, not as calibrated confidence."""
        if not images:
            return []
        return self._rank(self.embed(images), self.thing_embeddings, self.thing_labels, top_k)

    def name_medium(self, image: Image.Image) -> tuple:
        """(medium, probability): photograph, oil painting, woodblock print..."""
        return self._rank(self.embed([image]), self.medium_embeddings, self.medium_labels, 1)[0][0]


def load_concept_labeler(encoder_path: Path, vocab_path: Path) -> Optional[ConceptLabeler]:
    if not (encoder_path.exists() and vocab_path.exists()):
        return None
    vocab = np.load(vocab_path, allow_pickle=False)
    encoder = torch.jit.load(str(encoder_path)).float().eval()
    size = int(vocab["image_size"])
    # Squash (not centre-crop) to the model's square input: an area of a
    # 3x3 grid is rarely square, and a centre crop would throw away the
    # very edges that make it that area rather than its neighbour.
    preprocess = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=vocab["mean"].tolist(), std=vocab["std"].tolist()),
    ])
    return ConceptLabeler(
        encoder=encoder,
        thing_labels=[str(x) for x in vocab["thing_labels"]],
        thing_embeddings=torch.from_numpy(vocab["thing_embeddings"].astype(np.float32)),
        medium_labels=[str(x) for x in vocab["medium_labels"]],
        medium_embeddings=torch.from_numpy(vocab["medium_embeddings"].astype(np.float32)),
        logit_scale=float(vocab["logit_scale"]),
        preprocess=preprocess,
        model_card=str(vocab["model_card"]),
        vocab_hash=str(vocab["vocab_hash"]),
    )
