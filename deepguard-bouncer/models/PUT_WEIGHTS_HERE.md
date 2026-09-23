Three model files live in this folder. All three are already included —
you don't need to train or download anything to run the server.

| File | What it's for | If missing |
|---|---|---|
| `deepguard_bouncer.pth` | Stage 1 — the base real/AI detector. Every analysis needs this. | Server refuses to start cleanly; `/api/health` reports `model_loaded: false`. |
| `deepguard_attribution.pth` | Phase 2, Idea 2 — names *which* generator likely made a flagged image. | Server still runs fine; the "Likely source" panel and `generator_attribution` field are just never present. |
| `sscd_disc_mixup.torchscript.pt` | Phase 2, Idea 4 — the pretrained embedding model behind the Content Registry's crop/edit-robust matching. | Server still runs fine; `/api/registry/*` reports 503 (registry unavailable). |

Each one is trained/obtained completely independently, and each is
additive — the app degrades gracefully without any of the two optional
ones, never crashes. See `main.py`'s module docstring for exactly which
endpoint each one feeds.

To retrain `deepguard_bouncer.pth` or `deepguard_attribution.pth`, follow
the matching notebook in `colab_notebook/` and overwrite the file here
with the download. `sscd_disc_mixup.torchscript.pt` is not trained by
this project at all — it's Meta AI's own pretrained release (see the
README's Content Registry section for the source).

## Optional: the registry's "namer" (built locally, never committed)

| File | What it's for | If missing |
|---|---|---|
| `concept_labeler_image_encoder.torchscript.pt` (~176 MB) + `concept_labeler_vocabulary.npz` | Names what each area of a registered image looks like ("face", "landscape"), so registry explanations can list the *things* two images share. CLIP ViT-B/32's image half (OpenCLIP, DataComp-XL weights, MIT licence) plus a pre-encoded vocabulary. | Registry explanations still work in full — areas are named by position ("top left area") instead of by content. |

Too big for GitHub's 100 MB file limit, so both are gitignored. Build
them once per checkout (downloads 605 MB from Hugging Face the first
time, into the normal Hugging Face cache, not this repo):

```bash
pip install -r app/requirements-dev.txt
python tools/build_concept_labeler.py      # run from deepguard-bouncer/
```

## Written here by the server (gitignored, rebuilt automatically)

`content_registry.index`, `content_registry_metadata.json` and the
`content_registry_features/` folder are the registry's own state: the
FAISS index, the signed records, and each entry's distinctive points
(no pixels). They are seeded from `demo_artworks/` on first start;
delete all three to reset the registry.

See `README.md` for the exact setup commands.
