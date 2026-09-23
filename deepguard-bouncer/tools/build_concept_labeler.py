"""
Deep-Guard -- one-time build of the Content Registry's "namer".

The registry explains a match area by area ("face", "hands", "landscape"...).
Naming what is in an area is a job for a vision-language model, so this
script prepares one, once, and writes two files into models/:

  concept_labeler_image_encoder.torchscript.pt
      CLIP ViT-B/32's IMAGE half only, as TorchScript with fp16 weights
      (~176 MB). The server loads it with plain torch.jit.load and casts
      it back to fp32 -- no open_clip needed at runtime, the same way the
      SSCD model is loaded.

  concept_labeler_vocabulary.npz
      The fixed list of names the namer can use, each already turned into
      a CLIP text embedding here (so the TEXT half of CLIP never has to be
      shipped or loaded), plus the exact preprocessing numbers the image
      half expects.

Model: OpenCLIP ViT-B-32 trained on DataComp-1B, weights
"datacomp_xl_s13b_b90k" (Hugging Face: laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K),
MIT licence, 72.7% ImageNet zero-shot top-1. The 605 MB checkpoint is
downloaded once into the normal Hugging Face cache (~/.cache/huggingface),
not into this repo. Both output files are gitignored: the encoder is too
big for GitHub, so every checkout runs this script once. Without these
files the registry still works and still explains matches -- it names
areas by position ("top-left area") instead of by content.

Prompt ensembling follows the CLIP paper (Radford et al., 2021): each name
is written into several sentence templates, the text embeddings are
averaged, then re-normalised.

Run from deepguard-bouncer/ (needs internet the first time):
    pip install -r app/requirements-dev.txt
    python tools/build_concept_labeler.py
"""

import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

MODEL_NAME = "ViT-B-32"
PRETRAINED = "datacomp_xl_s13b_b90k"
MODEL_CARD = "OpenCLIP ViT-B/32, DataComp-XL weights (MIT licence)"

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
ENCODER_PATH = MODELS_DIR / "concept_labeler_image_encoder.torchscript.pt"
VOCAB_PATH = MODELS_DIR / "concept_labeler_vocabulary.npz"
DEMO_IMAGE = Path(__file__).resolve().parents[1] / "demo_artworks" / "mona_lisa.jpg"

# "label|phrase": the label is what the page shows, the phrase is what goes
# into the sentence templates. A bare label is used as its own phrase.
# Curated for what a 3x3 area of a photo, painting or screenshot tends to
# contain: body parts, clothing, nature, buildings, objects, and plain
# backgrounds (so an empty area gets an honest "plain background" instead
# of being forced onto the nearest object).
#
# Tuned once on the 3 demo paintings + 8 test photos. Deliberately left
# out: age and gender words (a painted adult's face came out "child", and
# guessing either from appearance is not the registry's business), and
# toy-like words ("doll" fired on the Mona Lisa as a whole), and
# "watermark" (it fired on a painting's cracked varnish -- a false
# "watermark" is actively misleading on a copy-detection page; real text
# overlays still come out as "text").
THINGS = """
person|a person
group of people|a group of people
crowd|a crowd of people
face|a face
eyes|eyes
mouth|a mouth
nose|a nose
ear|an ear
hair|hair
beard|a beard
hand|a hand
hands|hands
fingers|fingers
arm|an arm
legs|legs
feet|feet
shoulder|a shoulder
neck|a neck
dress|a dress
shirt|a shirt
jacket|a jacket
coat|a coat
sweater|a sweater
sleeve|a sleeve
clothes|clothes
fabric|folds of fabric
hat|a hat
cap|a cap
helmet|a helmet
crown|a crown
veil|a veil
scarf|a scarf
glasses|glasses
sunglasses|sunglasses
necktie|a necktie
shoes|shoes
necklace|a necklace
jewellery|jewellery
bag|a bag
umbrella|an umbrella
sky|the sky
clouds|clouds
sun|the sun
moon|the moon
crescent moon|a crescent moon
stars|stars
night sky|a night sky
sunset|a sunset
rainbow|a rainbow
lightning|lightning
fog|fog
smoke|smoke
fire|fire
landscape|a landscape
mountain|a mountain
hills|hills
valley|a valley
cliff|a cliff
rocks|rocks
sand|sand
beach|a beach
desert|a desert
field|a field
grass|grass
road|a road
path|a path
bridge|a bridge
river|a river
lake|a lake
sea|the sea
wave|a wave
water|water
waterfall|a waterfall
snow|snow
ice|ice
island|an island
tree|a tree
trees|trees
forest|a forest
palm tree|a palm tree
cypress tree|a tall dark cypress tree
bush|a bush
leaves|leaves
branches|branches
flower|a flower
flowers|flowers
rose|a rose
sunflowers|sunflowers
garden|a garden
plant|a plant
fruit|fruit
vegetables|vegetables
potato|a potato
dog|a dog
cat|a cat
horse|a horse
bird|a bird
birds|birds
fish|a fish
cow|a cow
sheep|sheep
chicken|a chicken
duck|a duck
lion|a lion
tiger|a tiger
elephant|an elephant
bear|a bear
deer|a deer
monkey|a monkey
rabbit|a rabbit
snake|a snake
butterfly|a butterfly
insect|an insect
dinosaur|a dinosaur
dragon|a dragon
building|a building
buildings|buildings
house|a house
houses|houses
village|a village
town|a town
city|a city
skyline|a city skyline
street|a street
church|a church
church spire|a church spire
rooftops|rooftops
tower|a tower
castle|a castle
temple|a temple
mosque|a mosque
palace|a palace
ruins|ruins
window|a window
door|a door
wall|a wall
roof|a roof
fence|a fence
stairs|stairs
pillar|a pillar
arch|an arch
room|a room
kitchen|a kitchen
bedroom|a bedroom
office|an office
classroom|a classroom
shop|a shop
market|a market
stadium|a stadium
park|a park
playground|a playground
factory|a factory
harbour|a harbour
lighthouse|a lighthouse
windmill|a windmill
car|a car
bus|a bus
truck|a truck
bicycle|a bicycle
motorcycle|a motorcycle
train|a train
boat|a boat
boats|boats
ship|a ship
sailboat|a sailboat
airplane|an airplane
helicopter|a helicopter
rocket|a rocket
table|a table
desk|a desk
whiteboard|a whiteboard
chair|a chair
sofa|a sofa
bed|a bed
lamp|a lamp
candle|a candle
book|a book
books|books
bookshelf|a bookshelf
paper|a sheet of paper
newspaper|a newspaper
sign|a sign
screen|a screen
computer|a computer
laptop|a laptop
phone|a phone
keyboard|a keyboard
camera|a camera
clock|a clock
mirror|a mirror
picture frame|a picture frame
vase|a vase
cup|a cup
bottle|a bottle
glass|a drinking glass
plate|a plate
bowl|a bowl
food|food
cake|a cake
bread|bread
pizza|a pizza
basket|a basket
box|a box
ball|a ball
toy|a toy
guitar|a guitar
piano|a piano
violin|a violin
drum|a drum
flag|a flag
balloon|a balloon
sword|a sword
statue|a statue
sculpture|a sculpture
curtain|a curtain
carpet|a carpet
pillow|a pillow
blanket|a blanket
text|text
handwriting|handwriting
logo|a logo
signature|a signature
numbers|numbers
pattern|a pattern
stripes|stripes
checkerboard|a checkerboard pattern
swirls|swirling patterns
geometric shapes|geometric shapes
dots|dots
brushstrokes|brushstrokes
shadow|a shadow
reflection|a reflection
plain background|a plain background
dark background|a dark background
white background|a white background
blurry background|a blurry background
"""

THING_TEMPLATES = [
    "a photo of {}.",
    "a painting of {}.",
    "a close-up of {}.",
    "a picture showing {}.",
]

MEDIUMS = """
photograph|a photograph
black-and-white photograph|a black and white photograph
oil painting|an oil painting on canvas
watercolour|a watercolour painting
woodblock print|a Japanese ukiyo-e woodblock print
drawing|a pencil drawing
digital art|a piece of digital art
cartoon|a cartoon
illustration|an illustration
screenshot|a screenshot
3D render|a 3D render
"""

MEDIUM_TEMPLATES = [
    "{}.",
    "this image is {}.",
    "an example of {}.",
]


def parse(block: str) -> tuple[list[str], list[str]]:
    labels, phrases = [], []
    for line in block.strip().splitlines():
        label, _, phrase = line.strip().partition("|")
        labels.append(label.strip())
        phrases.append((phrase or label).strip())
    if len(set(labels)) != len(labels):
        dupes = sorted({l for l in labels if labels.count(l) > 1})
        raise SystemExit(f"Duplicate labels in vocabulary: {dupes}")
    return labels, phrases


def encode_names(model, tokenizer, phrases: list[str], templates: list[str]) -> np.ndarray:
    rows = []
    with torch.no_grad():
        for phrase in phrases:
            tokens = tokenizer([t.format(phrase) for t in templates])
            emb = model.encode_text(tokens).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
            mean = emb.mean(dim=0)
            rows.append((mean / mean.norm()).numpy())
    return np.stack(rows).astype(np.float32)


def main() -> None:
    try:
        import open_clip
    except ImportError:
        sys.exit("open_clip is not installed. Run: pip install -r app/requirements-dev.txt")

    t0 = time.perf_counter()
    print(f"Loading {MODEL_NAME} / {PRETRAINED} (downloads ~605 MB on first run)...")
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    cfg = open_clip.get_model_preprocess_cfg(model)
    image_size = cfg["size"] if isinstance(cfg["size"], int) else cfg["size"][0]
    print(f"  loaded in {time.perf_counter() - t0:.1f}s; preprocess cfg: {cfg}")

    thing_labels, thing_phrases = parse(THINGS)
    medium_labels, medium_phrases = parse(MEDIUMS)
    print(f"Encoding {len(thing_labels)} things x {len(THING_TEMPLATES)} templates "
          f"and {len(medium_labels)} mediums x {len(MEDIUM_TEMPLATES)} templates...")
    thing_emb = encode_names(model, tokenizer, thing_phrases, THING_TEMPLATES)
    medium_emb = encode_names(model, tokenizer, medium_phrases, MEDIUM_TEMPLATES)

    # ---- reference output from open_clip itself, before anything is converted
    from PIL import Image
    from torchvision import transforms
    mean, std = cfg["mean"], cfg["std"]
    prep = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    x = prep(Image.open(DEMO_IMAGE).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        ref = model.encode_image(x).float()

    # ---- export the image half only. Traced from a deep copy: a traced
    # module shares its parameter tensors with the original, so .half() on
    # it would silently convert open_clip's own model too.
    visual = copy.deepcopy(model.visual).eval()
    example = torch.randn(2, 3, image_size, image_size)
    with torch.no_grad():
        traced = torch.jit.trace(visual, example)
    traced = traced.half()
    MODELS_DIR.mkdir(exist_ok=True)
    torch.jit.save(traced, str(ENCODER_PATH))

    # ---- prove the round trip (fp16 on disk, fp32 at runtime) matches open_clip
    reloaded = torch.jit.load(str(ENCODER_PATH)).float().eval()
    with torch.no_grad():
        got = reloaded(x).float()
    cos = torch.nn.functional.cosine_similarity(ref, got).item()
    print(f"  export check: cosine(open_clip, exported fp16->fp32) = {cos:.5f}")
    if cos < 0.999:
        sys.exit("Exported encoder does not reproduce open_clip's output -- not saving the vocabulary.")

    vocab_hash = hashlib.sha256(
        json.dumps([thing_labels, thing_phrases, THING_TEMPLATES, medium_labels, medium_phrases,
                    MEDIUM_TEMPLATES, MODEL_NAME, PRETRAINED]).encode()
    ).hexdigest()[:16]
    np.savez_compressed(
        VOCAB_PATH,
        thing_labels=np.array(thing_labels),
        thing_embeddings=thing_emb.astype(np.float16),
        medium_labels=np.array(medium_labels),
        medium_embeddings=medium_emb.astype(np.float16),
        logit_scale=np.float32(model.logit_scale.exp().item()),
        image_size=np.int32(image_size),
        mean=np.array(mean, np.float32),
        std=np.array(std, np.float32),
        model_card=np.array(MODEL_CARD),
        vocab_hash=np.array(vocab_hash),
    )
    print(f"Wrote {ENCODER_PATH.name} ({ENCODER_PATH.stat().st_size / 1e6:.0f} MB) and "
          f"{VOCAB_PATH.name} ({VOCAB_PATH.stat().st_size / 1e3:.0f} KB), vocab {vocab_hash}, "
          f"total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
