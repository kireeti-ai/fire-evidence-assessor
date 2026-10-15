"""Generate synthetic fire/smoke/hard-negative images from open-weight generators (GPU, run on the VM).

Families (licence read from the HF model card, non-gated): sd15 (CreativeML OpenRAIL-M), sdxl (OpenRAIL++),
pixart (OpenRAIL++), kolors (Apache-2.0). Outputs are research data for detector evaluation; keep internal.
All images are normalised to longest side 512 + JPEG q90 so format/size cannot be a shortcut.
"""
import argparse, csv, itertools, json, os, random, sys, time, traceback, zlib
import torch
from PIL import Image

SCENES = ["a hillside with dry grass", "a pine forest", "a suburban street", "a rural highway", "a farm field",
          "a canyon", "a lakeside campsite", "a mountain ridge", "a village at the forest edge", "a eucalyptus woodland"]
STYLE = ["photo taken with a smartphone", "amateur snapshot from a car window", "handheld phone photo, slightly shaky",
         "photo from a hiking trail", "photo from a backyard", "phone photo at dusk", "phone photo at midday"]
FIRE = ["wildfire with visible orange flames and smoke", "grass fire with bright flames", "flames climbing a tree, thick smoke",
        "burning brush with flames near the road"]
SMOKE = ["a plume of grey wildfire smoke rising behind the hills", "thick white smoke column in the distance, no visible flames",
         "hazy smoke drifting over the ridge from a distant fire"]
NEG = [("a golden sunset with orange sky", "sunset"), ("morning fog over the valley", "fog"), ("hazy sky with sun glare", "haze"),
       ("a burned hillside after the fire, charred black trees, no flames", "burn_scar"), ("dry yellow vegetation in summer", "dry_veg"),
       ("a small campfire in a fire pit", "campfire"), ("cooking smoke from a barbecue grill", "cooking_smoke"), ("a clear blue sky", "clear")]

def prompts(n, seed):
    rng = random.Random(seed); out = []
    for i in range(n):
        k = rng.choices(["flames", "smoke", "neg"], [0.35, 0.25, 0.4])[0]; sc, st = rng.choice(SCENES), rng.choice(STYLE)
        if k == "flames": core, tag, f, s = rng.choice(FIRE), "", 1, 1
        elif k == "smoke": core, tag, f, s = rng.choice(SMOKE), "", 0, 1
        else:
            core, tag = rng.choice(NEG); f = s = 0
            if tag in ("campfire",): f = 1  # visible flames, but not a wildfire: label flames=1, keep tag
            if tag in ("cooking_smoke",): s = 1
        out.append(dict(prompt=f"{st} of {core}, {sc}", flames=f, smoke=s, tag=tag, seed=rng.randrange(2**31)))
    return out

def load(family):
    from diffusers import AutoPipelineForText2Image, PixArtAlphaPipeline, KolorsPipeline
    if family == "sd15":
        p = AutoPipelineForText2Image.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None); kw = dict(num_inference_steps=25, guidance_scale=7.5, height=512, width=512)
    elif family == "sdxl":
        p = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16"); kw = dict(num_inference_steps=25, guidance_scale=6.0, height=768, width=768)
    elif family == "pixart":
        p = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16); kw = dict(num_inference_steps=20, guidance_scale=4.5, height=768, width=768)
    elif family == "kolors":
        p = KolorsPipeline.from_pretrained("Kwai-Kolors/Kolors-diffusers", torch_dtype=torch.float16, variant="fp16"); kw = dict(num_inference_steps=25, guidance_scale=5.0, height=768, width=768)
    return p.to("cuda"), kw

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--family", required=True); ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    os.makedirs(a.out + "/img", exist_ok=True); pipe, kw = load(a.family); rows = []; t0 = time.time()
    for i, pr in enumerate(prompts(a.n, a.seed + zlib.crc32(a.family.encode()) % 1000)):
        g = torch.Generator("cuda").manual_seed(pr["seed"])
        try: im = pipe(prompt=pr["prompt"], generator=g, **kw).images[0]
        except Exception: traceback.print_exc(); continue
        s = 512 / max(im.size); im = im.convert("RGB").resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        rel = f"img/{a.family}_{i:05d}.jpg"; im.save(f"{a.out}/{rel}", quality=90)
        rows.append(dict(path=rel, family=a.family, **pr, synthetic=1, edited_or_composited=0))
        if i % 25 == 0: print(a.family, i, f"{time.time()-t0:.0f}s", flush=True)
    with open(f"{a.out}/manifest_{a.family}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("DONE", a.family, len(rows), flush=True)

if __name__ == "__main__":
    main()
