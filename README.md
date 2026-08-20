# 场记 / Continuity

A [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) plugin that keeps
generated assets **the same thing across calls**, and refuses to let a failed generation
pass as a success.

**It hosts no models.** You point it at backend URLs — self-hosted, or any
OpenAI-compatible provider.

> 场记 is the continuity supervisor on a film set. Their entire job is two things:
> make sure the actor's costume, hair and props match between takes, and catch the
> mistake on set before it gets cut into the film. That is exactly this plugin's job.

## Why

Generation backends are stateless. Ask for the same character twice and you get two
people who merely resemble each other. This is not a prompt problem and not a sampling
problem — measured on Qwen3-TTS, four lines from one "voice" description:

| | pitch spread across 4 lines |
|---|---|
| straight to the model (default sampling) | 125 Hz |
| straight to the model, **greedy decoding** | **242 Hz — worse** |
| through Continuity (pinned reference) | **5 Hz** |

Under greedy decoding the seed is provably inert (seeds 5 / 99 / 777 produced one
identical sha256), so randomness was fully eliminated — and it still drifted 242 Hz.
**Identity is a function of the input text, not of the random draw.** `temperature=0`
and `top_k=1` cannot fix it. The only fix is pinning identity to a reference artifact.

The same applies to images. A survey of the current MCP ecosystem (MiniMax-MCP,
openrouter-mcp-multimodal, AtlasCloud, the dsh vision/draw plugins, and four game-asset
servers) found voice cloning in several, **visual subject pinning in none, and output
verification in none**.

## What it does

**1. Pin identity.** Cast once, then every later call references that cast.

```
create_actor(name, voice)        -> audition clip; listen before you commit
actor_tts(actor, text)           -> same timbre every line

create_character(name, appearance)   -> reference sheet
create_animal(name, appearance)
create_object(name, appearance)      -> three-quarter framing; geometry is what drifts
subject_image(subject, scene)        -> same look, new scene / angle / outfit
```

Identity and wardrobe are separate: pin the face and build, then change clothes in the
scene prompt. Measured — a reference in an indigo robe, asked for `wearing heavy red
armor`, comes back in armor with the same face.

**2. Refuse degenerate output.** A backend that miscomputes returns a perfectly
well-formed all-zero WAV, or a flat grey PNG, with HTTP 200. Every artifact is checked
(image standard deviation, audio RMS and non-finite samples) and the job fails loudly
instead of returning `status: done` over garbage.

**3. Real alpha.** Diffusion models draw "transparent background" as an opaque
checkerboard. `remove_bg` turns it into an actual RGBA cutout — required for sprites.

## Backend contract

The image backend **must accept a reference image** (FLUX.2-style native `ref_images`,
IP-Adapter, or PuLID for faces). Without it, identity pinning cannot work, and this
plugin will tell you so rather than silently degrading.

## Status

Scaffold. The implementation is being extracted from a production MCP server into this
backend-agnostic form.

```
bundle/   dsh bundle (npm) — one plugin row; dsh spawns and supervises the server below
python/   MCP server — pinning, guardrails, verification, cutout. No model weights.
```

## License

MIT
