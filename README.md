# 场记 / Continuity

A [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) plugin that gives an
agent the five senses **and remembers what it made** — the same character stays the same
character across every call, and a failed generation is never allowed to pass as a success.

Runs locally. Models are lazy-loaded per request and released when idle, so **when you are
not using it the GPU is untouched** — 0.21 GB resident, measured. You can play a game on
the same card.

> 场记 is the continuity supervisor on a film set. Their entire job is two things: make sure
> the costume, hair and props match between takes, and catch the mistake on set before it is
> cut into the film. That is exactly this plugin's job.

## Zero residency

Measured on an RX 7800 XT, engines running, no request in flight:

| | GPU |
|---|---|
| idle | **0.21 GB** |
| during image generation | 6.80 GB |
| 2 s after it finishes | **0.21 GB** (holds at 10 / 30 / 60 s) |
| during TTS | 2.35 GB |
| 120 s after TTS | **0.21 GB** |

Images are free: the engine streams weights per request and never keeps them resident.
Audio is released by an idle timer (`AUDIO_IDLE_UNLOAD_S`, default 120 s) — not immediately,
because someone voicing ten lines in a row should not pay a reload each time.

**Reload costs nothing measurable**: the same TTS request took 3.0 s both cold and warm.
Weights are mmap'd and sit in page cache.

Requests are serialized, so peak = the single largest model = **6.80 GB**. An 8 GB card fits
the whole stack.

## Minimum requirements

| | Minimum | Notes |
|---|---|---|
| **GPU** | **8 GB VRAM** | Peak is 6.80 GB (measured). Requests are serialized, so peak is one model, not the sum. |
| **GPU API** | **Vulkan 1.2+** | AMD / NVIDIA / Intel. **No CUDA, no ROCm.** Kernels are SPIR-V compiled at runtime. |
| **Disk** | **~25 GB** | ~18 GB weights + ~4 GB container images + engine binaries. |
| **Host RAM** | **16 GB** (8 GB workable — see below) | Driven by transient peaks, not idle. |
| **CPU** | any x86-64 | Background removal runs on CPU (~7 s per 1024² image). |

Vulkan instead of CUDA is not a preference — it is why this runs at all. ROCm miscomputes
VAE decode on this GPU class ([ROCm#6633](https://github.com/ROCm/ROCm/issues/6633)):
five decodes of identical input returned five mutually uncorrelated results. Vulkan/RADV
compiles SPIR-V at runtime instead of looking up a per-arch kernel table, and is correct
and faster here. The side effect is portability across all three vendors.

### Host RAM in detail

Idle is negligible; the peaks are what sizes the machine. Total RSS across the three
service containers, measured:

| operation | peak RSS |
|---|---|
| idle | 0.52 GB |
| music (30 s) | 0.50 GB |
| speech | 1.63 GB |
| image (1024²) | 4.94 GB |
| `remove_bg` `quality="best"` | **7.74 GB** |
| `remove_bg` `quality="fast"` | 1.33 GB |

Background removal is the ceiling, and its cost is **independent of input size** — 256 / 512 /
1024 px all peak at ~6.8 GB, because BiRefNet runs at a fixed internal resolution. The memory
is transient (it returns to baseline afterwards), but it must exist at that moment.

**On a 16 GB machine everything works.** On **8 GB**, use `quality="fast"` (u2netp): peak drops
to 1.33 GB and it runs in 0.4 s instead of 6.8 s. The trade is edge quality — the fast model
leaves a grey halo where BiRefNet gives clean soft edges, which matters for sprites and does
not for a quick mask.

Weights are mmap'd, so beyond these peaks extra RAM only buys page cache. Without it the
first image after eviction costs +3 s (15.0 s vs 12.0 s, measured).

### On smaller cards

6.80 GB is a floor for this model family, and **quantizing the diffusion model does not
move it**: Q4_0 (2.29 GB of weights) peaks at 6.60 GB, Q8_0 (4.01 GB) at 6.59 GB — identical.
The bottleneck is the **8 GB unquantized 4B text encoder**, not the diffusion model. Lowering
resolution does not help either (512 / 768 / 1024 all peak identically; only time changes).

Ship Q4_0 anyway — same VRAM, 1.7 GB less disk.

Going below 8 GB means changing the text encoder or the model family. That is possible, but
it moves identity pinning from native `ref_images` to IP-Adapter, which is **not yet
verified here** — and identity pinning is the whole point.

## Two things it actually does

**1. Identity survives across calls.** Generation backends are stateless: ask for the same
character twice and you get two people who merely resemble each other. Measured on
Qwen3-TTS, four lines from one voice description:

| | pitch spread across 4 lines |
|---|---|
| straight to the model (default sampling) | 125 Hz |
| straight to the model, **greedy decoding** | **242 Hz — worse** |
| through Continuity (pinned reference) | **5 Hz** |

Under greedy decoding the seed is provably inert — seeds 5 / 99 / 777 produced one identical
sha256 — so randomness was fully eliminated, and it still drifted 242 Hz. **Identity is a
function of the input text, not of the random draw.** `temperature=0` and `top_k=1` cannot fix
it. Only pinning to a reference artifact can.

```
create_actor(name, voice)          -> audition clip; listen before you commit
actor_tts(actor, text)             -> same timbre every line

create_character / create_animal / create_object (name, appearance)
subject_image(subject, scene)      -> same look, new scene / angle / outfit
```

Identity and wardrobe are separate: pin the face and build, then change clothes in the scene
prompt. A reference in an indigo robe, asked for `wearing heavy red armor`, comes back in
armor with the same face.

**2. Degenerate output is refused.** A backend that miscomputes returns a perfectly
well-formed all-zero WAV, or a flat grey PNG, with HTTP 200. Every artifact is checked
(image standard deviation, audio RMS, non-finite samples) and the job fails loudly rather
than returning `status: done` over garbage.

Plus `remove_bg`: diffusion models draw "transparent background" as an opaque checkerboard;
this turns it into a real RGBA cutout, which sprites require.

## Prior art

A survey of the current MCP ecosystem — MiniMax-MCP, openrouter-mcp-multimodal, AtlasCloud,
the dsh vision/draw plugins, and four game-asset servers — found voice cloning in several,
**visual subject pinning in none, and output verification in none**.

## Bring your own backend (optional)

Local engines are the default, but every backend is a URL. Point it at your own server or any
OpenAI-compatible provider and the local models are never loaded.

One hard requirement either way: the image backend **must accept a reference image**
(FLUX.2-style native `ref_images`, IP-Adapter, or PuLID for faces). Without it, identity
pinning cannot work — and the plugin says so instead of silently degrading.

## Status

Scaffold. The implementation is being extracted from a production MCP server.

```
bundle/   dsh bundle (npm) — one plugin row; dsh spawns and supervises the server below
python/   MCP server — pinning, guardrails, verification, cutout, backend lifecycle
```

## License

MIT
