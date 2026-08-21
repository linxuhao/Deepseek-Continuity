# 场记 / Continuity

A [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) plugin that gives an
agent local image / speech / music / SFX generation **and remembers what it made** — the same
character stays the same character across every call, and a failed generation is never allowed
to pass as a success.

Runs locally. Models are lazy-loaded per request and released when idle, so **when you are
not using it the GPU is untouched** — 0.21 GiB resident, measured. You can play a game on
the same card.

> 场记 is the continuity supervisor on a film set. Their entire job is two things: make sure
> the costume, hair and props match between takes, and catch the mistake on set before it is
> cut into the film. That is exactly this plugin's job.

## Install

```bash
uvx --from dsh-continuity continuity-setup
```

That one command does the whole backend: preflight → build the engines → fetch only the weights
this machine can use → start them.

> The PyPI distribution is [**`dsh-continuity`**](https://pypi.org/project/dsh-continuity/)
> (the import name stays `continuity_mcp`). It is *not* `continuity-mcp` — that name on PyPI
> belongs to an unrelated project, so do not `uvx continuity-mcp`.
>
> To run from source instead: `uvx --from git+https://github.com/linxuhao/Deepseek-Continuity continuity-setup`

Then add the plugin to your dsh profile. **`dsh plugin` shells out to pnpm**, so install that
first if you have not (`corepack enable pnpm`); without it the command stops at
`pnpm not found on PATH`:

```bash
dsh plugin --profile <your-profile> add dsh-plugin-continuity
```

The bundle reads its settings from the environment, so export what `continuity-setup` printed
for your machine before booting the profile:

```bash
export CONTINUITY_STATE_DIR=~/.continuity
export CONTINUITY_SD_SERVER=http://127.0.0.1:9020
export CONTINUITY_AUDIO_SERVER=http://127.0.0.1:9021
```

To wire it by hand instead — `continuity-setup` prints this block filled in for your machine:

```yaml
- insert:
    - id: continuity
      name: '@deepseek-ai/dsh-mcp-client'
      config:
        serverName: continuity
        transport: stdio
        command: uvx
        args: ['--from', 'dsh-continuity', 'continuity-mcp']
        env:
          CONTINUITY_STATE_DIR: !!js process.env.CONTINUITY_STATE_DIR ?? ''
          SD_SERVER: !!js process.env.CONTINUITY_SD_SERVER ?? ''
          AUDIO_SERVER: !!js process.env.CONTINUITY_AUDIO_SERVER ?? ''
```

(the complete row, with every passthrough documented: [`bundle/cordis.patch.yml`](https://github.com/linxuhao/Deepseek-Continuity/blob/main/bundle/cordis.patch.yml))

`continuity-setup` checks the machine before it downloads anything, and sizes the install to
what it finds. Run `continuity-setup --check` first to see what it would do — that reads
hardware and changes nothing:

```
体检结果:
  GPU     AMD Radeon RX 7800 XT (RADV NAVI32)  (16.0 GiB, 此刻可用 15.8 GiB, DISCRETE_GPU, vulkan device 1)
          未选 AMD Radeon RX 7900 XTX (RADV NAVI31) (24.0 GiB, 此刻可用 1.4 GiB)
          跳过 llvmpipe —— 软件渲染, 不是真显卡
  内存    30.9 GiB
  磁盘    3118.4 GiB 可用 / 需要 30 GiB
  生图    启用
  音频    启用
  抠图默认档  best
```

Two details in there that exist because the naive version is wrong:

- **It skips `llvmpipe`.** The software rasterizer advertises 30.9 GiB of "VRAM" (it is your
  system RAM) and would win any "pick the biggest card" contest. Everything would then run on
  the CPU — working, looking completely normal, and unusably slow.
- **It picks by free VRAM, gates by total VRAM.** On the machine above the 24 GiB card has
  1.4 GiB actually free because another process holds it; picking by size would select it and
  then OOM. But "is this card good enough" is a hardware question, so that one uses the total —
  otherwise a 16 GiB card would be rejected for having a game open.

## Minimum requirements

| | Minimum | Notes |
|---|---|---|
| **GPU** | **8 GiB VRAM** | Peak is 6.80 GiB (measured). Requests are serialized, so peak is one model, not the sum. |
| **GPU API** | **Vulkan 1.2+** | **No CUDA, no ROCm.** Kernels are SPIR-V compiled at runtime. |
| **Disk** | **30 GiB during install**, 19.5 GiB after | 17.4 weights + 2.1 runtime image + 8.5 build layers (reclaimable). |
| **Host RAM** | **16 GiB** (8 GiB workable — see below) | Driven by transient peaks, not idle. |
| **CPU** | any x86-64 | Background removal runs on CPU. |

Audio-only installs (see below) need **20 GiB during install, 9.5 GiB after**.

All VRAM/RAM figures on this page are **GiB** (2³⁰ bytes), which is what `rocm-smi` and
`vulkaninfo` report. An earlier version of this README labelled them GB; that was wrong and
made the headroom look tighter than it is.

Vulkan instead of CUDA is not a preference — it is why this runs at all. ROCm miscomputes
VAE decode on this GPU class ([ROCm#6633](https://github.com/ROCm/ROCm/issues/6633)):
five decodes of identical input returned five mutually uncorrelated results. Vulkan/RADV
compiles SPIR-V at runtime instead of looking up a per-arch kernel table, and is correct
and faster here. The side effect is portability across all three vendors.

### GPU vendors

| | How the container gets the GPU | Status |
|---|---|---|
| **AMD** | `/dev/dri` + mesa RADV inside the image | **Tested** (RX 7800 XT, RX 7900 XTX) |
| **Intel** | `/dev/dri` + mesa ANV inside the image — same mechanism | Untested |
| **NVIDIA** | `nvidia-container-toolkit` injects the host driver (`docker-compose.nvidia.yml`) | Untested |

I only have AMD cards, so I will not claim more than that. Nothing in the code is
AMD-specific — no CUDA, no ROCm, no HIP, no `/dev/kfd`, no `gfx` targets — and ggml's Vulkan
backend is widely run on NVIDIA. But "widely run" is not "I verified it".

The NVIDIA path is a genuinely different wiring, not just a different card: NVIDIA's Vulkan
ICD lives in the host driver and must be injected by `nvidia-container-toolkit`, with
`NVIDIA_DRIVER_CAPABILITIES` including `graphics` — the default `compute,utility` gives you
working CUDA and an empty device list in Vulkan. `continuity-setup` detects NVIDIA, uses the
right compose overlay, and tells you the path is unverified. Reports either way are welcome.

### Host RAM in detail

Idle is negligible; the peaks are what sizes the machine.

| operation | peak RSS |
|---|---|
| idle | 0.52 GiB |
| music | 0.50 GiB |
| speech | 1.63 GiB |
| image (1024²) | 4.94 GiB |
| `remove_bg` `quality="best"` | **7.74 GiB** |
| `remove_bg` `quality="fast"` | 1.33 GiB |

Background removal is the ceiling, and its cost is **independent of input size** — 256 / 512 /
1024 px all peak at ~6.8 GiB, because BiRefNet runs at a fixed internal resolution.

**On 16 GiB everything works.** Below 12 GiB, `continuity-setup` sets the default to
`quality="fast"` (u2netp): peak drops to 1.33 GiB and it runs in 0.6 s instead of 7.2 s. On a
typical game sprite the two are hard to tell apart by eye — checked side by side over a magenta
backdrop with the edges zoomed. `best` remains the default where there is room, because the
models do differ in principle on fine edges (hair, semi-transparent fringes), but treat `fast`
as a legitimate choice rather than a degraded fallback.

## One rule, not a tier list

Jobs are serialized, so **at any moment exactly one model is needed**. Everything else is
released before the job starts. That is the whole VRAM policy. (The one exception is a split
deployment: if the image backend is on a different host from the audio one, they are not
competing for a card, so nothing is released — freeing local VRAM for a remote job buys
nothing and costs a reload.)

It buys a property worth more than a few saved seconds: **peak VRAM is a constant 6.80 GiB
regardless of what you call, in what order.** Measured over an alternating
speech→image→speech→image sequence:

| | peak | speech | image | 6 calls |
|---|---|---|---|---|
| keep models resident | **10.94 GiB** | 2.8 s avg | 11.5 s | 42.9 s |
| release what isn't needed | **6.79 GiB** | 4.8 s | 11.6 s | 49.2 s |

Keeping them resident is 16% faster and **does not fit an 8 GiB card** — and "voice a line,
then draw something" is the most ordinary sequence there is. An earlier version of this README
quoted 7.84 GiB for that overlap; that came from a lighter sequence I happened to test, and
using it as the ceiling was wrong. A cloned voice keeps its reference audio resident too, which
is where the rest comes from.

**What the reload actually costs:** 4.8 s instead of 1.2 s, and only on the first call after
switching away. Ten dialogue lines in a row pay it once:

```
第 1 句 4.63s   之后九句平均 1.19s   十句合计 15.4s
```

So there is no VRAM tier list, and no 12 GiB threshold. Above 8 GiB every card behaves
identically. Below 8 GiB the installer explains why image generation will not fit and asks
whether to install the audio half alone — it does not quietly substitute a different product:

```
  生图    显存不足
          Fake GTX 1060 只有 6.0 GiB, 而生图实测峰值 6.80 GiB, 需要 8 GiB。
          换更小的生图模型省不下这部分 (Q4 与 Q8 峰值相同 6.60 / 6.59), 降分辨率也不行
          —— 瓶颈是那个 8 GiB 不量化的文本编码器。
          音频那半仍然可以装: 铸声/配音/音乐/音效/抠图都能用, 4 GiB 就够。

  ⚠️ 这张卡装不了生图那半。
     只装音频那半 (铸声/配音/音乐/音效/抠图)? [y/N]
```

The audio-only install is a real product, not a consolation prize: casting voices, dialogue,
music, SFX and cutout all work in 4 GiB.

**What does not adapt at all: the image model.** Quantizing it does not move VRAM —
Q4_0 (2.29 GiB of weights) peaks at 6.60 GiB, Q8_0 (4.01 GiB) at 6.59 GiB, identical. Lowering
resolution does not help either (512 / 768 / 1024 all peak the same; only time changes). The
bottleneck is the **8 GiB unquantized 4B text encoder**. So there is no "medium" image tier to
offer, only installed or not. (Q4_0 ships anyway — same VRAM, 1.7 GiB less disk.)

Going below 8 GiB for images means changing the text encoder or the model family. That is
possible, but it moves identity pinning from native `ref_images` to IP-Adapter, which is
**not verified here** — and identity pinning is the whole point.

The one thing that does still key off a resource is host RAM, and it is a different resource:
below 12 GiB RAM the cutout default drops to `quality="fast"` (see above).

## Zero residency

Measured on an RX 7800 XT with nothing else on the card:

| | GPU |
|---|---|
| idle | **0.21 GiB** |
| during image generation | 6.80 GiB |
| 2 s after it finishes | **0.21 GiB** |
| during TTS | 2.39 GiB |
| 120 s after TTS | **0.21 GiB** |

Images are free: the engine streams weights per request and never keeps them resident.
Audio is released by an idle timer (`AUDIO_IDLE_UNLOAD_S`, default 120 s) — not immediately,
because someone voicing ten lines in a row should not pay a reload each time. Reload costs
nothing measurable: the same TTS request took 3.0 s both cold and warm, because weights are
mmap'd and sit in page cache.

Requests are serialized and everything unneeded is released first, so peak = the single largest
model, always. The idle timer covers the one case the rule cannot: after the *last* job there is
no next job to trigger a release, so the timer does it. Closing the agent releases the VRAM too —
the MCP server unloads on exit rather than leaving the engines holding it.

## Two things it actually does

**1. Identity survives across calls.** Generation backends are stateless: ask for the same
character twice and you get two people who merely resemble each other. Measured on
Qwen3-TTS as pitch spread across four lines of one character — same voice description,
same lines, the only variable being whether a reference was pinned:

| voice under test | straight to the model | through Continuity |
|---|---|---|
| a bright narrator | 125 Hz | **5 Hz** |
| an elderly gravelly voice | 74 Hz | **29 Hz** |

Two different voices, two different magnitudes, same direction. **Read the ratio, not the
headline number** — how far a description drifts depends on the description. And treat f0
spread as a proxy, not the verdict: autocorrelation pitch tracking makes octave errors on low
gravelly voices (an earlier run of the table above reported 76 Hz where the octave-corrected
figure is 29), so the numbers above anchor each line's search range to the reference. The real
acceptance test is listening to the audition clip, which is why `create_actor` hands you one.

What the number cannot show is the part that matters most: **the drift is not random.**

| | pitch spread across 4 lines |
|---|---|
| default sampling | 125 Hz |
| **greedy decoding** | **242 Hz — worse** |
| pinned reference | 5 Hz |

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

**Already cast your character somewhere else?** `import_actor` and `import_subject` pin an
artifact you supply — a real voice recording, an ElevenLabs clip, a character sheet from
another tool — and everything downstream behaves identically. Audio is normalized to 24 kHz
mono for you (44.1 kHz stereo in, verified: reference f0 identical, and an imported actor
tracks a natively-cast one to 11 Hz).

### Looking at what it made

The pinning tools tell the agent to *look at the reference before committing to it*. So they
return the image, not just its path — a 512 px JPEG (~35 KB) alongside the text, as MCP image
content. **There is no VLM in this plugin and there will not be one:** a vision model wants its
own VRAM, which would destroy the property that peak = the single largest model, and the 8 GiB
floor rests on that. The harness already has a model; hand it the picture instead of running a
second one.

Verified end to end on dsh `0.1.1-rc.1` with the vision model — plain `deepseek-v4-flash`
does **not** accept images and answers `INVALID_REQUEST: This model does not support image`:

```yaml
- id: agent-default-model
  config:
    provider: deepseek-official
    model: deepseek-v4-flash-vision-exp
```

Asked to pin *"a square metal lantern with EXACTLY FIVE blue glass panels and a green
handle"* and then check the render against that description item by item, the agent answered:

> **面板数量 — 不符合。** 图中实际可见的是 4 块蓝色面板(正面 2 + 右侧面 2)，并非 5 块。
> 而且从"每面 2 块"的网格规律看，若其余两面同规格，总数应为 8 块。

It counted, it disagreed with the prompt it had just been given, and it said what it actually
saw. That is the loop the statistical checks cannot close: they catch a grey PNG, this catches
*"that is not the thing I asked for."*

On a model without image input the block degrades to `[image unavailable]` and the run
continues normally — observed on dsh, not assumed; the agent then says it received no image
rather than guessing from `appearance`. `CONTINUITY_INLINE_IMAGES=0` sends text only.

One older caveat, corrected: an earlier experiment here had a self-hosted 27B VLM score 9/9/10
on chest renders whose lids were visibly the wrong shape, and I had written that off as "VLM
judges are blind to geometry". The panel-counting result above is evidence that was a
statement about *that* model, not a general law. Writing geometry explicitly into `appearance`
is still the cheaper fix, but the check is now worth running.

**2. Degenerate output is refused.** A backend that miscomputes returns a perfectly
well-formed all-zero WAV, or a flat grey PNG, with HTTP 200. Every artifact is checked
(image standard deviation, audio RMS, non-finite samples) and the call fails loudly rather
than reporting success over garbage. Cutouts additionally get a quality report — mostly
transparent, nothing removed, subject shattered into fragments, holes eaten through the
subject — each with a specific warning instead of a silent pass.

Plus `remove_bg`: diffusion models draw "transparent background" as an opaque checkerboard;
this turns it into a real RGBA cutout, which sprites require. And `gen_sfx`, which synthesizes
sfxr-style game SFX procedurally — bit-identical for a given seed, milliseconds, no GPU —
because a diffusion model is the wrong instrument for a 40 ms coin pickup.

## Tools

19 tools. Everything returns **absolute local file paths**, not URLs — the agent and the
engines are on the same machine, so a path can go straight into your game project without a
download step, and there is no file server to run or misconfigure.

| | |
|---|---|
| voice | `create_actor` `import_actor` `actor_tts` `list_actors` `delete_actor` `generate_speech` |
| look | `create_character` `create_animal` `create_object` `import_subject` `subject_image` `list_subjects` `delete_subject` `generate_image` |
| audio | `generate_music` `gen_sfx` |
| post | `remove_bg` `slice_sheet` |
| meta | `continuity_status` |

`generate_image` and `generate_speech` exist for one-offs and say so in their own descriptions:
they explicitly tell the agent that what they produce will not come back on the next call, and
point at the pinning tools for anything recurring.

## Limits, and why each one exists

Every number here is a measured failure boundary, not a policy.

| limit | value | what happens past it |
|---|---|---|
| line length | 200 chars | 600 chars wedged the GPU: `amdgpu GPU reset(6)`, device lost, an unrelated process on the *other* card killed. 200 is half the largest known-safe value. |
| reference audio | 15 s | ~0.19 GiB VRAM per second: 15 s → 6.59 GiB, 30 s → 9.04 GiB. 15 s is the last value that stays under the image peak, so voice never becomes the ceiling. One value for every card — 3–10 s is already enough to pin a timbre, so a bigger cap on bigger cards would only mean "this clip imports on my machine and not on yours". |
| casting script | 45 chars | It produces the reference audio, which is then re-read on every later line. Char count is a bad proxy (60 chars measured 19.1 s, not the 13.7 s the ratio predicts), so the real duration is checked after casting and reported. |
| image size | 1024 px | 1280 pushed VRAM to 14.5/16.4 GiB; 2048 sent the driver into `restore_userptr_worker` thrashing with the process stuck in uninterruptible `D` state — worse than a clean OOM. |
| music length | 120 s | Not a safety limit: the engine silently truncates at 120 s and reports success. The limit turns that into an explicit `clamped` field. |

Imported audio below 24 kHz is accepted but flagged: upsampling cannot restore the octave
that was thrown away, so the clone comes out duller than the file you gave it. That is worth
a warning rather than a silent pass — it is the same failure shape as everything else this
plugin exists to catch.

**Oversized inputs are handled differently by type, on purpose.** An image that is too large is
resized and the result is reported back to you (`原图 2400x1600 → 存为 1024x682`) — a scaled
picture still depicts the same thing. Reference audio that is too long is **rejected, not
trimmed**: cutting the tail off the audio would leave the transcript describing something the
audio no longer says, and that alignment is exactly what the cloning depends on. Trimming it
silently would hand you an actor that imported successfully and sounds like someone else.

## Bring your own backend (optional)

There are two independent backend URLs, so **you can move one capability off-box and keep the
other local**:

| | env var | what it must be |
|---|---|---|
| image | `SD_SERVER` | a **stable-diffusion.cpp `sd-server`** (`/sdcpp/v1/img_gen` + poll, accepts `ref_images`) |
| audio | `AUDIO_SERVER` | an **audio.cpp `audiocpp_server`** (`/v1/tasks/run`, `/v1/tasks/unload_models`) |

Tell the installer which half is yours and it skips that half entirely — no weights, no engine,
no VRAM gate — while the tools stay registered:

```bash
continuity-setup --sd-server http://your-box:9020      # 生图你自己供; 本地只装音频
continuity-setup --audio-server http://your-box:9021   # 反过来
```

That matters more than it sounds: without it, BYO-ing the image half still downloaded **10.1 GiB**
of image weights and started a local `sd-server` nobody would ever call, then refused to enable
image tools because the local card was too small. On a 5.3 GiB integrated GPU, `--sd-server`
turns "生图 显存不足" into "生图(BYO)" and downloads nothing.

Note the deliberate split from `--no-image`: that one means *"I don't want this capability"*
(tools unregistered); `--sd-server` means *"I supply this capability"* (tools work normally).

Either half also works when set purely at runtime via the two env vars, whether you installed
from PyPI or wired the dsh plugin — the cordis row passes them straight through.
`continuity_status` names whichever side is unreachable. `gen_sfx` needs no backend at all.

**Be clear about what "your own backend" means here: the same engine, elsewhere.** It is not a
provider abstraction. The client speaks sd.cpp's and audio.cpp's specific HTTP shapes, so you
cannot point `SD_SERVER` at an OpenAI-compatible endpoint, a ComfyUI instance, or a bare
IP-Adapter server and expect it to work. What it *is* good for: running the engines on a
beefier box, or sharing one backend between several agents. (An earlier version of this README
implied any reference-image-capable backend would do. That was never true of the code.)

One constraint if you go remote: identity pinning needs the image backend to accept a
reference image. sd.cpp's `ref_images` is what the code uses; without it there is no pinning,
which is the whole point.

Reference *audio* used to be a second constraint — the engine was handed a filesystem path and
opened the file itself, so a remote audio backend meant casting succeeded and every line after
it failed. That was not an engine limitation, it was the wrong endpoint: `/v1/audio/speech`
takes the reference inline as base64 (5 MiB cap; a 15 s reference is ~720 KB), exactly the way
the image path had always passed `ref_images`. Both halves are symmetric now and nothing has to
share a directory.

## Prior art

A survey of the current MCP ecosystem — MiniMax-MCP, openrouter-mcp-multimodal, AtlasCloud,
the dsh vision/draw plugins, and four game-asset servers — found voice cloning in several,
**visual subject pinning in none, and output verification in none**.

## Layout

```
bundle/   dsh bundle (npm) — one plugin row; dsh spawns and supervises the MCP server
src/      the MCP server: pinning, guardrails, verification, cutout, VRAM lifecycle
src/continuity_mcp/deploy/   compose + engine Dockerfile + weight manifest
```

## License

MIT
