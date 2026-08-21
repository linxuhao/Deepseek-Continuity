# dsh-plugin-continuity

The [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) plugin row for
**[场记 / Continuity](https://github.com/linxuhao/Deepseek-Continuity)** — local image / voice /
music / SFX generation for an agent, where the same character stays the same character across
every call, and a failed generation is never allowed to pass as a success.

## This package is only half of it

It contains one thing: a cordis patch that tells dsh how to spawn the MCP server. **Install the
backend first**, or every tool will report that it cannot reach the engines:

```bash
uvx --from dsh-continuity continuity-setup
```

That checks your machine (VRAM / RAM / disk), builds the two ggml/Vulkan engines, downloads only
the weights your card can actually use, and starts them. It needs **8 GiB VRAM** and Vulkan —
no CUDA, no ROCm. Below 8 GiB it explains why image generation will not fit and offers the
audio-only install (4 GiB).

## Install

```bash
dsh plugin --profile <your-profile> add dsh-plugin-continuity
```

Tools appear as `mcp__continuity__*` — `create_actor` / `actor_tts`, `create_character` /
`create_object` / `subject_image`, `import_actor` / `import_subject`, `remove_bg`,
`slice_sheet`, `gen_sfx`, and more. 19 in total.

Configuration is all environment variables, and `continuity-setup` prints the values for your
machine when it finishes. See [`cordis.patch.yml`](https://github.com/linxuhao/Deepseek-Continuity/blob/main/bundle/cordis.patch.yml)
for every passthrough, and the [main README](https://github.com/linxuhao/Deepseek-Continuity)
for what the thing actually does and the measurements behind every limit.

MIT.
