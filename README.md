# assetcast — 给 agent 用的资产一致性与输出验证层

一个 [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) 插件。
它**不托管任何模型**，只要求你给它 backend URL（自己 host，或指向任何 OpenAI 兼容的服务）。

## 它只做两件事

**1. 同一个角色前后是同一个人。**

生成模型每次调用给的都是"长得像但不是同一个"的东西。同一个 NPC 的四句台词是四个嗓子，
同一个角色的头像/立绘/地图小人是三个人，同一个宝箱换个角度就是另一个箱子。

| | 同一角色四次输出的漂移 |
|---|---|
| 直接调模型（默认采样）| 基频极差 125 Hz |
| 直接调模型 + 贪心解码 | **242 Hz（更差）** |
| 经 assetcast 定妆/铸声 | **5 Hz** |

关键在于这不是随机性问题：贪心解码下 seed 已经完全失效（三个 seed 同一个 sha256），
可它照样漂 242 Hz。**外观和音色是输入文本的函数**，锁 `temperature` / `top_k` 锁不住。
唯一的解法是把它钉在一段参考音 / 一张定妆图上。

**2. 生成失败不会伪装成成功。**

后端算错时会返回一个格式完全合法的全零 WAV、或一张纯灰图，HTTP 200。
assetcast 对每个产物做退化检查（图像 std、音频 RMS/非有限值），不合格就让任务显式失败。

## 状态

骨架阶段。核心实现来自一个已在生产运行的 MCP server，正在抽成 backend 无关的形式。

## 结构

```
bundle/   dsh bundle (npm): 一行 plugin row, 由 dsh 拉起并托管下面这个 server
python/   MCP server: 一致性层 + 护栏 + 退化检查 + 抠图, 不含任何模型权重
```

## License

MIT
