# 当前 RFSQ 方法数据流

> 当前配置：动作长度 `32`，latent registers `H=8`，latent 维度 `D=4`，两级 FSQ 的 `levels=[8,5,5,5]`，每级 codebook 大小为 `1000`。

## 1. RFSQ Action Tokenizer

```mermaid
flowchart LR
    A["动作轨迹<br/>a: B×32×A"] --> N["LinearNormalizer<br/>动作归一化"]
    N --> E["RegisterEncoder<br/>2 层 Transformer<br/>32 actions + 8 registers"]
    E --> Z["连续 latent<br/>z: B×8×4"]

    subgraph RFSQ["Two-stage Residual FSQ"]
        direction TB
        Z --> S1["Stage 1 FSQ<br/>tanh bound + round STE<br/>levels: 8,5,5,5"]
        S1 --> Q1["粗量化 q₁: B×8×4<br/>idx₁: B×8"]
        Z --> RES["残差<br/>r₁ = z - q₁"]
        Q1 --> RES
        RES --> RN["学习式残差标准化<br/>r′ = (r₁-μ₂)/σ₂<br/>σ₂ = clamp exp logσ₂"]
        RN --> S2["Stage 2 FSQ<br/>tanh bound + round STE<br/>levels: 8,5,5,5"]
        S2 --> Q2N["q₂′: B×8×4<br/>idx₂: B×8"]
        Q2N --> Q2["反标准化<br/>q₂ = q₂′×σ₂ + μ₂"]
        Q1 --> SUM["重建 latent<br/>ẑ = q₁ + q₂<br/>B×8×4"]
        Q2 --> SUM
    end

    Q1 -. idx₁ .-> TOK["Tokenizer 原生 stage-major tokens<br/>idx₁¹…idx₁⁸ | idx₂¹…idx₂⁸<br/>B×16"]
    Q2N -. idx₂ .-> TOK
    SUM --> D["SinglePassDecoder<br/>4 层 causal Transformer<br/>8 latents → 32 actions"]
    D --> AH["重建动作 â_norm<br/>B×32×A"]
    N -. target a_norm .-> LOSS["MSE 重建损失"]
    AH --> LOSS
    LOSS -. 反向传播 .-> E
    LOSS -. 反向传播 .-> RFSQ
    LOSS -. 反向传播 .-> D

    classDef data fill:#e9f2ff,stroke:#0c66e4,color:#0747a6,stroke-width:1.5px;
    classDef module fill:#eae6ff,stroke:#6554c0,color:#403294,stroke-width:1.5px;
    classDef decision fill:#fff7d6,stroke:#e2b203,color:#7f5f01,stroke-width:1.5px;
    classDef output fill:#e3fcef,stroke:#22a06b,color:#216e4e,stroke-width:1.5px;
    class A,Z,Q1,Q2N,TOK data;
    class N,E,S1,RN,S2,Q2,D module;
    class RES decision;
    class SUM,AH,LOSS output;
```

Tokenizer 的两个 stage 各自产生 8 个 token。它们使用相同大小的离散词表，但 FSQ 模块参数不共享。`μ₂` 和 `logσ₂` 是可学习参数。

## 2. Pair-aware Policy 训练

```mermaid
flowchart LR
    ACT["GT 动作轨迹<br/>B×32×A"] --> FT["冻结的 RFSQ Tokenizer"]
    FT --> SM["stage-major tokens<br/>q₁¹…q₁⁸ | q₂¹…q₂⁸"]
    SM --> IL["成对交错重排<br/>q₁¹,q₂¹,q₁²,q₂²,…,q₁⁸,q₂⁸"]
    IL --> GT["干净目标<br/>BOS + 16 tokens"]

    OBS["观测 obs<br/>RGB + state，To=2"] --> OE["Fused Observation Encoder"]
    OE --> FEAT["condition features<br/>B×To×d"]
    FEAT --> AR["Causal AR Transformer<br/>vocab = 1000 + BOS"]

    GT --> MODE{"self-conditioned<br/>prefix enabled?"}
    MODE -- "否" --> TF["Teacher forcing<br/>并行输入 shifted GT"]
    TF --> AR

    MODE -- "是" --> STEP["逐 token forward_step<br/>KV cache"]
    STEP --> AR
    AR -. "greedy argmax<br/>限制在 action vocab<br/>并 detach" .-> MIX
    GT -. "GT anchor" .-> MIX
    MIX["Pair-level prefix 选择<br/>每个 pair 只抽一次 Bernoulli<br/>q₁ᶦ 与 q₂ᶦ 共用同一选择"] --> STEP

    AR --> LOGITS["16 个位置的 logits"]
    GT -. "监督始终使用干净 GT" .-> CE["Cross Entropy<br/>mean over B×16"]
    LOGITS --> CE

    NOTE["核心语义<br/>use_k_tokens = K 表示 K 个完整 latent pair<br/>policy 实际生成 2K 个离散 token"]

    classDef data fill:#e9f2ff,stroke:#0c66e4,color:#0747a6,stroke-width:1.5px;
    classDef module fill:#eae6ff,stroke:#6554c0,color:#403294,stroke-width:1.5px;
    classDef self fill:#fff7d6,stroke:#e2b203,color:#7f5f01,stroke-width:1.5px;
    classDef output fill:#e3fcef,stroke:#22a06b,color:#216e4e,stroke-width:1.5px;
    class ACT,SM,GT,OBS,FEAT,LOGITS data;
    class FT,IL,OE,AR,TF,STEP module;
    class MODE,MIX,NOTE self;
    class CE output;
```

Pair-aware self-conditioning 的关键点是：同一个 Bernoulli 决策同时控制一对 `(q₁ᶦ,q₂ᶦ)` 的 prefix 来源，因此不会出现“半对使用预测、半对使用 GT”的情况。无论 prefix 如何混合，交叉熵目标始终是干净的 interleaved GT token。

## 3. Policy 推理与动作执行

```mermaid
flowchart LR
    OBS["当前观测<br/>RGB + state，To=2"] --> OE["Observation Encoder"]
    OE --> AR["Causal AR generate<br/>temperature + top-k"]
    BOS((BOS)) --> AR
    AR --> TOK["生成 2K 个 interleaved tokens<br/>q₁¹,q₂¹,…,q₁ᴷ,q₂ᴷ"]
    TOK --> SPLIT["奇偶位拆分<br/>idx₁ = tokens 0::2<br/>idx₂ = tokens 1::2"]
    SPLIT --> EMB1["Stage 1<br/>index → q₁"]
    SPLIT --> EMB2["Stage 2<br/>index → q₂′<br/>q₂=q₂′×σ₂+μ₂"]
    EMB1 --> SUM["K 个完整 latent<br/>q₁ + q₂"]
    EMB2 --> SUM
    SUM --> PAD["补齐到 8 个 latent<br/>缺失尾部置零并 mask"]
    PAD --> DEC["冻结的 SinglePassDecoder"]
    DEC --> CHUNK["预测动作块<br/>action_pred: B×32×A"]
    CHUNK --> EXEC["执行前 16 步动作"]
    EXEC -. "获取新观测并滚动重规划" .-> OBS

    classDef data fill:#e9f2ff,stroke:#0c66e4,color:#0747a6,stroke-width:1.5px;
    classDef module fill:#eae6ff,stroke:#6554c0,color:#403294,stroke-width:1.5px;
    classDef output fill:#e3fcef,stroke:#22a06b,color:#216e4e,stroke-width:1.5px;
    class OBS,TOK data;
    class OE,AR,SPLIT,EMB1,EMB2,PAD,DEC module;
    class BOS,SUM,CHUNK,EXEC output;
```

## 对应代码

- `oat/tokenizer/oat/quantizer/rfsq.py`：两阶段 residual FSQ。
- `oat/tokenizer/oat/tokenizer.py`：动作归一化、编码、量化、解码。
- `oat/policy/oatpolicy_rfsq_pair.py`：stage-major/interleaved 转换、pair-aware 训练与推理。
- `oat/policy/self_conditioned_prefix.py`：self-conditioned prefix 配置和概率调度。

> 注意：self-conditioned 分支只有在 `self_conditioned_prefix.enabled: true` 时启用；否则走普通 teacher forcing。
