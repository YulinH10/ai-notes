import random

TOPIC_POOL = [
    "什么是神经网络：从感知机到多层网络",
    "反向传播算法详解",
    "梯度下降的各种变体：SGD、Adam、AdaGrad",
    "卷积神经网络（CNN）的工作原理",
    "循环神经网络（RNN）与序列建模",
    "LSTM 和 GRU：解决长期依赖问题",
    "Transformer 架构：注意力就是你所需要的一切",
    "自注意力机制（Self-Attention）深入理解",
    "BERT：双向预训练语言模型",
    "GPT 系列模型的演进历程",
    "词嵌入：从 Word2Vec 到 Contextual Embeddings",
    "大语言模型（LLM）的训练流程",
    "RLHF：基于人类反馈的强化学习",
    "提示工程（Prompt Engineering）实践指南",
    "RAG：检索增强生成技术",
    "Fine-tuning 与 LoRA 微调技术",
    "向量数据库与语义搜索",
    "Tokenization：文本如何变成数字",
    "损失函数大全：交叉熵、MSE 与其他",
    "正则化技术：Dropout、L1/L2、BatchNorm",
    "迁移学习的原理与实践",
    "生成对抗网络（GAN）入门",
    "变分自编码器（VAE）详解",
    "扩散模型（Diffusion Models）的数学原理",
    "强化学习基础：马尔可夫决策过程",
    "Q-Learning 与深度 Q 网络（DQN）",
    "多模态模型：视觉与语言的融合",
    "AI Agent 的设计与实现",
    "模型部署：从训练到生产环境",
    "模型量化与推理优化",
    "AI 伦理与安全：对齐问题",
    "Few-shot 和 Zero-shot 学习",
    "知识蒸馏：大模型到小模型的知识传递",
    "混合专家模型（MoE）架构解析",
    "KV Cache 与 LLM 推理加速",
]


def pick_random_topic():
    return random.choice(TOPIC_POOL)
