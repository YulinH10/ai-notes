import re
from datetime import datetime

import anthropic

from .config import load_config
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


def generate_post(topic: str) -> tuple[str, str]:
    """Generate a blog post about the given topic. Returns (filename, full_markdown)."""
    config = load_config()

    if not config["api_key"]:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. "
            "Set it in .env file or as environment variable."
        )

    client = anthropic.Anthropic(api_key=config["api_key"])
    gen_config = config["generation"]

    message = client.messages.create(
        model=gen_config["model"],
        max_tokens=gen_config["max_tokens"],
        temperature=gen_config["temperature"],
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(topic=topic)}
        ],
    )

    body = message.content[0].text
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    slug = _slugify(topic)
    filename = f"{date_str}-{slug}.md"

    tags = _extract_tags(topic)
    frontmatter = (
        f"---\n"
        f'title: "{topic}"\n'
        f"date: {now.strftime('%Y-%m-%dT%H:%M:%S+08:00')}\n"
        f"draft: false\n"
        f"tags: {tags}\n"
        f'categories: ["{config["blog"]["default_categories"][0]}"]\n'
        f'summary: "AI 学习笔记：{topic}"\n'
        f"---\n\n"
    )

    return filename, frontmatter + body


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text[:60].strip("-") or "post"


def _extract_tags(topic: str) -> list[str]:
    known_tags = {
        "transformer": "Transformer",
        "attention": "Attention",
        "cnn": "CNN",
        "rnn": "RNN",
        "lstm": "LSTM",
        "gru": "GRU",
        "bert": "BERT",
        "gpt": "GPT",
        "gan": "GAN",
        "vae": "VAE",
        "llm": "LLM",
        "rlhf": "RLHF",
        "rag": "RAG",
        "lora": "LoRA",
        "diffusion": "Diffusion",
        "embedding": "Embedding",
        "fine-tuning": "Fine-tuning",
        "微调": "Fine-tuning",
        "强化学习": "强化学习",
        "神经网络": "神经网络",
        "深度学习": "深度学习",
        "自然语言处理": "NLP",
        "nlp": "NLP",
        "计算机视觉": "CV",
        "agent": "AI Agent",
        "提示工程": "Prompt Engineering",
        "prompt": "Prompt Engineering",
    }
    topic_lower = topic.lower()
    tags = [v for k, v in known_tags.items() if k in topic_lower]
    return tags or ["AI基础"]
