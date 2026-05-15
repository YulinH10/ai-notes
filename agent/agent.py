"""CLI entry point for the AI blog post generator agent."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from agent.generator import generate_post
from agent.publisher import save_post, git_push
from agent.topics import pick_random_topic


def main():
    parser = argparse.ArgumentParser(description="AI 学习博客文章生成 Agent")
    parser.add_argument("--topic", type=str, help="文章主题（不指定则随机选取）")
    parser.add_argument("--auto-push", action="store_true", help="生成后自动 git commit & push")
    parser.add_argument("--dry-run", action="store_true", help="只生成内容不写入文件")
    args = parser.parse_args()

    topic = args.topic or pick_random_topic()
    print(f"📝 正在生成文章：{topic}")

    filename, content = generate_post(topic)

    if args.dry_run:
        print(f"\n--- 文件名: {filename} ---\n")
        print(content)
        print("\n--- dry-run 模式，未写入文件 ---")
        return

    filepath = save_post(filename, content)
    print(f"✅ 文章已保存: {filepath}")

    if args.auto_push:
        print("🚀 正在推送到 GitHub...")
        git_push(filepath, topic)
        print("✅ 推送完成！")


if __name__ == "__main__":
    main()
