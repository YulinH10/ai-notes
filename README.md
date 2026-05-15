# AI学习笔记

每日 AI/ML 学习进度博客，使用 Hugo + PaperMod 搭建，配备 Claude Agent 自动生成文章。

## 快速开始

### 1. 安装依赖

```bash
brew install hugo    # macOS
make setup           # 创建 Python 虚拟环境并安装依赖
```

### 2. 配置 API Key

编辑 `.env` 文件，填入你的 Anthropic API Key：

```
ANTHROPIC_API_KEY=sk-ant-xxxxx
```

### 3. 生成文章

```bash
# 指定主题生成
make generate TOPIC="Transformer架构详解"

# 随机主题生成
make generate-random

# 生成并推送到 GitHub
make deploy-local TOPIC="今天学了注意力机制"
```

### 4. 本地预览

```bash
make serve
# 访问 http://localhost:1313/ai-notes/
```

## 自动化发布

GitHub Actions 每天北京时间 10:00 自动运行，随机选择 AI 话题生成文章并部署。

也可以在 GitHub Actions 页面手动触发，指定文章主题。

### 设置 GitHub Secrets

在仓库 Settings > Secrets and variables > Actions 中添加：

- `ANTHROPIC_API_KEY`: 你的 Anthropic API Key

## 项目结构

```
├── agent/          # 文章生成 Agent（Python）
├── config/         # Agent 配置
├── content/posts/  # 博客文章
├── themes/         # Hugo 主题
├── .github/        # GitHub Actions 工作流
└── hugo.toml       # Hugo 站点配置
```
