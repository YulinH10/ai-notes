.PHONY: setup generate generate-random serve deploy-local

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	cp -n .env.example .env || true
	@echo "✅ 环境已就绪。请编辑 .env 填入你的 ANTHROPIC_API_KEY"

generate:
	.venv/bin/python -m agent.agent --topic "$(TOPIC)"

generate-random:
	.venv/bin/python -m agent.agent

serve:
	hugo server -D

deploy-local:
	.venv/bin/python -m agent.agent --topic "$(TOPIC)" --auto-push
