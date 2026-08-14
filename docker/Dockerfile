FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

RUN mkdir -p /app/data
EXPOSE 8000

# 默认使用内置的离线 DemoProvider，开箱即用；
# 生产环境通过 CORTEX_CONFIG 指向真实模型配置（见 examples/config.example.json）。
CMD ["cortex", "serve", "--host", "0.0.0.0", "--port", "8000"]
