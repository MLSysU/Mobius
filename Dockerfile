# 使用官方的 miniconda 镜像作为基础
FROM continuumio/miniconda3

# 将环境文件复制到容器中
COPY environment.yml /tmp/environment.yml

# 创建 conda 环境并激活它
RUN conda env create -f /tmp/environment.yml && conda clean -a

# 激活环境所需的默认 shell 设置
SHELL ["conda", "run", "-n", "mobius", "/bin/bash", "-c"]

# 设置默认命令（可替换）
CMD ["python"]

