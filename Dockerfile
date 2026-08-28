# 三体计算星座开放科学平台 应用镜像 Dockerfile
# 注意：FROM 的基础镜像需要与平台分配的 base.image 保持一致。
# 在星载开发环境中查看 /home/spaceapp/project/base.image 获取准确地址。
FROM 10.107.104.55:8082/base/base:base

WORKDIR /workspace

# 离线安装 Python 依赖：wheels/ 目录由开发环境 pip download 预先下载好
# （平台容器内无法访问外网源，必须用 --no-index 离线安装）
COPY requirements.txt .
COPY wheels/ /tmp/wheels/
RUN pip3 install --no-cache-dir --no-index --find-links=/tmp/wheels -r requirements.txt \
    && rm -rf /tmp/wheels

# 复制推理代码、启动脚本与模型
COPY code-server/ /workspace/code-server/
COPY models/ /workspace/models/
COPY run.sh /workspace/run.sh

RUN chmod +x /workspace/run.sh

# 应用默认启动命令由 YAML 中的 cmd 指定，此处仅声明工作目录
CMD ["/workspace/run.sh"]
