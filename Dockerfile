# 三体计算星座开放科学平台 应用镜像 Dockerfile
# 注意：FROM 的基础镜像需要与平台分配的 base.image 保持一致。
# 在星载开发环境中查看 /home/spaceapp/project/base.image 获取准确地址。
FROM 10.107.104.55:8082/base/base:base

WORKDIR /workspace

# 离线安装 Python 依赖：wheels/ 目录由开发环境 pip download 预先下载好
# （平台容器内无法访问外网源，必须用 --no-index 离线安装）
# 安装 Python 依赖：离线 wheels + 强制重装
# 平台的差分包合并机制不会带出基础镜像里的已有包，
# 因此所有依赖必须 --force-reinstall --no-deps 显式写入差分层。
# 构建前需在开发环境执行: pip3 download -r requirements.txt -d wheels/
COPY requirements.txt .
COPY wheels/ /tmp/wheels/
RUN pip3 install --no-cache-dir --no-index --force-reinstall --no-deps \
    /tmp/wheels/pandas-*.whl \
    /tmp/wheels/rasterio-*.whl \
    /tmp/wheels/tqdm-*.whl \
    /tmp/wheels/numpy-*.whl \
    /tmp/wheels/affine-*.whl \
    /tmp/wheels/attrs-*.whl \
    /tmp/wheels/certifi-*.whl \
    /tmp/wheels/click-*.whl \
    /tmp/wheels/click_plugins-*.whl \
    /tmp/wheels/cligj-*.whl \
    /tmp/wheels/pyparsing-*.whl \
    /tmp/wheels/python_dateutil-*.whl \
    /tmp/wheels/pytz-*.whl \
    /tmp/wheels/tzdata-*.whl \
    /tmp/wheels/six-*.whl \
    && rm -rf /tmp/wheels
RUN chmod +x /workspace/run.sh

# 应用默认启动命令由 YAML 中的 cmd 指定，此处仅声明工作目录
CMD ["/workspace/run.sh"]
