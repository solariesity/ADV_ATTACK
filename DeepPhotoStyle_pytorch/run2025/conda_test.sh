#!/bin/bash

# 测试到清华镜像源的基础网络连通性 (ICMP)
ping -c 4 mirrors.tuna.tsinghua.edu.cn

# 测试到清华镜像源的HTTP端口连通性 (TCP)
curl -v --connect-timeout 10 https://mirrors.tuna.tsinghua.edu.cn 2>&1 | head -20

# 查看当前的conda配置
conda info

# 查看conda的详细配置，特别是频道和代理设置
conda config --show

# 查看 channels 优先级配置
conda config --show channels

# 查看当前环境的所有环境变量，过滤出可能影响网络的变量
env | grep -i proxy
env | grep -i http

# 检查系统级别的代理配置
echo "http_proxy: $http_proxy"
echo "https_proxy: $https_proxy" 
echo "HTTP_PROXY: $HTTP_PROXY"
echo "HTTPS_PROXY: $HTTPS_PROXY"

# 检查是否有conda特定的配置
cat ~/.condarc 2>/dev/null || echo "No .condarc file found"

# 测试直接访问conda-forge的repodata
curl -v https://conda.anaconda.org/conda-forge/linux-64/repodata.json 2>&1 | head -10

# 测试清华镜像源的访问
curl -v https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/linux-64/repodata.json 2>&1 | head -10

# 检查域名解析是否正确
nslookup mirrors.tuna.tsinghua.edu.cn
nslookup conda.anaconda.org