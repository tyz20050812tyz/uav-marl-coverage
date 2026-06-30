#!/bin/bash

set -u

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR" || exit 1

PORT="${UAV_MARL_PORT:-8600}"
HOST="${UAV_MARL_HOST:-127.0.0.1}"
LOCAL_ENV_FILE="$PROJECT_DIR/.env.local"
KNOWN_VENV="${UAV_MARL_VENV:-/Users/wayne/环境/uav_marl_venv}"
PROJECT_VENV="$PROJECT_DIR/.venv"

print_step() {
  printf "\n\033[1;36m==> %s\033[0m\n" "$1"
}

print_ok() {
  printf "\033[1;32m✓ %s\033[0m\n" "$1"
}

print_warn() {
  printf "\033[1;33m! %s\033[0m\n" "$1"
}

print_error() {
  printf "\033[1;31m✗ %s\033[0m\n" "$1"
}

pause_on_exit() {
  printf "\n按回车键关闭窗口..."
  read -r _
}

trap pause_on_exit EXIT

print_step "UAV MARL 一键启动"
echo "项目目录: $PROJECT_DIR"
echo "服务地址: http://$HOST:$PORT"

if [ -f "$LOCAL_ENV_FILE" ]; then
  # shellcheck disable=SC1090
  source "$LOCAL_ENV_FILE"
fi

print_step "检测 Python 虚拟环境"

if [ -x "$KNOWN_VENV/bin/python" ]; then
  VENV_DIR="$KNOWN_VENV"
  print_ok "检测到已有虚拟环境: $VENV_DIR"
elif [ -x "$PROJECT_VENV/bin/python" ]; then
  VENV_DIR="$PROJECT_VENV"
  print_ok "检测到项目本地虚拟环境: $VENV_DIR"
else
  VENV_DIR="$PROJECT_VENV"
  print_warn "未检测到可用虚拟环境，将创建: $VENV_DIR"
  if ! command -v python3 >/dev/null 2>&1; then
    print_error "未找到 python3，请先安装 Python 3.9+"
    exit 1
  fi
  python3 -m venv "$VENV_DIR" || exit 1
  print_ok "虚拟环境创建完成"
fi

PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

print_step "检查 Python 版本"
"$PYTHON" - <<'PY'
import sys
if sys.version_info < (3, 9):
    raise SystemExit("Python 版本过低，需要 3.9+")
print(f"Python {sys.version.split()[0]}")
PY
if [ $? -ne 0 ]; then
  print_error "Python 版本检查失败"
  exit 1
fi

print_step "安装 / 更新项目依赖"
"$PYTHON" -m pip install --upgrade pip
"$PIP" install -r requirements.txt
if [ $? -ne 0 ]; then
  print_error "依赖安装失败，请检查网络或 requirements.txt"
  exit 1
fi
print_ok "依赖已就绪"

print_step "检查 DeepSeek API Key"

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  print_warn "未检测到 DEEPSEEK_API_KEY"
  printf "请输入 DeepSeek API Key（输入时不会显示，留空则跳过 LLM 报告功能）: "
  stty -echo
  read -r INPUT_KEY
  stty echo
  printf "\n"

  if [ -n "$INPUT_KEY" ]; then
    export DEEPSEEK_API_KEY="$INPUT_KEY"
    {
      echo "# 本文件由 start_uav_marl.command 自动生成，请勿提交到公开仓库"
      echo "export DEEPSEEK_API_KEY=\"$INPUT_KEY\""
    } > "$LOCAL_ENV_FILE"
    chmod 600 "$LOCAL_ENV_FILE"
    print_ok "DeepSeek API Key 已保存到 .env.local"
  else
    print_warn "已跳过 DeepSeek 配置；训练和结构化总结仍可正常使用"
  fi
else
  print_ok "已检测到 DEEPSEEK_API_KEY"
fi

print_step "检查端口占用"
if command -v lsof >/dev/null 2>&1; then
  PORT_PID="$(lsof -ti tcp:"$PORT" 2>/dev/null || true)"
  if [ -n "$PORT_PID" ]; then
    print_warn "端口 $PORT 已被占用，进程 PID: $PORT_PID"
    printf "是否结束该进程并继续启动？[y/N]: "
    read -r KILL_OLD
    case "$KILL_OLD" in
      y|Y|yes|YES)
        kill $PORT_PID 2>/dev/null || true
        sleep 1
        print_ok "已尝试释放端口 $PORT"
        ;;
      *)
        print_error "端口被占用，已取消启动。你也可以设置 UAV_MARL_PORT 换端口。"
        exit 1
        ;;
    esac
  fi
fi

print_step "启动 Web 控制台"
echo "浏览器将打开: http://$HOST:$PORT"
if command -v open >/dev/null 2>&1; then
  (sleep 1.5 && open "http://$HOST:$PORT") >/dev/null 2>&1 &
fi

echo
echo "服务运行中。停止服务请在此窗口按 Ctrl+C。"
echo

"$PYTHON" web_server.py --host "$HOST" --port "$PORT"
