@echo off
setlocal enabledelayedexpansion

:: ============================================================
::  UAV MARL 一键启动 (Windows)
::  双击运行即可启动 Web 控制台
:: ============================================================

:: ----- 项目路径 -----
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%" || exit /b 1

:: ----- 端口与地址（可通过系统环境变量覆盖）-----
if "%UAV_MARL_PORT%"=="" (set "PORT=8600") else (set "PORT=%UAV_MARL_PORT%")
if "%UAV_MARL_HOST%"=="" (set "HOST=127.0.0.1") else (set "HOST=%UAV_MARL_HOST%")

set "LOCAL_ENV_FILE=%PROJECT_DIR%.env.local"
if "%UAV_MARL_VENV%"=="" (set "KNOWN_VENV=%USERPROFILE%\环境\uav_marl_venv") else (set "KNOWN_VENV=%UAV_MARL_VENV%")
set "PROJECT_VENV=%PROJECT_DIR%.venv"

:: ----- 辅助输出 -----
call :print_step "UAV MARL 一键启动"
echo 项目目录: %PROJECT_DIR%
echo 服务地址: http://%HOST%:%PORT%

:: ----- 加载 .env.local -----
if exist "%LOCAL_ENV_FILE%" (
    call :print_ok "检测到 .env.local，正在加载环境变量..."
    for /f "usebackq delims=" %%L in (`findstr /r /c:"^export " "%LOCAL_ENV_FILE%" 2^>nul`) do (
        set "line=%%L"
        set "line=!line:export =!"
        set "line=!line:"=!"
        for /f "tokens=1,* delims==" %%x in ("!line!") do (
            if not "%%x"=="" set "%%x=%%y"
        )
    )
) else (
    call :print_warn "未检测到 .env.local，将跳过"
)

:: ----- 检测 Python 虚拟环境 -----
call :print_step "检测 Python 虚拟环境"

if exist "%KNOWN_VENV%\Scripts\python.exe" (
    set "VENV_DIR=%KNOWN_VENV%"
    call :print_ok "检测到已有虚拟环境: !VENV_DIR!"
) else if exist "%PROJECT_VENV%\Scripts\python.exe" (
    set "VENV_DIR=%PROJECT_VENV%"
    call :print_ok "检测到项目本地虚拟环境: !VENV_DIR!"
) else (
    set "VENV_DIR=%PROJECT_VENV%"
    call :print_warn "未检测到可用虚拟环境，将创建: !VENV_DIR!"

    :: 查找 python
    where python >nul 2>&1
    if %errorlevel% neq 0 (
        where python3 >nul 2>&1
        if %errorlevel% neq 0 (
            call :print_error "未找到 python，请先安装 Python 3.9+"
            goto :exit_pause
        )
        set "PYTHON_BASE=python3"
    ) else (
        set "PYTHON_BASE=python"
    )

    call :print_step "正在创建虚拟环境..."
    !PYTHON_BASE! -m venv "!VENV_DIR!"
    if %errorlevel% neq 0 (
        call :print_error "虚拟环境创建失败"
        goto :exit_pause
    )
    call :print_ok "虚拟环境创建完成"
)

set "PYTHON=%VENV_DIR%\Scripts\python.exe"
set "PIP=%VENV_DIR%\Scripts\pip.exe"

:: ----- 检查 Python 版本 -----
call :print_step "检查 Python 版本"
"%PYTHON%" -c "import sys; v=sys.version_info; sys.exit(0 if v>=(3,9) else 1)"
if %errorlevel% neq 0 (
    call :print_error "Python 版本过低，需要 3.9+"
    goto :exit_pause
)
for /f "tokens=*" %%v in ('"%PYTHON%" -c "import sys; print(f'Python {sys.version.split()[0]}')"') do (
    call :print_ok "%%v"
)

:: ----- 安装 / 更新项目依赖 -----
call :print_step "安装 / 更新项目依赖"
"%PYTHON%" -m pip install --upgrade pip
"%PIP%" install -r "%PROJECT_DIR%requirements.txt"
if %errorlevel% neq 0 (
    call :print_error "依赖安装失败，请检查网络或 requirements.txt"
    goto :exit_pause
)
call :print_ok "依赖已就绪"

:: ----- 检查 DeepSeek API Key -----
call :print_step "检查 DeepSeek API Key"

if "%DEEPSEEK_API_KEY%"=="your_deepseek_api_key_here" set "DEEPSEEK_API_KEY="

if "%DEEPSEEK_API_KEY%"=="" (
    call :print_warn "未检测到 DEEPSEEK_API_KEY"

    :: 使用 PowerShell 安全读取密钥（输入时不显示）
    set "HAS_PS=0"
    powershell -Command "exit 0" >nul 2>&1 && set "HAS_PS=1"

    if "!HAS_PS!"=="1" (
        echo 请输入 DeepSeek API Key（输入时不会显示，留空则跳过 LLM 报告功能）:
        for /f "delims=" %%i in ('powershell -Command "$pwd=Read-Host -AsSecureString; [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($pwd))"') do set "INPUT_KEY=%%i"
    ) else (
        echo PowerShell 不可用，密钥输入将明文显示
        set /p INPUT_KEY="请输入 DeepSeek API Key（留空则跳过 LLM 报告功能）: "
    )

    if not "!INPUT_KEY!"=="" (
        set "DEEPSEEK_API_KEY=!INPUT_KEY!"
        (
            echo # 本文件由 start_uav_marl.bat 自动生成，请勿提交到公开仓库
            echo export DEEPSEEK_API_KEY="!INPUT_KEY!"
        ) > "%LOCAL_ENV_FILE%"
        call :print_ok "DeepSeek API Key 已保存到 .env.local"
    ) else (
        call :print_warn "已跳过 DeepSeek 配置；训练和结构化总结仍可正常使用"
    )
) else (
    call :print_ok "已检测到 DEEPSEEK_API_KEY"
)

:: ----- 检查端口占用 -----
call :print_step "检查端口占用"

set "PORT_PID="
for /f "tokens=5" %%a in ('netstat -ano ^| findstr /r ":%PORT% .*LISTENING" 2^>nul') do (
    set "PORT_PID=%%a"
    goto :found_pid
)
goto :no_port_conflict

:found_pid
call :print_warn "端口 %PORT% 已被占用，进程 PID: %PORT_PID%"
set /p KILL_OLD="是否结束该进程并继续启动？[y/N]: "
if /i "!KILL_OLD!"=="y" (
    taskkill /PID !PORT_PID! /F >nul 2>&1
    timeout /t 1 /nobreak >nul
    call :print_ok "已尝试释放端口 %PORT%"
) else if /i "!KILL_OLD!"=="yes" (
    taskkill /PID !PORT_PID! /F >nul 2>&1
    timeout /t 1 /nobreak >nul
    call :print_ok "已尝试释放端口 %PORT%"
) else (
    call :print_error "端口被占用，已取消启动。你也可以设置 UAV_MARL_PORT 换端口。"
    goto :exit_pause
)

:no_port_conflict

:: ----- 启动 Web 控制台 -----
call :print_step "启动 Web 控制台"
echo 浏览器将打开: http://%HOST%:%PORT%

:: 延迟打开浏览器，给服务启动留时间
start "" /b cmd /c "timeout /t 2 /nobreak >nul && start http://%HOST%:%PORT%"

echo.
echo 服务运行中。停止服务请在此窗口按 Ctrl+C。
echo.

"%PYTHON%" "%PROJECT_DIR%web_server.py" --host "%HOST%" --port "%PORT%"

goto :exit_pause

:: ============================================================
::  辅助函数
:: ============================================================

:print_step
echo.
echo =^> %~1
exit /b 0

:print_ok
echo [OK] %~1
exit /b 0

:print_warn
echo [!!] %~1
exit /b 0

:print_error
echo [XX] %~1
exit /b 0

:exit_pause
echo.
echo 按回车键关闭窗口...
pause >nul
exit /b 0
