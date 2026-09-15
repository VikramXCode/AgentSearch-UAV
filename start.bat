@echo off
set "PROJECT_DIR=%~dp0"

echo Starting Backend API and Frontend UI in 2 terminals...

start "AgentSearch Backend" powershell -NoExit -Command "cd '%PROJECT_DIR%'; & 'C:\v\agentsearch-venv\Scripts\Activate.ps1'; python -m api.main"
start "AgentSearch Frontend" powershell -NoExit -Command "cd '%PROJECT_DIR%\frontend'; npm run dev"
