$ProjectDir = $PSScriptRoot
$FrontendDir = Join-Path $ProjectDir "frontend"
$VenvActivate = "C:\v\agentsearch-venv\Scripts\Activate.ps1"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   Starting AgentSearch Backend & Frontend" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Terminal 1: Backend API
Write-Host "Starting Backend API in Terminal 1..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ProjectDir'; & '$VenvActivate'; python -m api.main"

# Terminal 2: Frontend UI
Write-Host "Starting Frontend UI in Terminal 2..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$FrontendDir'; npm run dev"

Write-Host "Done! Both terminals launched." -ForegroundColor Cyan
