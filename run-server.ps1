# Simple PowerShell script to run server - press Ctrl+C or close window to stop
param(
    [switch]$Dev
)

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Error: Virtual environment not found." -ForegroundColor Red
    exit 1
}

if ($Dev) {
    Write-Host "Starting server in DEV mode (auto-reload)..." -ForegroundColor Green
    Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
    & .\.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload --workers 1
} else {
    Write-Host "Starting server..." -ForegroundColor Green
    Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
    & .\.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
}
