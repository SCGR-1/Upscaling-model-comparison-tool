.PHONY: run dev install help stop

run:
	@powershell -ExecutionPolicy Bypass -File run-server.ps1

dev:
	@powershell -ExecutionPolicy Bypass -File run-server.ps1 -Dev

stop:
	@powershell -Command "Get-Process | Where-Object {$$_.ProcessName -eq 'python'} | Where-Object {(Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue).OwningProcess -eq $$_.Id} | Stop-Process -Force; netstat -ano | findstr :8000 | ForEach-Object { if ($$_ -match '\s+(\d+)\s*$$') { Stop-Process -Id $$matches[1] -Force -ErrorAction SilentlyContinue } }; Write-Host 'Server stopped.'"

install:
	@.venv/Scripts/python.exe -m pip install -r requirements.txt

help:
	@echo Available commands:
	@echo   make run    - Start the server
	@echo   make dev    - Start the server with auto-reload
	@echo   make stop   - Stop the server
	@echo   make install - Install dependencies
	@echo   make help   - Show this help message

