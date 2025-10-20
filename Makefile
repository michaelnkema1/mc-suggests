.PHONY: run dev stop

run:
	./run.sh

dev:
	PORT=8001 ./run.sh

stop:
	pkill -f "uvicorn api.main:app" || true


