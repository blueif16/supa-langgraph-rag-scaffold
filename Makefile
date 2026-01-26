.PHONY: dev install migrate seed test clean

dev:
	docker-compose up --build

install:
	cd backend && poetry install

migrate:
	supabase db push

seed:
	supabase db seed

test:
	cd backend && poetry run pytest

clean:
	docker-compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
