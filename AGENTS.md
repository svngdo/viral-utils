# Repository Guidelines

## Project Structure & Module Organization

- `api/src/` contains the FastAPI backend. `main.py` registers routers; `config.py` and `database.py` handle settings and database connections.
- `api/src/systems/` separates HTTP endpoints (`router.py`), validation (`schemas.py`), and SQL operations (`repository.py`). Follow this pattern for new domains; shared helpers belong in `api/src/utils/`.
- `api/alembic/versions/` stores database migrations.
- `web/src/` contains the React/TypeScript application, starting with `main.tsx` and `App.tsx`. Imported assets live in `web/src/assets/`; static public files belong in `web/public/`.

## Build, Test, and Development Commands

Run backend commands from `api/` and frontend commands from `web/`.

- `uv sync`: install Python dependencies, including development tools; Python 3.13+ is required.
- `make dev`: start FastAPI with reload at `localhost:8001`.
- `uv run alembic upgrade head`: apply database migrations to the configured database.
- `uv run ruff check .` and `uv run ruff format .`: lint and format Python.
- `uv run pytest`: run backend tests once added.
- `bun install`: install frontend dependencies.
- `bun run dev`: start the Vite development server.
- `bun run build`: run TypeScript checks and create the production bundle.
- `bun run lint`: run ESLint, including React Hooks rules.
- `bun run preview`: serve the built frontend locally.

## Coding Style & Naming Conventions

Use four-space indentation, double quotes, and Ruff's 88-character line target for Python. Use `snake_case` for functions/modules and `PascalCase` for schema classes. Keep database operations asynchronous and SQL parameters bound.

Follow existing TypeScript style: two-space indentation, single quotes, and no semicolons. Name React components in `PascalCase` and hooks with a `use` prefix.

## Testing Guidelines

Pytest is declared, but no test suite or coverage threshold currently exists. Add backend tests under `api/tests/` as `test_*.py`, covering successful operations, validation errors, and missing records. Use an isolated test database. The frontend has no test runner configured; run lint/build and manually verify affected interactions.

## Commit & Pull Request Guidelines

Follow the history's Conventional Commits style: `feat(api): implement systems CRUD`, `refactor(web): simplify components`, or `chore: clean up project`. Keep commits focused. PRs should explain behavior changes, link relevant issues, list validation performed, and include screenshots for UI changes. Call out migrations and configuration changes.

## Configuration & Secrets

Configure `DATABASE_DIALECT`, `DATABASE_DRIVER`, and `DATABASE_PATH` through backend environment variables or `api/.env`; settings also load `.env.prod`. Keep secrets and local database files out of commits.
