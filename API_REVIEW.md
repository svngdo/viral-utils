# API review

Reviewed on 2026-10-08 at commit `3b6f184`. Report only; no implementation changes.

The router/repository split is clear, SQL parameters are bound safely, and the current structure is sufficient for this small API.

## Fix first

| Priority | Issue | Suggested improvement |
| --- | --- | --- |
| High | Frontend updates use `PUT`, but the API accepts only `PATCH`, so editing returns 405. Frontend IDs are also typed as numbers instead of UUID strings. | Align the method and ID types across API and frontend. See `api/src/systems/router.py` and `web/src/pages/Systems/`. |
| Medium | Duplicate system names cause an unhandled database error and return 500. | Handle unique-name conflicts on create and update with a clear 409 response. See `api/src/systems/repository.py`. |
| Medium | Updating a system never changes `updated_at`; its database default only runs on insert. | Set the timestamp in the update query. See `api/src/systems/repository.py`. |
| Medium | PATCH requires both name and description. A name-only request returns 422. | Allow omitted fields and preserve their stored values, or use PUT for full replacement. See `api/src/systems/schemas.py`. |
| Medium | The API returns only 100 systems by default, while the UI fetches once and searches locally. Later records are hidden. Negative limits are also accepted. | Add frontend pagination, validate `1 <= limit <= 100` and `offset >= 0`, and add an ID tie-breaker to the database sort. |

## Improve next

- **Configuration:** `config.py` always loads `.env.prod` after `.env`. Select the environment explicitly so local runs cannot unexpectedly use production settings. Add a safe `.env.example`.
- **Setup docs:** Add the missing `api/README.md` referenced by `pyproject.toml`. Document database settings, `uv sync`, `uv run alembic upgrade head`, and `make dev`. State that migrations are handwritten while Alembic has `target_metadata = None`.
- **Tests:** Add focused tests for CRUD, duplicate names, partial updates, pagination, timestamps, migration upgrade/downgrade, and Vietnamese search normalization. No test files were found.
- **Database lifecycle:** Dispose of the async engine during application shutdown in `main.py`.
- **Small cleanup:** Remove debug prints, return 201 when creating a system, and use `description: str | None = None` if descriptions should be optional on creation.
- **Unused code:** `constants.py` has no callers. `search_text` is stored and returned but never queried. Keep these only when their intended behavior is needed; consider server-side search when adding pagination.

## Validation

Ruff lint and formatting checks passed. Earlier temporary SQLite checks on this same code passed migration upgrade/downgrade and basic create/update/delete, and reproduced duplicate-name 500s, partial-update 422s, unchanged timestamps, and accepted negative limits. No full automated test suite was available.
