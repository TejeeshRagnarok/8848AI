"""Live database integration check.

Reads DATABASE_URL from .env and verifies:
1. SQLAlchemy engine connects to PostgreSQL.
2. SELECT 1 succeeds.
3. Alembic config loads correctly.

Run with::

    .venv\\Scripts\\python.exe tests\\live_db_check.py
"""

import sys
import os

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings


def main() -> None:
    if not settings.database_url:
        print("SKIP: DATABASE_URL is not configured in .env")
        print("      Copy .env.example to .env and set the password, then re-run.")
        sys.exit(0)

    # Mask credentials for display
    from sqlalchemy import make_url
    url_obj = make_url(settings.database_url)
    safe_url = f"{url_obj.drivername}://{url_obj.username}:***@{url_obj.host}:{url_obj.port}/{url_obj.database}"
    print(f"Connecting to: {safe_url}")

    # Test 1: Engine creation + SELECT 1, using the application's own engine
    from sqlalchemy import text

    from app.database import get_engine
    engine = get_engine()
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            row = result.scalar()
            assert row == 1, f"Expected 1, got {row}"
            print("  [OK] SELECT 1 succeeded")

            # Check server version
            ver = conn.execute(text("SHOW server_version")).scalar()
            print(f"  [OK] PostgreSQL server version: {ver}")

            # Check current database and user
            db = conn.execute(text("SELECT current_database()")).scalar()
            user = conn.execute(text("SELECT current_user")).scalar()
            print(f"  [OK] Connected as '{user}' to database '{db}'")
    except Exception as e:
        print(f"  [FAIL] Connection error: {type(e).__name__}: {e}")
        sys.exit(1)
    finally:
        engine.dispose()

    # Test 2: Alembic config loads
    try:
        from alembic.config import Config
        alembic_cfg = Config("alembic.ini")
        script_loc = alembic_cfg.get_main_option("script_location")
        print(f"  [OK] Alembic config loaded (script_location={script_loc})")
    except Exception as e:
        print(f"  [FAIL] Alembic config error: {type(e).__name__}: {e}")
        sys.exit(1)

    print("\nAll live checks passed.")


if __name__ == "__main__":
    main()
