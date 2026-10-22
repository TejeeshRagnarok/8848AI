"""Migration checks: revision chain and the SQL each migration emits.

SQL is rendered in Alembic's offline mode, so these tests never connect to
(or change) a database.
"""

import io
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE = "5900f7d42043"
MENU_REVISION = "25620f605c18"
ORDERS_REVISION = "09b7b2a79c59"
INVENTORY_REVISION = "802abd40fb6d"


def _config(buffer: io.StringIO | None = None) -> Config:
    cfg = Config(str(PROJECT_ROOT / "alembic.ini"), output_buffer=buffer)
    # Offline rendering only needs a dialect; env.py prefers DATABASE_URL if set.
    cfg.set_main_option("sqlalchemy.url", "postgresql+psycopg://offline@localhost/offline")
    return cfg


def _offline_sql(direction: str, revision_range: str) -> str:
    buffer = io.StringIO()
    getattr(command, direction)(_config(buffer), revision_range, sql=True)
    return buffer.getvalue()


@pytest.fixture(scope="module")
def script() -> ScriptDirectory:
    return ScriptDirectory.from_config(_config())


class TestRevisionChain:
    def test_single_head_is_inventory_revision(self, script: ScriptDirectory) -> None:
        assert script.get_heads() == [INVENTORY_REVISION]

    def test_inventory_revision_follows_orders_revision(self, script: ScriptDirectory) -> None:
        assert script.get_revision(INVENTORY_REVISION).down_revision == ORDERS_REVISION

    def test_orders_revision_follows_menu_revision(self, script: ScriptDirectory) -> None:
        assert script.get_revision(ORDERS_REVISION).down_revision == MENU_REVISION

    def test_menu_revision_follows_baseline(self, script: ScriptDirectory) -> None:
        assert script.get_revision(MENU_REVISION).down_revision == BASELINE

    def test_baseline_is_root(self, script: ScriptDirectory) -> None:
        assert script.get_revision(BASELINE).down_revision is None

    def test_linear_history(self, script: ScriptDirectory) -> None:
        revisions = [r.revision for r in script.walk_revisions()]
        assert revisions == [INVENTORY_REVISION, ORDERS_REVISION, MENU_REVISION, BASELINE]


class TestMenuMigrationSql:
    @pytest.fixture(scope="class")
    def upgrade_sql(self) -> str:
        return _offline_sql("upgrade", f"{BASELINE}:{MENU_REVISION}")

    @pytest.fixture(scope="class")
    def downgrade_sql(self) -> str:
        return _offline_sql("downgrade", f"{MENU_REVISION}:{BASELINE}")

    def test_upgrade_creates_both_tables(self, upgrade_sql: str) -> None:
        assert "CREATE TABLE restaurants" in upgrade_sql
        assert "CREATE TABLE menu_items" in upgrade_sql
        assert upgrade_sql.index("CREATE TABLE restaurants") < upgrade_sql.index(
            "CREATE TABLE menu_items"
        )

    def test_upgrade_uses_uuid_keys(self, upgrade_sql: str) -> None:
        assert "id UUID NOT NULL" in upgrade_sql
        assert "restaurant_id UUID NOT NULL" in upgrade_sql

    @pytest.mark.parametrize(
        "fragment",
        [
            "CONSTRAINT uq_restaurants_slug UNIQUE (slug)",
            "CONSTRAINT ck_menu_items_price_paise_positive CHECK (price_paise > 0)",
            "CONSTRAINT fk_menu_items_restaurant_id_restaurants FOREIGN KEY(restaurant_id) "
            "REFERENCES restaurants (id) ON DELETE RESTRICT",
            "CREATE INDEX ix_menu_items_restaurant_id ON menu_items (restaurant_id)",
            "timezone VARCHAR(64) DEFAULT 'Asia/Kolkata' NOT NULL",
            "currency VARCHAR(3) DEFAULT 'INR' NOT NULL",
            "is_active BOOLEAN DEFAULT true NOT NULL",
            "is_available BOOLEAN DEFAULT true NOT NULL",
            "created_at TIMESTAMP WITH TIME ZONE DEFAULT statement_timestamp() NOT NULL",
        ],
    )
    def test_upgrade_schema_fragments(self, upgrade_sql: str, fragment: str) -> None:
        assert fragment in upgrade_sql

    def test_upgrade_is_additive_only(self, upgrade_sql: str) -> None:
        assert "DROP" not in upgrade_sql
        assert "ALTER TABLE" not in upgrade_sql

    def test_downgrade_reverses_only_this_migration(self, downgrade_sql: str) -> None:
        drops = [line for line in downgrade_sql.splitlines() if line.startswith("DROP")]
        assert drops == [
            "DROP INDEX ix_menu_items_restaurant_id;",
            "DROP TABLE menu_items;",
            "DROP TABLE restaurants;",
        ]
        assert f"version_num='{BASELINE}'" in downgrade_sql


class TestOrdersMigrationSql:
    @pytest.fixture(scope="class")
    def upgrade_sql(self) -> str:
        return _offline_sql("upgrade", f"{MENU_REVISION}:{ORDERS_REVISION}")

    @pytest.fixture(scope="class")
    def downgrade_sql(self) -> str:
        return _offline_sql("downgrade", f"{ORDERS_REVISION}:{MENU_REVISION}")

    def test_upgrade_creates_sequence_before_tables(self, upgrade_sql: str) -> None:
        seq = upgrade_sql.index("CREATE SEQUENCE order_number_seq START WITH 100001")
        orders = upgrade_sql.index("CREATE TABLE orders")
        items = upgrade_sql.index("CREATE TABLE order_items")
        assert seq < orders < items

    @pytest.mark.parametrize(
        "fragment",
        [
            "order_number VARCHAR(20) DEFAULT 'ORD-' || nextval('order_number_seq') NOT NULL",
            "status VARCHAR(20) DEFAULT 'pending' NOT NULL",
            "subtotal_paise BIGINT NOT NULL",
            "total_paise BIGINT NOT NULL",
            "menu_item_id UUID,",  # nullable reference
            "CONSTRAINT ck_orders_status_valid CHECK (status IN ('pending', 'confirmed', "
            "'preparing', 'ready', 'completed', 'cancelled'))",
            "CONSTRAINT ck_orders_subtotal_paise_nonnegative CHECK (subtotal_paise >= 0)",
            "CONSTRAINT ck_orders_total_paise_nonnegative CHECK (total_paise >= 0)",
            "CONSTRAINT uq_orders_order_number UNIQUE (order_number)",
            "CONSTRAINT uq_orders_restaurant_id_idempotency_key UNIQUE "
            "(restaurant_id, idempotency_key)",
            "REFERENCES restaurants (id) ON DELETE RESTRICT",
            "REFERENCES orders (id) ON DELETE CASCADE",
            "REFERENCES menu_items (id) ON DELETE SET NULL",
            "CONSTRAINT ck_order_items_quantity_positive CHECK (quantity > 0)",
            "CONSTRAINT ck_order_items_line_total_matches CHECK "
            "(line_total_paise = unit_price_paise * quantity)",
            "CREATE INDEX ix_orders_restaurant_id_created_at ON orders (restaurant_id, created_at)",
            "CREATE INDEX ix_order_items_order_id ON order_items (order_id)",
            "CREATE INDEX ix_order_items_menu_item_id ON order_items (menu_item_id)",
        ],
    )
    def test_upgrade_schema_fragments(self, upgrade_sql: str, fragment: str) -> None:
        assert fragment in upgrade_sql

    def test_upgrade_does_not_touch_existing_tables(self, upgrade_sql: str) -> None:
        assert "DROP" not in upgrade_sql
        assert "ALTER TABLE" not in upgrade_sql
        assert "CREATE TABLE restaurants" not in upgrade_sql
        assert "CREATE TABLE menu_items" not in upgrade_sql

    def test_downgrade_reverses_only_this_migration(self, downgrade_sql: str) -> None:
        drops = [line for line in downgrade_sql.splitlines() if line.startswith("DROP")]
        assert drops == [
            "DROP INDEX ix_order_items_order_id;",
            "DROP INDEX ix_order_items_menu_item_id;",
            "DROP TABLE order_items;",
            "DROP INDEX ix_orders_restaurant_id_created_at;",
            "DROP TABLE orders;",
            "DROP SEQUENCE order_number_seq;",
        ]
        assert f"version_num='{MENU_REVISION}'" in downgrade_sql


class TestInventoryMigrationSql:
    @pytest.fixture(scope="class")
    def upgrade_sql(self) -> str:
        return _offline_sql("upgrade", f"{ORDERS_REVISION}:{INVENTORY_REVISION}")

    @pytest.fixture(scope="class")
    def downgrade_sql(self) -> str:
        return _offline_sql("downgrade", f"{INVENTORY_REVISION}:{ORDERS_REVISION}")

    @pytest.mark.parametrize(
        "fragment",
        [
            "CREATE TABLE inventory_items",
            "CONSTRAINT ck_inventory_items_reserved_within_on_hand CHECK (reserved <= on_hand)",
            "CONSTRAINT ck_inventory_items_on_hand_nonnegative CHECK (on_hand >= 0)",
            "REFERENCES menu_items (id) ON DELETE CASCADE",
            "CREATE TABLE inventory_movements",
            "CREATE UNIQUE INDEX uq_inventory_movements_order_item_kind ON inventory_movements "
            "(order_id, menu_item_id, kind) WHERE order_id IS NOT NULL",
            "CREATE TABLE payments",
            "CONSTRAINT uq_payments_order_id UNIQUE (order_id)",
            "CONSTRAINT ck_payments_mode_demo_only CHECK (mode = 'demo')",
            "CONSTRAINT ck_payments_amount_paise_positive CHECK (amount_paise > 0)",
            "ALTER TABLE orders ADD COLUMN inventory_state VARCHAR(20) DEFAULT 'none' NOT NULL",
            "ALTER TABLE orders ADD COLUMN reservation_expires_at TIMESTAMP WITH TIME ZONE",
            "UPDATE orders SET cancel_reason = 'staff' WHERE status = 'cancelled'",
            "CONSTRAINT ck_orders_cancel_reason_iff_cancelled CHECK "
            "((status = 'cancelled') = (cancel_reason IS NOT NULL))",
        ],
    )
    def test_upgrade_schema_fragments(self, upgrade_sql: str, fragment: str) -> None:
        assert fragment in upgrade_sql

    def test_upgrade_never_drops(self, upgrade_sql: str) -> None:
        assert "DROP" not in upgrade_sql

    def test_downgrade_reverses_only_this_migration(self, downgrade_sql: str) -> None:
        assert "DROP TABLE payments;" in downgrade_sql
        assert "DROP TABLE inventory_movements;" in downgrade_sql
        assert "DROP TABLE inventory_items;" in downgrade_sql
        assert "DROP TABLE orders" not in downgrade_sql
        assert f"version_num='{ORDERS_REVISION}'" in downgrade_sql
