"""Row counts scoped to test data.

Live tests run against the development database, which may also hold real
(demo) records.  Every test restaurant's slug starts with ``test-``, so
counting only their orders keeps assertions independent of other data.
"""

from sqlalchemy import Connection, func, select

from app.models import Order, Payment, Restaurant

_TEST_RESTAURANTS = select(Restaurant.id).where(Restaurant.slug.like("test-%"))
_TEST_ORDERS = select(Order.id).where(Order.restaurant_id.in_(_TEST_RESTAURANTS))


def count_test_orders(connection: Connection) -> int:
    return connection.execute(
        select(func.count()).select_from(Order).where(Order.restaurant_id.in_(_TEST_RESTAURANTS))
    ).scalar_one()


def count_test_payments(connection: Connection) -> int:
    return connection.execute(
        select(func.count()).select_from(Payment).where(Payment.order_id.in_(_TEST_ORDERS))
    ).scalar_one()
