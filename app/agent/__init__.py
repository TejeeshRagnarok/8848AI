"""LLM ordering agent: provider abstraction, tools, sessions and the agent loop.

The model only ever acts through the three tools in ``tools.py``, which call
the existing backend services.  It has no SQL, HTTP, filesystem or code
execution access, and it cannot place orders: orders are placed only after
explicit customer confirmation handled by ``service.py``.
"""
