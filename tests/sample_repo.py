"""A tiny fake repository whose functions have clearly different purposes, so tests can
check that a question retrieves the right one."""

SAMPLE_FILES: dict[str, bytes] = {
    "billing/invoice.py": b'''\
def calculate_invoice_total(items, tax_rate):
    """Add up line item prices and apply sales tax to produce the invoice total."""
    subtotal = sum(item.price * item.quantity for item in items)
    return round(subtotal * (1 + tax_rate), 2)
''',
    "auth/passwords.py": b'''\
import hashlib
import os


def hash_password(password):
    """Hash a user's password with a random salt before storing it in the database."""
    salt = os.urandom(16)
    return salt + hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
''',
    "net/retry.py": b'''\
import time


def retry_with_backoff(operation, attempts=5):
    """Retry a failing network request, waiting exponentially longer between attempts."""
    for attempt in range(attempts):
        try:
            return operation()
        except ConnectionError:
            time.sleep(2**attempt)
    raise ConnectionError("gave up")
''',
    "README.md": b"# Shop\n\nA small online shop backend.\n",
    "config.py": b"STRIPE_KEY = 'sk_live_" + b"a" * 24 + b"'\n",
}

# Expected best match for each natural-language question.
QUESTIONS: dict[str, tuple[str, str]] = {
    "How is the tax added to the bill?": ("billing/invoice.py", "def calculate_invoice_total"),
    "Where are user passwords hashed?": ("auth/passwords.py", "def hash_password"),
    "What happens when a network call fails?": ("net/retry.py", "def retry_with_backoff"),
}
