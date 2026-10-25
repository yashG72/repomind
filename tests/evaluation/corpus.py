"""Two small but realistic repositories used by the retrieval evaluation.

`octo/shop` is the repository the questions are asked about. `octo/blog` is indexed into the
same collection to test repository isolation: it deliberately reuses some names
(`hash_password`, `RateLimiter`) and has code that matches questions about the shop.
"""

import io
import subprocess
import tarfile
from pathlib import Path

SHOP: dict[str, bytes] = {
    "shop/billing/invoice.py": b'''\
from decimal import ROUND_HALF_UP, Decimal

from shop.billing.errors import InvoiceNotFoundError


def calculate_invoice_total(items, tax_rate):
    """Add up line item prices and apply sales tax to produce the invoice total."""
    subtotal = sum(Decimal(item.price) * item.quantity for item in items)
    total = subtotal * (1 + Decimal(tax_rate))
    return total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def apply_discount_code(total, code, codes):
    """Reduce a total by a percentage discount code; unknown codes leave it unchanged."""
    percent = codes.get(code.upper(), 0)
    return total * (100 - percent) / 100


def load_invoice(db, invoice_id):
    row = db.fetch_one("SELECT * FROM invoices WHERE id = ?", invoice_id)
    if row is None:
        raise InvoiceNotFoundError(invoice_id)
    return row
''',
    "shop/billing/errors.py": b'''\
class BillingError(Exception):
    """Base class for billing failures."""


class InvoiceNotFoundError(BillingError):
    def __init__(self, invoice_id):
        super().__init__(f"Invoice {invoice_id} does not exist")


class PaymentDeclined(BillingError):
    """The payment provider refused the charge."""
''',
    "shop/billing/payments.py": b'''\
import httpx

from shop.billing.errors import PaymentDeclined
from shop.net.retry import retry_with_backoff


class PaymentGateway:
    """Charges customer cards through the external payment provider's HTTP API."""

    def __init__(self, api_key, base_url="https://payments.example.com"):
        self._client = httpx.Client(base_url=base_url, headers={"Authorization": api_key})

    def charge(self, card_token, amount_cents, currency="usd"):
        response = retry_with_backoff(
            lambda: self._client.post(
                "/charges", json={"card": card_token, "amount": amount_cents, "currency": currency}
            )
        )
        if response.status_code == 402:
            raise PaymentDeclined("payment provider declined the card")
        response.raise_for_status()
        return response.json()["charge_id"]
''',
    "shop/auth/passwords.py": b'''\
import hashlib
import hmac
import os

ITERATIONS = 310_000


def hash_password(password):
    """Hash a user's password with a random salt before storing it in the database."""
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return salt + digest


def verify_password(password, stored):
    """Check a login attempt against the stored salted hash in constant time."""
    salt, expected = stored[:16], stored[16:]
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return hmac.compare_digest(digest, expected)
''',
    "shop/auth/sessions.py": b'''\
import secrets
import time

SESSION_TTL_SECONDS = 3600


class SessionStore:
    """Keeps logged-in users' sessions in memory, keyed by a random token."""

    def __init__(self):
        self._sessions = {}

    def create(self, user_id):
        token = secrets.token_urlsafe(32)
        self._sessions[token] = (user_id, time.time() + SESSION_TTL_SECONDS)
        return token

    def purge_expired(self):
        """Delete every session whose time-to-live has passed."""
        now = time.time()
        for token in [t for t, (_, expires) in self._sessions.items() if expires < now]:
            del self._sessions[token]
''',
    "shop/net/retry.py": b'''\
import random
import time


def retry_with_backoff(operation, attempts=5, base_delay=0.5):
    """Retry a failing network request, waiting exponentially longer between attempts."""
    for attempt in range(attempts):
        try:
            return operation()
        except ConnectionError:
            time.sleep(base_delay * 2**attempt + random.random() / 10)
    raise ConnectionError("gave up after repeated connection failures")
''',
    "shop/net/rate_limit.py": b'''\
import threading
import time


class RateLimiter:
    """Token bucket: allows at most `rate` calls per second, with short bursts up to `burst`."""

    def __init__(self, rate, burst):
        self.rate = rate
        self.burst = burst
        self._tokens = burst
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            self._tokens = min(self.burst, self._tokens + (now - self._updated) * self.rate)
            self._updated = now
            if self._tokens < 1:
                return False
            self._tokens -= 1
            return True
''',
    "shop/api/webhooks.py": b'''\
import hashlib
import hmac

from shop.billing.invoice import load_invoice


def verify_signature(secret, payload, signature_header):
    """Reject webhook calls whose HMAC signature does not match the shared secret."""
    expected = hmac.new(secret, payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header)


def handle_payment_webhook(db, secret, payload, signature_header, event):
    if not verify_signature(secret, payload, signature_header):
        return 401
    invoice = load_invoice(db, event["invoice_id"])
    db.execute("UPDATE invoices SET paid = 1 WHERE id = ?", invoice["id"])
    return 200
''',
    "shop/storage/uploads.py": b'''\
import os
import uuid

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_EXTENSIONS = {".png", ".jpg", ".pdf"}


def save_upload(directory, filename, data):
    """Store a user-uploaded file under a random name after checking its size and type."""
    extension = os.path.splitext(filename)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("file type not allowed")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("upload is too large")
    path = os.path.join(directory, uuid.uuid4().hex + extension)
    with open(path, "wb") as handle:
        handle.write(data)
    return path
''',
    "shop/config.py": b"""\
import os

DATABASE_URL = os.environ.get("SHOP_DATABASE_URL", "sqlite:///shop.db")
PAYMENT_API_KEY = os.environ.get("SHOP_PAYMENT_API_KEY", "")
DEFAULT_TAX_RATE = float(os.environ.get("SHOP_TAX_RATE", "0.08"))
""",
    "web/cart.js": b"""\
// Shopping cart state kept in the browser's localStorage.
export function addToCart(cart, product, quantity = 1) {
  const existing = cart.items.find((item) => item.id === product.id);
  if (existing) {
    existing.quantity += quantity;
  } else {
    cart.items.push({ id: product.id, price: product.price, quantity });
  }
  localStorage.setItem("cart", JSON.stringify(cart));
  return cart;
}
""",
    "README.md": b"""\
# Shop

A small online shop backend: invoices, card payments, user accounts and file uploads.

## Running

Set `SHOP_DATABASE_URL` and `SHOP_PAYMENT_API_KEY`, then start the server.
""",
}

BLOG: dict[str, bytes] = {
    "blog/users.py": b'''\
import bcrypt


def hash_password(password):
    """Blog accounts use bcrypt to hash passwords."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())
''',
    "blog/throttle.py": b'''\
import time


class RateLimiter:
    """Fixed-window limiter for comment posting."""

    def __init__(self, per_minute):
        self.per_minute = per_minute
        self.window = int(time.time() // 60)
        self.count = 0

    def allow(self):
        window = int(time.time() // 60)
        if window != self.window:
            self.window, self.count = window, 0
        self.count += 1
        return self.count <= self.per_minute
''',
    "blog/feed.py": b'''\
def render_rss_feed(posts):
    """Build the RSS XML feed of the latest blog posts."""
    items = "".join(f"<item><title>{p.title}</title></item>" for p in posts[:20])
    return f"<rss><channel>{items}</channel></rss>"
''',
}


# RepoMind's own source code at the Phase 3 merge (plus the CPU-only PyTorch change), read
# from this git repository. A real codebase of about 320 chunks, pinned to one commit so the
# evaluation is repeatable.
REPOMIND_SNAPSHOT_COMMIT = "5711c5613e83c3a565a0c549e7a40ea5fd01720a"


def load_repomind_snapshot() -> dict[str, bytes] | None:
    """Every regular file of the pinned commit, or None if this clone does not have it."""
    root = Path(__file__).resolve().parents[2]
    try:
        archive = subprocess.run(  # noqa: S603 (fixed arguments, no shell)
            ["git", "archive", "--format=tar", REPOMIND_SNAPSHOT_COMMIT],  # noqa: S607
            cwd=root,
            capture_output=True,
            check=True,
            timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            handle = tar.extractfile(member) if member.isfile() else None
            if handle is not None:
                files[member.name] = handle.read()
    return files
