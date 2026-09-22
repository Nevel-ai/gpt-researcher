"""Trusted lifetime for private research budget credentials, never provider headers."""
import base64
from functools import wraps
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
import inspect
import json
import logging
import os
import re
import sys
import time

from .budget import ResearchBudget, ResearchBudgetError, current_research_budget


def verify_run_capability(token, secret, now_ms=None):
    try:
        if not isinstance(token, str) or len(token) > 2048 or not isinstance(secret, str) or len(secret) < 32:
            raise ValueError()
        match = re.fullmatch(r"nbgt2\.([A-Za-z0-9_-]+)\.([A-Za-z0-9_-]{86})", token)
        if not match:
            raise ValueError()
        key = load_pem_public_key(secret.encode())
        signature = base64.urlsafe_b64decode(match[2] + "==")
        if not isinstance(key, Ed25519PublicKey) or base64.urlsafe_b64encode(signature).decode().rstrip("=") != match[2]:
            raise ValueError()
        key.verify(signature, ("nbgt2." + match[1]).encode())
        claims = json.loads(base64.urlsafe_b64decode(match[1] + "=" * (-len(match[1]) % 4)))
        if not isinstance(claims, dict) or set(claims) != {"version", "kind", "tool", "subject", "runId", "issuedAt", "expiresAt", "mode"}:
            raise ValueError()
        if type(claims["version"]) is not int or claims["version"] != 1 or claims["kind"] != "run" or claims["tool"] != "deep_research" or claims["mode"] not in {"shadow", "enforce"}:
            raise ValueError()
        subject = claims["subject"]
        if not isinstance(subject, dict) or set(subject) != {"userId", "workspaceId"} or any(not isinstance(v, str) or len(v) != 16 for v in subject.values()):
            raise ValueError()
        if not isinstance(claims["runId"], str) or not re.fullmatch(r"[A-Za-z0-9_:-]{1,128}", claims["runId"]):
            raise ValueError()
        issued, expires = claims["issuedAt"], claims["expiresAt"]
        now = time.time() * 1000 if now_ms is None else now_ms
        if (type(issued) is not int or type(expires) is not int or issued < 0 or expires > 2**53 - 1
                or not 0 < expires - issued <= 3_600_000 or issued > now + 60_000 or now >= expires):
            raise ValueError()
        return claims
    except Exception:
        raise ResearchBudgetError("budget_invalid_transition") from None


def verify_budget_start(data):
    """Versioned socket never accepts a start lacking an authenticated budget."""
    try:
        if not isinstance(data, str) or not data.startswith("start "):
            raise ValueError()
        request = json.loads(data[6:])
        private = request["headers"]["nevel_budget"]
        if not isinstance(private, dict) or set(private) != {"capability"}:
            raise ValueError()
        return verify_run_capability(private["capability"], os.environ.get("NEVEL_BUDGET_PUBLIC_KEY", ""))
    except Exception:
        raise ResearchBudgetError("budget_invalid_transition") from None


def with_research_budget(function):
    signature = inspect.signature(function)

    @wraps(function)
    async def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        headers = dict(bound.arguments.get("headers") or {})
        private = headers.pop("nevel_budget", None)
        bound.arguments["headers"] = headers
        if private is None:
            return await function(*bound.args, **bound.kwargs)
        if current_research_budget.get() is not None or not isinstance(private, dict) or set(private) != {"capability"}:
            raise ResearchBudgetError("budget_invalid_transition")
        # Verify the signed mode locally before considering shadow fallback.
        # A forged mode or invalid signature must never grant unmetered execution.
        capability = private["capability"]
        claims = verify_run_capability(capability, os.environ.get("NEVEL_BUDGET_PUBLIC_KEY", ""))
        try:
            budget = ResearchBudget(capability, claims["mode"])
        except Exception:
            if claims["mode"] != "shadow":
                raise ResearchBudgetError() from None
            logging.getLogger(__name__).warning("Shadow research budget callback configuration unavailable")
            return await function(*bound.args, **bound.kwargs)
        token = current_research_budget.set(budget)
        try:
            result = await function(*bound.args, **bound.kwargs)
            # Legacy inner catches may turn a denied substep into an empty
            # fallback. Never advertise the overall paid run as successful then.
            budget.raise_if_denied()
            return result
        except Exception:
            # SDKs may replace a transport denial with a connection/status
            # exception. Preserve the run's nonretryable budget classification.
            budget.raise_if_denied()
            raise
        finally:
            primary_error = sys.exc_info()[0]
            try:
                await budget.aclose()
            except Exception:
                if primary_error is None:
                    raise ResearchBudgetError() from None
                logging.getLogger(__name__).error("Research budget client cleanup failed")
            finally:
                current_research_budget.reset(token)
    return wrapped
