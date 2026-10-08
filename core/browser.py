"""Browser control: Playwright with one persistent profile per account.

Why this file is structured the way it is
-----------------------------------------
* **One thread owns Playwright.** The sync API can only be used from the thread
  that created it, and the permission gate runs every tool in a worker thread
  (so a hung tool cannot hang the assistant). So the browser lives behind an
  *actor*: :class:`BrowserManager` owns a single thread, tools post commands to
  it, and results come back through a queue. Nothing else touches Playwright.
* **Sessions persist.** Each profile is a real Chromium user-data directory at
  ``profiles/<label>``. Logins (including 2FA) are done by you, by hand, once;
  cookies survive restarts. GARVIS never reads those files and never types a
  credential.
* **Human handover is a state, not a message.** CAPTCHAs, 2FA prompts, login
  walls and bot walls put the *profile* into ``takeover`` state. Every later
  automated action on that profile is refused - with an explanation - until you
  say you are done. The detection is heuristic (see :class:`BlockDetector`) and
  it is deliberately biased towards handing control back too often rather than
  too rarely.
* **Only allowlisted domains.** :meth:`BrowserManager.check_site` runs on every
  navigation, so a redirect cannot smuggle the browser onto a site the config
  does not permit. The gate checks the same thing: defence in depth.

What this cannot do, stated plainly: log you in, solve a CAPTCHA, complete 2FA,
or read a page that requires a login you have not done. Some sites (banks, most
big retail, anything behind Cloudflare) will never be automatable, and the
detector will hand them back to you rather than grind against them.
"""

from __future__ import annotations

import os
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Block detection (pure logic: no Playwright, fully unit-testable)
# ---------------------------------------------------------------------------
CAPTCHA = "captcha"
TWO_FACTOR = "2fa"
BOT_BLOCK = "bot_block"
LOGIN_REQUIRED = "login_required"
CONSENT = "consent"
STOPPED = "stopped"            # the human pressed STOP EVERYTHING

_CAPTCHA_MARKERS = (
    "recaptcha", "g-recaptcha", "hcaptcha", "cf-turnstile", "turnstile",
    "verify you are human", "verify you are a human", "are you a robot",
    "i'm not a robot", "i am not a robot", "not a robot", "human verification",
    "checking your browser", "press and hold", "select all images",
    "prove you are human", "complete the security check", "captcha",
)
_TWO_FACTOR_MARKERS = (
    "two-factor", "two factor", "2-step", "2 step verification", "two-step",
    "verification code", "security code", "one-time code", "one time code",
    "one-time password", "authenticator app", "authentication code",
    "enter the code we sent", "enter the 6-digit", "6-digit code", "sms code",
    "confirm it's you", "confirm it is you", "we sent a code", "otp",
)
_BOT_BLOCK_MARKERS = (
    "unusual traffic", "automated queries", "we have detected unusual",
    "your request has been blocked", "access denied", "you have been blocked",
    "too many requests", "rate limit exceeded", "request blocked",
    "blocked by", "unusual activity", "suspicious activity",
    "verify your identity to continue", "automated access",
)
_LOGIN_MARKERS = (
    "sign in to continue", "log in to continue", "please log in", "please sign in",
    "you must be logged in", "session expired", "sign in to your account",
    "enter your password",
)
_CONSENT_MARKERS = (
    "accept all cookies", "before you continue", "we value your privacy",
    "manage your cookie", "consent to the use of cookies",
)


@dataclass
class BlockSignal:
    """A reason GARVIS should stop and hand the browser back to the human."""

    kind: str
    evidence: str
    url: str = ""
    title: str = ""

    @property
    def message(self) -> str:
        base = {
            CAPTCHA: "There is a human-verification check on that page.",
            TWO_FACTOR: "That page is asking for a verification code.",
            BOT_BLOCK: "The site has blocked automated access.",
            LOGIN_REQUIRED: "That page needs you to sign in first.",
            CONSENT: "That page is showing a cookie/consent prompt.",
            STOPPED: "You stopped me on this page, so it is yours.",
        }.get(self.kind, "That page needs a human.")
        return f"{base} {self.evidence}".strip()


class BlockDetector:
    """Heuristics that decide when to hand control back to the user.

    Detection runs on the page's visible text, its title, the URL and a slice of
    HTML (for CAPTCHA iframes and widgets that render invisibly). Tuned to
    over-trigger: a false positive costs you one "continue", a false negative
    means GARVIS grinds against a wall - or worse, clicks through something it
    should not have touched.
    """

    def __init__(self, enabled: dict[str, bool] | None = None) -> None:
        enabled = enabled or {}
        self.detect_captcha = enabled.get("captcha", True)
        self.detect_2fa = enabled.get("2fa", True)
        self.detect_botblock = enabled.get("bot_block", True)
        self.detect_login = enabled.get("login", True)
        self.detect_consent = enabled.get("consent", False)

    def analyze(self, url: str = "", title: str = "", text: str = "", html: str = "") -> BlockSignal | None:
        haystack_text = f"{title}\n{text}".lower()
        haystack_all = f"{haystack_text}\n{html[:20000].lower()}"

        if url.startswith("data:") or url.startswith("about:"):
            return BlockSignal(CAPTCHA, "the browser is on an interstitial page", url, title)

        # 2FA first: it is the most specific, and its markers rarely appear by accident.
        if self.detect_2fa:
            hit = self._find(haystack_text, _TWO_FACTOR_MARKERS, html_ok=False)
            if hit:
                return BlockSignal(TWO_FACTOR, f"(matched: {hit!r})", url, title)

        if self.detect_captcha:
            hit = self._find(haystack_all, _CAPTCHA_MARKERS)
            if hit:
                return BlockSignal(CAPTCHA, f"(matched: {hit!r})", url, title)

        if self.detect_botblock:
            hit = self._find(haystack_all, _BOT_BLOCK_MARKERS)
            if hit:
                return BlockSignal(BOT_BLOCK, f"(matched: {hit!r})", url, title)

        if self.detect_login:
            hit = self._find(haystack_text, _LOGIN_MARKERS)
            if hit and self._looks_like_a_login_form(html):
                return BlockSignal(LOGIN_REQUIRED, f"(matched: {hit!r})", url, title)

        if self.detect_consent:
            hit = self._find(haystack_text, _CONSENT_MARKERS)
            if hit:
                return BlockSignal(CONSENT, f"(matched: {hit!r})", url, title)

        if self._looks_like_a_status_wall(title, text):
            return BlockSignal(BOT_BLOCK, "(empty page with an error title)", url, title)
        return None

    @staticmethod
    def _find(haystack: str, needles: tuple[str, ...], html_ok: bool = True) -> str | None:
        for needle in needles:
            if needle in haystack:
                return needle
        return None

    @staticmethod
    def _looks_like_a_login_form(html: str) -> bool:
        lowered = html[:20000].lower()
        return 'type="password"' in lowered or "type='password'" in lowered or 'name="password"' in lowered

    @staticmethod
    def _looks_like_a_status_wall(title: str, text: str) -> bool:
        lowered = f"{title} {text}".strip().lower()
        if not lowered:
            return True  # blank page
        for marker in ("403 forbidden", "404 not found", "429", "503 service", "access denied", "error"):
            if marker in lowered and len(lowered) < 400:
                return True
        return False


# ---------------------------------------------------------------------------
# Credentials and dangerous controls (also pure logic)
# ---------------------------------------------------------------------------
_PASSWORD_HINTS = (
    "password", "passwd", "pwd", "passphrase", "passcode", "pin",
    "otp", "totp", "2fa", "mfa", "verification", "security code", "auth code",
    "one-time", "one_time", "token", "secret", "cvv", "cvc", "card number",
    "cardnumber", "card_number", "cardnumber", "security number", "ssn",
    "social security", "iban", "routing", "two-factor", "two_factor", "2-factor",
    "authenticator", "digit code", "digit-code", "enter the code", "sms code",
    "confirmation code", "card security", "expiry",
)

# A run of 13-19 digits is a card/account number, with or without spaces/dashes.
_CARD_NUMBER_RE = re.compile(r"(?:\d[\s-]?){13,19}")
# "6-digit", "8 digit", "code of 6 digits"...
_DIGIT_CODE_RE = re.compile(r"\b\d{1,2}[\s-]?digit")


_SECRET_TOKENS = frozenset({
    "password", "passwd", "pwd", "passphrase", "passcode", "pin", "otp", "totp",
    "2fa", "mfa", "cvv", "cvc", "ssn", "iban", "secret", "token", "cardnumber",
    "card_number",
})
_SECRET_PAIRS = (("card", "number"), ("credit", "card"), ("one", "time"),
                 ("security", "code"), ("auth", "code"), ("sms", "code"))


def selector_names_a_secret(selector: str) -> bool:
    """True when the selector itself names a credential field.

    Used as the always-available half of the password refusal: it needs no page,
    so it cannot fail open the way a live DOM lookup can. Matching is by whole
    token, so ``#spinner`` and ``#shipping-address`` are *not* treated as PIN
    fields, while ``#card-number`` and ``input[type=password]`` are.
    """
    tokens = [token for token in re.split(r"[^a-z0-9]+", str(selector or "").lower()) if token]
    if any(token in _SECRET_TOKENS for token in tokens):
        return True
    pairs = set(zip(tokens, tokens[1:]))
    return any(pair in pairs for pair in _SECRET_PAIRS)


def looks_like_a_secret_field(hint: str) -> bool:
    """Decide whether a field is for a credential GARVIS must never fill."""
    text = str(hint or "").lower()
    if not text:
        return False
    if re.search(r"\bpassword\b|\bpasscode\b|\bpin\b|cvv|cvc|otp|totp|2fa|mfa", text):
        return True
    if _DIGIT_CODE_RE.search(text):
        return True
    if _CARD_NUMBER_RE.search(text):
        return True  # a long digit run is a number we must never handle
    return any(needle in text for needle in _PASSWORD_HINTS)


MONEY_WORDS = (
    "pay", "buy", "purchase", "checkout", "place order", "confirm order", "subscribe",
    "upgrade", "donate", "tip", "transfer", "wire", "send money", "add to cart",
    "billing", "card details", "payment",
)
DESTRUCTIVE_WORDS = (
    "delete", "remove", "erase", "unsubscribe", "cancel", "close account",
    "deactivate", "revoke", "disable", "unlink", "disconnect", "wipe", "reset",
)
SEND_WORDS = (
    "send", "post", "publish", "submit", "upload", "share", "tweet", "reply",
    "comment", "message", "invite", "follow", "subscribe",
)
VERIFY_WORDS = (
    "verify", "captcha", "i'm not a robot", "i am not a robot", "human",
    "challenge", "security check",
)


def classify_click_target(text: str, selector: str = "") -> tuple[str | None, str]:
    """Classify what a click would do, from the element's text/label.

    Returns ``(tier_or_blocked, reason)``:
      * ``("blocked", ...)`` for human-verification widgets - never automated;
      * ``("red", ...)`` for anything that spends money, deletes, or (when
        ``never_auto_submit`` is on) submits content to a third party;
      * ``(None, "")`` for an ordinary click, which stays YELLOW.
    """
    haystack = f"{text} {selector}".lower()
    if any(word in haystack for word in VERIFY_WORDS):
        return "blocked", (
            "that control is a human-verification step; it is yours to complete, "
            "not mine to click"
        )
    for word in MONEY_WORDS:
        if word in haystack:
            return "red", f"the control looks like it spends money ('{word}')"
    for word in DESTRUCTIVE_WORDS:
        if word in haystack:
            return "red", f"the control looks destructive or irreversible ('{word}')"
    for word in SEND_WORDS:
        if word in haystack:
            return "red", f"the control would send or publish something ('{word}')"
    return None, ""


# ---------------------------------------------------------------------------
# Driver: the Playwright-facing layer
# ---------------------------------------------------------------------------
@dataclass
class PageSnapshot:
    url: str = ""
    title: str = ""
    text: str = ""
    html_excerpt: str = ""
    selector: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url, "title": self.title, "text": self.text,
            "html_excerpt": self.html_excerpt, "selector": self.selector,
        }


@dataclass
class DriverResult:
    ok: bool
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)


class PlaywrightDriver:
    """The real thing: launches Chromium and talks to pages.

    Every method is called from the actor thread only.
    """

    def __init__(self, headless: bool = False, slow_mo_ms: int = 0, log: Any = None) -> None:
        self.headless = bool(headless)
        self.slow_mo_ms = int(slow_mo_ms or 0)
        self.log = log
        self._playwright = None
        self._contexts: dict[str, Any] = {}
        self._pages: dict[str, Any] = {}
        self._page_timeout_ms: int = 15000
        self._nav_timeout_ms: int = 30000
        self.available = False
        self.reason = ""

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> DriverResult:
        try:
            from playwright.sync_api import sync_playwright  # noqa: PLC0415
        except Exception as exc:
            self.available = False
            self.reason = (
                f"Playwright is not installed ({exc}). Install it with "
                f"`pip install playwright` then `playwright install chromium`."
            )
            return DriverResult(False, self.reason)
        try:
            # No browser is launched here on purpose: every profile is its own
            # *persistent* Chromium context (that is what keeps logins), so
            # starting one now would waste a process. This only proves that
            # Playwright works and that Chromium has been downloaded.
            self._playwright = sync_playwright().start()
            executable = self._playwright.chromium.executable_path
            if not executable or not os.path.exists(executable):
                raise RuntimeError(f"the Chromium binary is missing (looked in {executable})")
            self.available = True
            self.reason = ""
            return DriverResult(True, "playwright ready")
        except Exception as exc:
            self.available = False
            self.reason = (
                f"Could not start Chromium ({exc}). If this is the first run, "
                f"execute: playwright install chromium"
            )
            return DriverResult(False, self.reason)

    def stop(self) -> None:
        for label, context in list(self._contexts.items()):
            try:
                context.close()
            except Exception:
                pass
            self._contexts.pop(label, None)
            self._pages.pop(label, None)
        try:
            if self._playwright is not None:
                self._playwright.stop()
        except Exception:
            pass
        self._playwright = None

    # -- profiles ----------------------------------------------------------
    def open_profile(self, label: str, profile_dir: Path, page_timeout_ms: int, nav_timeout_ms: int) -> DriverResult:
        self._page_timeout_ms = page_timeout_ms
        self._nav_timeout_ms = nav_timeout_ms
        if label in self._pages:
            return DriverResult(True, f"profile '{label}' is already open", {"open": True})
        profile_dir.mkdir(parents=True, exist_ok=True)
        try:
            # A *persistent* context is what keeps cookies, localStorage and
            # logins across restarts. It cannot be created from the browser
            # object (that would be incognito-like), so it is launched here
            # directly from playwright.chromium.
            context = self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(profile_dir),
                headless=self.headless,
                viewport={"width": 1440, "height": 900},
                locale="en-US",
                slow_mo=self.slow_mo_ms or None,
                args=["--disable-blink-features=AutomationControlled"],
                accept_downloads=False,
            )
            page = context.pages[0] if context.pages else context.new_page()
            page.set_default_timeout(self._page_timeout_ms)
            page.set_default_navigation_timeout(self._nav_timeout_ms)
            self._contexts[label] = context
            self._pages[label] = page
            return DriverResult(True, f"opened browser profile '{label}'", {"user_data_dir": str(profile_dir)})
        except Exception as exc:
            return DriverResult(False, f"could not open browser profile '{label}': {exc}")

    def close_profile(self, label: str) -> DriverResult:
        context = self._contexts.pop(label, None)
        self._pages.pop(label, None)
        if context is None:
            return DriverResult(True, f"profile '{label}' was not open")
        try:
            context.close()
        except Exception as exc:
            return DriverResult(False, f"could not close profile '{label}': {exc}")
        return DriverResult(True, f"closed browser profile '{label}'")

    def open_profiles(self) -> list[str]:
        return sorted(self._pages)

    def page(self, label: str) -> Any:
        return self._pages.get(label)

    # -- navigation and reading -------------------------------------------
    def goto(self, label: str, url: str) -> DriverResult:
        page = self._pages.get(label)
        if page is None:
            return DriverResult(False, f"profile '{label}' is not open")
        try:
            response = page.goto(url, wait_until="domcontentloaded", timeout=self._nav_timeout_ms)
            status = getattr(response, "status", None)
            return DriverResult(
                True,
                f"navigated to {page.url}" + (f" (HTTP {status})" if status else ""),
                {"url": page.url, "status": status},
            )
        except Exception as exc:
            return DriverResult(False, f"navigation to {url} failed: {exc}")

    def snapshot(self, label: str, selector: str = "", max_chars: int = 8000) -> DriverResult:
        page = self._pages.get(label)
        if page is None:
            return DriverResult(False, f"profile '{label}' is not open")
        try:
            url = page.url
            title = page.title()
            if selector:
                element = page.query_selector(selector)
                if element is None:
                    return DriverResult(False, f"no element matched '{selector}'")
                text = element.inner_text() or ""
                html = element.inner_html() or ""
            else:
                text = page.inner_text("body") or ""
                html = page.content() or ""
            return DriverResult(
                True,
                "",
                PageSnapshot(
                    url=url, title=title, text=text[:max_chars],
                    html_excerpt=html[:20000], selector=selector,
                ).to_dict(),
            )
        except Exception as exc:
            return DriverResult(False, f"could not read the page: {exc}")

    def element_hints(self, label: str, selector: str) -> dict[str, str]:
        """Attributes of an element, used to spot credential fields."""
        page = self._pages.get(label)
        if page is None:
            return {}
        try:
            element = page.query_selector(selector)
            if element is None:
                return {}
            hints: dict[str, str] = {}
            for attribute in ("type", "name", "id", "placeholder", "autocomplete",
                              "aria-label", "title", "data-testid"):
                value = element.get_attribute(attribute)
                if value:
                    hints[attribute] = value
            return hints
        except Exception:
            return {}

    # -- interaction -------------------------------------------------------
    def click(self, label: str, selector: str = "", text: str = "") -> DriverResult:
        page = self._pages.get(label)
        if page is None:
            return DriverResult(False, f"profile '{label}' is not open")
        try:
            if selector:
                page.click(selector, timeout=self._page_timeout_ms)
                return DriverResult(True, f"clicked '{selector}'")
            if text:
                locator = page.get_by_text(text, exact=False).first
                locator.click(timeout=self._page_timeout_ms)
                return DriverResult(True, f"clicked the element containing {text!r}")
            return DriverResult(False, "no selector or text given")
        except Exception as exc:
            return DriverResult(False, f"click failed: {exc}")

    def type_text(self, label: str, selector: str, text: str, submit: bool = False) -> DriverResult:
        page = self._pages.get(label)
        if page is None:
            return DriverResult(False, f"profile '{label}' is not open")
        try:
            page.fill(selector, text, timeout=self._page_timeout_ms)
            if submit:
                page.press(selector, "Enter")
            return DriverResult(True, f"typed {len(text)} characters into '{selector}'"
                                      + (" and pressed Enter" if submit else ""))
        except Exception as exc:
            return DriverResult(False, f"typing failed: {exc}")

    def press(self, label: str, key: str) -> DriverResult:
        page = self._pages.get(label)
        if page is None:
            return DriverResult(False, f"profile '{label}' is not open")
        try:
            page.keyboard.press(key)
            return DriverResult(True, f"pressed {key}")
        except Exception as exc:
            return DriverResult(False, f"key press failed: {exc}")

    def screenshot(self, label: str, path: Path, full_page: bool = True) -> DriverResult:
        page = self._pages.get(label)
        if page is None:
            return DriverResult(False, f"profile '{label}' is not open")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(path), full_page=full_page)
            return DriverResult(True, f"screenshot saved to {path}", {"path": str(path)})
        except Exception as exc:
            return DriverResult(False, f"screenshot failed: {exc}")

    def wait(self, seconds: float) -> None:
        time.sleep(max(0.0, float(seconds)))



# ---------------------------------------------------------------------------
# The fake driver: used by the tests and by --browser-check on machines
# without Chromium. It mirrors PlaywrightDriver's method surface exactly.
# ---------------------------------------------------------------------------

class FakePage:
    """One in-memory page. Real browsers keep one per profile; so does the fake."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.url = "about:blank"
        self.title = ""
        self.page_text = ""
        self.page_html = ""
        self.element_attributes: dict[str, dict[str, str]] = {}
        self.element_texts: dict[str, str] = {}


class FakeDriver:
    """In-memory driver for tests and for `--check` on machines without Chromium.

    It implements the same surface as :class:`PlaywrightDriver`, so the whole
    tool layer (tiers, guards, block detection, framing) can be tested without
    a browser binary. Each profile has its own page, so ``page_text`` and
    friends always describe the profile that was used last::

        driver.page_text = "Verify you are human"     # the current profile's page
        driver.page_for("work").page_text = "..."     # a named profile's page
    """

    def __init__(self, **_: Any) -> None:
        self.available = True
        self.reason = ""
        self._open: dict[str, Any] = {}
        self._pages: dict[str, FakePage] = {"": FakePage("")}
        self._current = ""
        self.clicked: list[str] = []
        self.typed: list[tuple[str, str, bool]] = []
        self.pressed: list[str] = []
        self.visited: list[str] = []
        self.screenshots: list[str] = []

    # -- pages: page_text/url/title/... describe the profile used last --------
    def page_for(self, label: str) -> FakePage:
        page = self._pages.get(label)
        if page is None:
            page = self._pages[label] = FakePage(label)
        self._current = label
        return page

    @property
    def _page(self) -> FakePage:
        return self._pages[self._current]

    def __getattr__(self, name: str) -> Any:
        # Any unknown attribute that a FakePage has (url, title, page_text,
        # page_html, element_attributes, element_texts) reads from the current page.
        if name.startswith("_"):
            raise AttributeError(name)
        page = self.__dict__.get("_pages", {}).get(self.__dict__.get("_current", ""))
        if page is not None and hasattr(page, name):
            return getattr(page, name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        pages = self.__dict__.get("_pages")
        if pages is not None and name not in ("_pages", "_current", "_open"):
            page = pages.get(self.__dict__.get("_current", ""))
            if page is not None and hasattr(page, name):
                setattr(page, name, value)
                return
        object.__setattr__(self, name, value)

    # -- lifecycle
    def start(self) -> DriverResult:
        return DriverResult(True, "fake browser ready")

    def stop(self) -> None:
        self._open.clear()

    def open_profile(self, label: str, profile_dir: Path, page_timeout_ms: int, nav_timeout_ms: int) -> DriverResult:
        profile_dir.mkdir(parents=True, exist_ok=True)
        self._open[label] = {"profile_dir": str(profile_dir)}
        return DriverResult(True, f"opened browser profile '{label}'", {"user_data_dir": str(profile_dir)})

    def close_profile(self, label: str) -> DriverResult:
        removed = self._open.pop(label, None)
        return DriverResult(True, f"closed browser profile '{label}'" if removed else f"profile '{label}' was not open")

    def open_profiles(self) -> list[str]:
        return sorted(self._open)

    # -- navigation and reading
    def goto(self, label: str, url: str) -> DriverResult:
        if label not in self._open:
            return DriverResult(False, f"profile '{label}' is not open")
        page = self.page_for(label)
        page.url = url
        self.visited.append(url)
        if not page.title:
            page.title = url.split("//")[-1].split("/")[0]
        return DriverResult(True, f"navigated to {url}", {"url": url, "status": 200})

    def snapshot(self, label: str, selector: str = "", max_chars: int = 8000) -> DriverResult:
        if label not in self._open:
            return DriverResult(False, f"profile '{label}' is not open")
        page = self.page_for(label)
        if selector and not self._selector_matches(page, selector):
            return DriverResult(False, f"no element matched '{selector}'")
        text = page.page_text if not selector else page.element_texts.get(selector, "")
        return DriverResult(
            True, "",
            PageSnapshot(url=page.url, title=page.title, text=text[:max_chars],
                         html_excerpt=page.page_html[:20000], selector=selector).to_dict(),
        )

    def _selector_matches(self, page: FakePage, selector: str) -> bool:
        """Best-effort matching: literal text, or an id/class without its prefix."""
        if selector in page.page_html or selector in page.page_text:
            return True
        token = selector.lstrip("#.[]").split(" ")[0].strip()
        if not token:
            return False
        return token in page.page_html or token in page.page_text

    def element_hints(self, label: str, selector: str) -> dict[str, str]:
        return dict(self.page_for(label).element_attributes.get(selector, {}))

    # -- interaction
    def click(self, label: str, selector: str = "", text: str = "") -> DriverResult:
        if label not in self._open:
            return DriverResult(False, f"profile '{label}' is not open")
        self.page_for(label)
        self.clicked.append(selector or text)
        return DriverResult(True, f"clicked {selector or text!r}")

    def type_text(self, label: str, selector: str, text: str, submit: bool = False) -> DriverResult:
        if label not in self._open:
            return DriverResult(False, f"profile '{label}' is not open")
        self.page_for(label)
        self.typed.append((selector, text, submit))
        return DriverResult(True, f"typed into '{selector}'")

    def press(self, label: str, key: str) -> DriverResult:
        if label not in self._open:
            return DriverResult(False, f"profile '{label}' is not open")
        self.pressed.append(key)
        if key.lower() == "enter":
            self.page_for(label).page_text += "\n[submitted]"
        return DriverResult(True, f"pressed {key}")

    def screenshot(self, label: str, path: Path, full_page: bool = True) -> DriverResult:
        if label not in self._open:
            return DriverResult(False, f"profile '{label}' is not open")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x89PNG\r\n\x1a\n fake screenshot")
        self.screenshots.append(str(path))
        return DriverResult(True, f"screenshot saved to {path}", {"path": str(path)})

    def wait(self, seconds: float) -> None:
        return


# ---------------------------------------------------------------------------
# The manager: actor thread + profiles + takeover state
# ---------------------------------------------------------------------------
@dataclass
class ProfileState:
    label: str
    path: Path
    open: bool = False
    takeover: BlockSignal | None = None
    last_url: str = ""
    last_action_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "path": str(self.path),
            "open": self.open,
            "last_url": self.last_url,
            "takeover": self.takeover.kind if self.takeover else None,
            "takeover_detail": self.takeover.evidence if self.takeover else "",
        }


_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,39}$")


def validate_label(label: str) -> str:
    """Profile labels become directory names: validate hard, refuse traversal."""
    text = str(label or "").strip()
    if not _LABEL_RE.match(text):
        raise ValueError(
            f"'{label}' is not a usable profile name. Use letters, digits, spaces, '-' or '_' "
            f"(max 40 chars), e.g. 'work' or 'shop-account'."
        )
    if text in (".", "..") or "/" in text or "\\" in text:
        raise ValueError(f"'{label}' is not a valid profile name.")
    return text


class BrowserError(RuntimeError):
    """Raised for browser problems the model should hear about verbatim."""


class BrowserManager:
    """Owns the browser actor, the profiles and the human-takeover state."""

    def __init__(
        self,
        cfg: Any,
        activity: Any = None,
        log: Any = None,
        driver: Any | None = None,
        notify: Callable[[str], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.activity = activity
        self.log = log
        self.notify = notify
        self.enabled = bool(cfg.get("browser.enabled", True))
        self.profiles_dir: Path = cfg.resolve_path(cfg.get("browser.profiles_dir", "profiles"))
        self.default_profile = str(cfg.get("browser.default_profile", "default"))
        self.headless = bool(cfg.get("browser.headless", False))
        self.slow_mo_ms = int(cfg.get("browser.slow_mo_ms", 0))
        self.action_timeout_s = float(cfg.get("browser.action_timeout_s", 15))
        self.navigation_timeout_s = float(cfg.get("browser.navigation_timeout_s", 30))
        self.max_open_profiles = int(cfg.get("browser.max_open_profiles", 3))
        self.shots_dir: Path = cfg.resolve_path(cfg.get("browser.screenshots_dir", "logs/browser_shots"))
        self.never_auto_submit = bool(cfg.get("browser.never_auto_submit", True))
        self.human = cfg.section("browser").get("human_takeover", {}) or {}
        self.wait_for_human_s = float(self.human.get("wait_for_human_s", 300))
        self.detector = BlockDetector(
            {
                "captcha": bool(self.human.get("detect_captcha", True)),
                "2fa": bool(self.human.get("detect_2fa", True)),
                "bot_block": bool(self.human.get("detect_botblock", True)),
                "login": bool(self.human.get("detect_login", True)),
                "consent": bool(self.human.get("detect_consent", False)),
            }
        )

        self.allowed_sites = cfg.allowed_sites()
        self.blocked_sites = cfg.blocked_sites()
        self._driver_factory: Callable[[], Any] = driver if driver is not None else (
            lambda: PlaywrightDriver(headless=self.headless, slow_mo_ms=self.slow_mo_ms, log=self.log)
        )
        self._driver: Any | None = driver  # injected drivers are used directly
        self._injected_driver = driver is not None
        self._commands: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._state: dict[str, ProfileState] = {}
        self._lock = threading.RLock()
        self._startup = DriverResult(True, "not started")
        self._start_attempted = False
        self._last_error = ""

    # -- site policy -------------------------------------------------------
    def check_site(self, url: str) -> tuple[bool, str]:
        """Allowlist check, repeated here so a redirect cannot bypass the gate."""
        from core.permissions import domain_of  # local import: avoids a cycle

        domain = domain_of(url)
        if not domain:
            return False, f"I could not read a host name out of '{url}'."
        for blocked in self.blocked_sites:
            if domain == blocked or domain.endswith("." + blocked):
                return False, (
                    f"{domain} is on the never-automate list (banking/payments). "
                    f"Open it yourself."
                )
        if not self.allowed_sites:
            return False, "browser.allowed_sites is empty, so I am not navigating anywhere."
        for allowed in self.allowed_sites:
            if domain == allowed or domain.endswith("." + allowed):
                return True, ""
        return False, (
            f"{domain} is not in browser.allowed_sites. Add it to config.yaml if you want me "
            f"to go there."
        )

    # -- profiles ----------------------------------------------------------
    def profile_path(self, label: str) -> Path:
        return self.profiles_dir / validate_label(label)

    def list_profiles(self) -> list[dict[str, Any]]:
        """Profiles on disk, plus which are open and whether one is waiting for you."""
        found: list[dict[str, Any]] = []
        if not self.profiles_dir.exists():
            self.profiles_dir.mkdir(parents=True, exist_ok=True)
        for entry in sorted(self.profiles_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            state = self._state.get(entry.name)
            found.append(
                ProfileState(label=entry.name, path=entry).to_dict()
                if state is None
                else {**state.to_dict(), "open": self.is_open(entry.name)}
            )
        return found

    def is_open(self, label: str) -> bool:
        with self._lock:
            return label in self._state and self._state[label].open

    def takeover_state(self, label: str) -> BlockSignal | None:
        with self._lock:
            state = self._state.get(label)
            return state.takeover if state else None

    # -- actor plumbing ----------------------------------------------------
    def start(self) -> DriverResult:
        """Start the actor and the browser. Idempotent; safe to call at startup."""
        self._start_attempted = True
        if not self.enabled:
            return DriverResult(False, "browser is disabled in config.yaml")
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run_actor, name="garvis-browser", daemon=True)
            self._thread.start()
        result = self._call("start", timeout=max(30.0, self.navigation_timeout_s + 10))
        return result

    def _run_actor(self) -> None:
        """The one thread allowed to touch Playwright."""
        driver = self._driver
        try:
            if driver is None:
                driver = self._driver_factory()
                self._driver = driver
            self._startup = driver.start()
            if self.log:
                (self.log.info if self._startup.ok else self.log.warning)(
                    "browser: %s", self._startup.message
                )
            if self.activity:
                self.activity.event("system", f"browser actor started: {self._startup.message}",
                                    ok=self._startup.ok)
        except Exception as exc:
            self._startup = DriverResult(False, f"browser driver failed to start: {exc}")
            if self.log:
                self.log.error("browser driver failed: %s", exc)

        while True:
            item = self._commands.get()
            if item is None:
                break
            op, args, kwargs, reply = item
            try:
                handler = getattr(self, f"_op_{op}", None)
                if handler is None:
                    reply.put(DriverResult(False, f"unknown browser operation '{op}'"))
                    continue
                reply.put(handler(driver, *args, **kwargs))
            except Exception as exc:
                if self.log:
                    self.log.warning("browser op %s failed: %s", op, exc)
                reply.put(DriverResult(False, f"browser operation '{op}' failed: {exc}"))
        try:
            driver.stop()
        except Exception:
            pass

    def ensure_started(self) -> DriverResult:
        """Start the browser on first use, so the model does not have to know."""
        if not self.enabled:
            return DriverResult(False, "browser is disabled in config.yaml")
        with self._lock:
            running = self._thread is not None and self._thread.is_alive()
        if running:
            return self._startup
        if not bool(self.cfg.get("browser.auto_start", True)):
            return DriverResult(False, "the browser is not running (browser.auto_start is false)")
        return self.start()

    def _call(self, op: str, *args: Any, timeout: float | None = None, **kwargs: Any) -> DriverResult:
        """Post a command to the actor thread and wait for the result."""
        if not self.enabled:
            return DriverResult(False, "browser is disabled in config.yaml")
        if self._thread is None or not self._thread.is_alive():
            started = self.ensure_started()
            if not started.ok:
                return started
        limit = timeout if timeout is not None else (self.action_timeout_s + 5)
        reply: queue.Queue = queue.Queue(maxsize=1)
        self._commands.put((op, args, kwargs, reply))
        try:
            return reply.get(timeout=limit)
        except queue.Empty:
            return DriverResult(
                False,
                f"the browser did not answer within {limit:.0f}s (it may be stuck on a slow page). "
                f"Nothing was changed by me since then.",
            )

    def shutdown(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._commands.put(None)
        thread.join(timeout=10)
        self._thread = None
        self._driver = None if not self._injected_driver else self._driver

    close = shutdown

    # -- operations (actor thread) ----------------------------------------
    def _op_start(self, driver: Any) -> DriverResult:
        if getattr(driver, "available", False):
            return DriverResult(True, f"browser ready ({driver.__class__.__name__})")
        return DriverResult(False, getattr(driver, "reason", "browser unavailable"))

    def _op_open(self, driver: Any, label: str, url: str = "") -> DriverResult:
        if not getattr(driver, "available", False):
            return DriverResult(False, getattr(driver, "reason", "browser unavailable"))
        state = self._state_for(label)
        if not state.open:
            open_count = len(driver.open_profiles())
            if open_count >= self.max_open_profiles and label not in driver.open_profiles():
                oldest = driver.open_profiles()[0]
                driver.close_profile(oldest)
                if oldest in self._state:
                    self._state[oldest].open = False
                if self.log:
                    self.log.info("closed busy profile '%s' to stay under the profile cap", oldest)
            result = driver.open_profile(
                label, state.path,
                int(self.action_timeout_s * 1000), int(self.navigation_timeout_s * 1000),
            )
            if not result.ok:
                return result
            state.open = True
        if url:
            nav = driver.goto(label, url)
            if not nav.ok:
                return nav
            state.last_url = str(nav.data.get("url", url))
        state.last_action_at = time.time()
        return DriverResult(True, f"profile '{label}' ready" + (f" at {state.last_url}" if state.last_url else ""),
                            {"profile": label, "url": state.last_url})

    def _op_goto(self, driver: Any, label: str, url: str) -> DriverResult:
        if not self.is_open(label):
            return DriverResult(False, f"profile '{label}' is not open. Use browser.open first.")
        result = driver.goto(label, url)
        if result.ok:
            self._state[label].last_url = str(result.data.get("url", url))
            self._state[label].last_action_at = time.time()
        return result

    def _op_snapshot(self, driver: Any, label: str, selector: str = "", max_chars: int = 8000) -> DriverResult:
        if not self.is_open(label):
            return DriverResult(False, f"profile '{label}' is not open. Use browser.open first.")
        return driver.snapshot(label, selector=selector, max_chars=max_chars)

    def _op_click(self, driver: Any, label: str, selector: str = "", text: str = "", submit: bool = False) -> DriverResult:
        if not self.is_open(label):
            return DriverResult(False, f"profile '{label}' is not open. Use browser.open first.")
        result = driver.click(label, selector=selector, text=text)
        if result.ok and submit:
            driver.press(label, "Enter")
        if result.ok:
            self._state[label].last_action_at = time.time()
        return result

    def _op_type(self, driver: Any, label: str, selector: str, text: str, submit: bool = False) -> DriverResult:
        if not self.is_open(label):
            return DriverResult(False, f"profile '{label}' is not open. Use browser.open first.")
        result = driver.type_text(label, selector, text, submit=submit)
        if result.ok:
            self._state[label].last_action_at = time.time()
        return result

    def _op_press(self, driver: Any, label: str, key: str) -> DriverResult:
        if not self.is_open(label):
            return DriverResult(False, f"profile '{label}' is not open. Use browser.open first.")
        result = driver.press(label, key)
        if result.ok:
            self._state[label].last_action_at = time.time()
        return result

    def _op_screenshot(self, driver: Any, label: str, name: str = "") -> DriverResult:
        if not self.is_open(label):
            return DriverResult(False, f"profile '{label}' is not open. Use browser.open first.")
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", name or f"{label}-{int(time.time())}")
        path = self.shots_dir / (safe if safe.endswith(".png") else f"{safe}.png")
        return driver.screenshot(label, path, full_page=bool(self.cfg.get("browser.screenshot_full_page", True)))

    def _op_hints(self, driver: Any, label: str, selector: str) -> DriverResult:
        if not self.is_open(label):
            return DriverResult(False, f"profile '{label}' is not open.")
        return DriverResult(True, "", {"hints": driver.element_hints(label, selector)})

    def _op_close(self, driver: Any, label: str) -> DriverResult:
        result = driver.close_profile(label)
        state = self._state.get(label)
        if state:
            state.open = False
            state.takeover = None
        return result

    def _op_resume(self, driver: Any, label: str, wait_s: float = 0.0) -> DriverResult:
        """Clear a takeover, optionally waiting for the block to disappear."""
        state = self._state.get(label)
        if state is None or state.takeover is None:
            return DriverResult(True, f"nothing was waiting on '{label}'")
        deadline = time.time() + max(0.0, float(wait_s))
        while True:
            snapshot = driver.snapshot(label, max_chars=4000)
            signal = None
            if snapshot.ok:
                data = snapshot.data
                signal = self.detector.analyze(
                    url=str(data.get("url", "")), title=str(data.get("title", "")),
                    text=str(data.get("text", "")), html=str(data.get("html_excerpt", "")),
                )
            if signal is None:
                was = state.takeover
                state.takeover = None
                if self.activity:
                    self.activity.event("system", f"browser takeover cleared for '{label}' ({was.kind})")
                return DriverResult(True, f"cleared: '{label}' is mine to use again")
            if time.time() >= deadline:
                return DriverResult(
                    False,
                    f"still blocked on '{label}': {signal.message} "
                    f"(I waited {wait_s:.0f}s). Finish it, then say 'continue'.",
                )
            time.sleep(2.0)

    def _op_status(self, driver: Any) -> DriverResult:
        return DriverResult(True, "", {
            "driver": driver.__class__.__name__,
            "available": bool(getattr(driver, "available", False)),
            "reason": getattr(driver, "reason", ""),
            "open_profiles": driver.open_profiles(),
            "default_profile": self.default_profile,
            "allowed_sites": self.allowed_sites,
            "blocked_sites": self.blocked_sites,
        })

    # -- state helpers -----------------------------------------------------
    def _state_for(self, label: str) -> ProfileState:
        with self._lock:
            if label not in self._state:
                self._state[label] = ProfileState(label=label, path=self.profile_path(label))
            return self._state[label]

    # -- public operations (called from tools) -----------------------------
    def open(self, url: str, profile: str = "") -> DriverResult:
        """Open a profile (if needed) and navigate to ``url``."""
        label = profile or self.default_profile
        try:
            validate_label(label)
        except ValueError as exc:
            return DriverResult(False, str(exc))
        allowed, why = self.check_site(url)
        if not allowed:
            return DriverResult(False, why)
        started = self.ensure_started()
        if not started.ok:
            return started
        result = self._call("open", label, url, timeout=self.navigation_timeout_s + 5)
        if result.ok:
            self._after_action(label, url)
        return result

    def goto(self, url: str, profile: str = "") -> DriverResult:
        label = profile or self.default_profile
        allowed, why = self.check_site(url)
        if not allowed:
            return DriverResult(False, why)
        started = self.ensure_started()
        if not started.ok:
            return started
        result = self._call("goto", label, url, timeout=self.navigation_timeout_s + 5)
        if result.ok:
            self._after_action(label, url)
        return result

    def _open_hint(self, label: str) -> DriverResult | None:
        """A failure the model can act on when it forgot to open the profile.

        Naming the profiles that *are* open saves a round trip (and the model
        cannot guess labels, so it must be told).
        """
        with self._lock:
            open_labels = sorted(name for name, state in self._state.items() if state.open)
        if label in open_labels:
            return None
        if open_labels:
            return DriverResult(
                False,
                f"profile '{label}' is not open. Open profiles: {', '.join(open_labels)}. "
                f"Pass profile='{open_labels[0]}' (or open '{label}' first).",
            )
        return DriverResult(False, f"profile '{label}' is not open. Use browser.open first.")

    def read(self, profile: str = "", selector: str = "", max_chars: int = 8000) -> DriverResult:
        label = profile or self.default_profile
        hint = self._open_hint(label)
        if hint:
            return hint
        return self._call("snapshot", label, selector=selector,
                          max_chars=max_chars, timeout=self.action_timeout_s)

    def screenshot(self, profile: str = "", name: str = "") -> DriverResult:
        label = profile or self.default_profile
        return self._call("screenshot", label, name=name,
                          timeout=self.action_timeout_s)

    def element_hints(self, profile: str, selector: str) -> dict[str, str]:
        """Field attributes (type/name/placeholder...) used to spot credential fields."""
        result = self._call("hints", profile, selector, timeout=self.action_timeout_s)
        return dict(result.data.get("hints", {})) if result.ok else {}

    def click(self, profile: str = "", selector: str = "", text: str = "", submit: bool = False) -> DriverResult:
        label = profile or self.default_profile
        blocked = self.takeover_state(label)
        if blocked is not None:
            return DriverResult(False, self.takeover_refusal(label, blocked))
        result = self._call("click", label, selector=selector, text=text, submit=submit)
        if result.ok:
            self._after_action(label, "")
        return result

    def type_text(self, profile: str = "", selector: str = "", text: str = "", submit: bool = False) -> DriverResult:
        label = profile or self.default_profile
        blocked = self.takeover_state(label)
        if blocked is not None:
            return DriverResult(False, self.takeover_refusal(label, blocked))
        result = self._call("type", label, selector, text, submit=submit)
        if result.ok:
            self._after_action(label, "")
        return result

    def press(self, profile: str = "", key: str = "Enter") -> DriverResult:
        label = profile or self.default_profile
        blocked = self.takeover_state(label)
        if blocked is not None:
            return DriverResult(False, self.takeover_refusal(label, blocked))
        result = self._call("press", label, key)
        if result.ok:
            self._after_action(label, "")
        return result

    def close_profile(self, profile: str = "") -> DriverResult:
        return self._call("close", profile or self.default_profile)

    def resume(self, profile: str = "", wait_s: float = 0.0) -> DriverResult:
        """The user says "continue": clear the takeover for a profile."""
        label = profile or self.default_profile
        return self._call("resume", label, wait_s=wait_s, timeout=max(self.action_timeout_s, wait_s + 5))

    def status(self) -> DriverResult:
        return self._call("status")

    def halt_all(self, reason: str = "STOP EVERYTHING was triggered") -> list[str]:
        """Emergency stop: every open profile is handed back to the human.

        Called by the kill switch. GARVIS will not touch a page again until the
        user says "continue", so a stop cannot be followed by a stray click.
        """
        halted: list[str] = []
        with self._lock:
            for label, state in self._state.items():
                if state.open:
                    state.takeover = BlockSignal(STOPPED, reason)
                    halted.append(label)
        if halted:
            if self.notify:
                try:
                    self.notify(
                        f"Stopped the browser on {', '.join(halted)}. Say 'continue' when you "
                        f"want me to use that profile again."
                    )
                except Exception:
                    pass
            if self.activity:
                self.activity.event(
                    "system", f"browser halted on {', '.join(halted)}: {reason}", ok=True
                )
            if self.log:
                self.log.warning("browser halted on %s: %s", ", ".join(halted), reason)
        return halted

    # -- takeover handling -------------------------------------------------
    def takeover_refusal(self, label: str, signal: BlockSignal) -> str:
        return (
            f"STOPPED: profile '{label}' is waiting for you. {signal.message} "
            f"I will not touch that page until you tell me to continue."
        )

    def _after_action(self, label: str, url_hint: str) -> None:
        """Look at the page after every action; hand over if a human is needed."""
        snapshot = self._call("snapshot", label, selector="", max_chars=4000)
        signal: BlockSignal | None = None
        if snapshot.ok:
            data = snapshot.data
            url = str(data.get("url", "") or url_hint)
            if url and not url.startswith(("about:", "data:")):
                allowed, why = self.check_site(url)
                if not allowed:
                    signal = BlockSignal(BOT_BLOCK, f"it redirected somewhere I am not allowed: {why}", url)
            if signal is None:
                signal = self.detector.analyze(
                    url=str(data.get("url", "") or url_hint),
                    title=str(data.get("title", "")),
                    text=str(data.get("text", "")),
                    html=str(data.get("html_excerpt", "")),
                )
                if signal and signal.kind == LOGIN_REQUIRED and not self.human.get("detect_login", True):
                    signal = None
        else:
            message = snapshot.message.lower()
            if "timeout" in message or "not open" in message:
                signal = BlockSignal(BOT_BLOCK, f"the page did not respond ({snapshot.message})", url_hint)

        if signal is None:
            return
        state = self._state_for(label)
        if state.takeover is None:
            state.takeover = signal
            if self.activity:
                self.activity.event(
                    "system",
                    f"browser handover ({signal.kind}) on profile '{label}': {signal.evidence}",
                    ok=False,
                    extra={"url": signal.url},
                )
            if self.log:
                self.log.warning(
                    "handing the browser back to the user: %s (%s)", signal.kind, signal.evidence
                )
            if self.notify:
                try:
                    self.notify(
                        f"I need you to take over: {signal.message} "
                        f"Finish it in the browser window, then tell me to continue."
                    )
                except Exception:
                    pass

    def describe(self) -> str:
        """One honest line about the browser, for --check and the startup log."""
        bits = [f"enabled={self.enabled}", f"profiles_dir={self.profiles_dir}"]
        if not self.enabled:
            bits.append("off in config.yaml")
        elif not self._start_attempted:
            auto = bool(self.cfg.get("browser.auto_start", True))
            bits.append("starts on the first browser call" if auto else "not started (auto_start is false)")
        elif self._startup.ok:
            bits.append("driver=ready")
        else:
            bits.append(f"UNAVAILABLE: {self._startup.message}")
        if self._state:
            open_labels = ", ".join(sorted(self._state)) or "none"
            bits.append(f"known profiles: {open_labels}")
        return ", ".join(bits)


# ---------------------------------------------------------------------------
# Verification support (requirement 6)
# ---------------------------------------------------------------------------
def after_action_note(services: dict[str, Any], tool_name: str, args: dict[str, Any]) -> str | None:
    """Evidence that a browser action landed: the page afterwards.

    The permission gate calls this after a state-changing browser tool, because
    a click cannot be re-read from disk the way a written file can. Returning
    None is a real answer ("nothing to add"): the gate then tells the model the
    action is unchecked instead of implying it was verified.
    """
    if not str(tool_name).startswith("browser."):
        return None
    manager = (services or {}).get("browser")
    if manager is None:
        return None
    try:
        result = manager.screenshot(
            str(args.get("profile", "") or ""),
            name=f"after-{str(tool_name).split('.')[-1]}",
        )
    except Exception:  # a verification must never break the action it checks
        return None
    if not getattr(result, "ok", False):
        return None
    path = (getattr(result, "data", None) or {}).get("path")
    if not path:
        return None
    return f"the page was captured afterwards ({path}); compare it with what was expected"
