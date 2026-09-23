"""
src/ai_explanation.py
=====================
Generates a plain-language explanation of a delivery-time prediction using
an LLM.  The ML model (Random Forest) remains solely responsible for the
numeric prediction; the LLM only explains what factors the model weighted
and why the number makes sense in context.

Supported backends (tried in order of config):
  1. IBM watsonx.ai  — REST API  /ml/v1/text/generation
  2. OpenAI          — openai Python SDK (also supports compatible endpoints)

Credentials are loaded exclusively from:
  - Streamlit secrets  (.streamlit/secrets.toml)  — preferred in Streamlit Cloud
  - Environment variables                           — preferred for local / CI use

NEVER hard-code API keys in this file or anywhere else in the project.

Return contract
---------------
get_ai_explanation() always returns an AIResult named-tuple:
  .text   — the generated explanation string, or None
  .error  — one of the AIError enum values (OK when text is set)
  .detail — a safe, user-displayable description of any error (no keys/tokens)

The caller checks .error to decide what message to show. It never needs to
parse exception messages or inspect raw HTTP responses.
"""

from __future__ import annotations

import enum
import logging
import os
from typing import NamedTuple

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Watsonx.ai constants
# ---------------------------------------------------------------------------
WX_IAM_URL      = "https://iam.cloud.ibm.com/identity/token"
WX_API_VERSION  = "2024-05-31"
WX_DEFAULT_MODEL = "ibm/granite-3-3-8b-instruct"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class AIError(enum.Enum):
    OK                   = "ok"
    NOT_CONFIGURED       = "not_configured"       # no credentials found at all
    MISSING_FIELD        = "missing_field"         # e.g. project_id blank for watsonx
    AUTH_FAILED          = "auth_failed"           # 401 / IAM token exchange failure
    RATE_LIMITED         = "rate_limited"          # 429
    MODEL_NOT_FOUND      = "model_not_found"       # 404 on model
    SERVER_ERROR         = "server_error"          # 5xx from provider
    NETWORK_ERROR        = "network_error"         # connection timeout / DNS
    UNKNOWN_BACKEND      = "unknown_backend"       # backend value not recognised
    SDK_MISSING          = "sdk_missing"           # openai package not installed
    UNEXPECTED           = "unexpected"            # anything else


class AIResult(NamedTuple):
    text:   str | None
    error:  AIError
    detail: str        # safe for display — never contains keys or tokens


def _ok(text: str) -> AIResult:
    return AIResult(text=text, error=AIError.OK, detail="")


def _err(code: AIError, detail: str) -> AIResult:
    logger.warning("AI explanation error [%s]: %s", code.value, detail)
    return AIResult(text=None, error=code, detail=detail)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(
    prediction_min: float,
    distance_km: float,
    traffic: str,
    weather: str,
    vehicle_condition: int,
    multiple_deliveries: int,
    rider_rating: float,
    rider_age: int,
    city: str,
    festival: str,
    mae: float,
) -> str:
    vc_label = {0: "poor", 1: "fair", 2: "good", 3: "excellent"}.get(
        int(vehicle_condition), str(vehicle_condition)
    )
    return f"""You are a helpful assistant that explains food delivery time predictions.

A Random Forest machine learning model predicted that this delivery will take approximately {prediction_min:.0f} minutes (typical error: \xb1{mae} minutes).

Here are the order details the model used:
- Delivery distance: {distance_km:.1f} km
- Traffic density: {traffic}
- Weather condition: {weather}
- Vehicle condition: {vc_label} ({vehicle_condition}/3)
- Simultaneous deliveries by the rider: {multiple_deliveries}
- Rider rating: {rider_rating:.1f}/5.0
- Rider age: {rider_age} years
- City type: {city}
- Festival period: {festival}

Write a short, friendly explanation (3-5 sentences) for a customer. Cover:
1. The main factors that likely contributed to this estimate.
2. Any conditions that may increase or decrease the time.
3. A clear reminder that this is a model-based estimate, not a guarantee.

Important rules:
- Do not claim that any factor *causes* a longer or shorter delivery. Use "is associated with" or "the model weighted".
- Do not repeat the exact numeric prediction in every sentence.
- Keep the tone helpful and reassuring.
- Do not use bullet points. Write in plain paragraphs only.
"""


# ---------------------------------------------------------------------------
# IAM token helper (watsonx.ai)
# ---------------------------------------------------------------------------

def _get_iam_token(api_key: str) -> tuple[str | None, AIResult | None]:
    """
    Exchanges an IBM Cloud API key for a short-lived IAM bearer token.
    Returns (token, None) on success or (None, AIResult) on failure.
    """
    try:
        resp = requests.post(
            WX_IAM_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                "apikey": api_key,
            },
            timeout=15,
        )
        if resp.status_code == 400:
            return None, _err(AIError.AUTH_FAILED,
                "IBM Cloud API key was rejected (HTTP 400). "
                "Check that the key is valid and has not expired.")
        if resp.status_code == 401:
            return None, _err(AIError.AUTH_FAILED,
                "IBM Cloud API key authentication failed (HTTP 401). "
                "Verify the key in your secrets/environment variables.")
        resp.raise_for_status()
        return resp.json()["access_token"], None
    except requests.exceptions.ConnectionError:
        return None, _err(AIError.NETWORK_ERROR,
            "Could not reach IBM Cloud IAM (iam.cloud.ibm.com). "
            "Check your internet connection.")
    except requests.exceptions.Timeout:
        return None, _err(AIError.NETWORK_ERROR,
            "IBM Cloud IAM request timed out after 15 s. Try again.")
    except Exception as exc:
        return None, _err(AIError.UNEXPECTED,
            f"Unexpected error obtaining IAM token: {type(exc).__name__}")


# ---------------------------------------------------------------------------
# Backend: IBM watsonx.ai
# ---------------------------------------------------------------------------

def _call_watsonx(
    prompt: str,
    api_key: str,
    project_id: str,
    url: str,
    model_id: str = WX_DEFAULT_MODEL,
    max_new_tokens: int = 250,
) -> AIResult:
    token, err = _get_iam_token(api_key)
    if err is not None:
        return err

    endpoint = f"{url.rstrip('/')}/ml/v1/text/generation?version={WX_API_VERSION}"
    payload = {
        "model_id":   model_id,
        "project_id": project_id,
        "input":      prompt,
        "parameters": {
            "decoding_method":    "greedy",
            "max_new_tokens":     max_new_tokens,
            "min_new_tokens":     40,
            "repetition_penalty": 1.1,
        },
    }

    try:
        resp = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type":  "application/json",
                "Accept":        "application/json",
            },
            json=payload,
            timeout=30,
        )

        if resp.status_code == 401:
            return _err(AIError.AUTH_FAILED,
                "watsonx.ai rejected the request (HTTP 401). "
                "The IAM token may have expired — try again.")
        if resp.status_code == 403:
            return _err(AIError.AUTH_FAILED,
                "watsonx.ai returned HTTP 403 Forbidden. "
                "Check that your project_id is correct and the API key has access.")
        if resp.status_code == 404:
            return _err(AIError.MODEL_NOT_FOUND,
                f"Model '{model_id}' was not found in your watsonx.ai project (HTTP 404). "
                "Check the model_id in your secrets file.")
        if resp.status_code == 429:
            return _err(AIError.RATE_LIMITED,
                "watsonx.ai rate limit reached (HTTP 429). Wait a moment and try again.")
        if resp.status_code >= 500:
            return _err(AIError.SERVER_ERROR,
                f"watsonx.ai returned a server error (HTTP {resp.status_code}). "
                "The service may be temporarily unavailable.")

        resp.raise_for_status()
        text = resp.json()["results"][0]["generated_text"].strip()
        if not text:
            return _err(AIError.UNEXPECTED, "watsonx.ai returned an empty response.")
        return _ok(text)

    except requests.exceptions.ConnectionError:
        return _err(AIError.NETWORK_ERROR,
            f"Could not reach watsonx.ai at '{url}'. Check the url in your secrets file "
            "and your internet connection.")
    except requests.exceptions.Timeout:
        return _err(AIError.NETWORK_ERROR,
            "watsonx.ai request timed out after 30 s. The service may be overloaded.")
    except (KeyError, IndexError, ValueError) as exc:
        return _err(AIError.UNEXPECTED,
            f"Unexpected response format from watsonx.ai: {type(exc).__name__}")
    except Exception as exc:
        return _err(AIError.UNEXPECTED,
            f"Unexpected error calling watsonx.ai: {type(exc).__name__}")


# ---------------------------------------------------------------------------
# Backend: OpenAI (or any OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------

def _call_openai(
    prompt: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    base_url: str | None = None,
    max_tokens: int = 250,
) -> AIResult:
    try:
        import openai
    except ImportError:
        return _err(AIError.SDK_MISSING,
            "The 'openai' Python package is not installed. "
            "Run: pip install openai>=1.30.0")

    try:
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url

        client = openai.OpenAI(**kwargs)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        text = response.choices[0].message.content
        if not text or not text.strip():
            return _err(AIError.UNEXPECTED, "The AI model returned an empty response.")
        return _ok(text.strip())

    except openai.AuthenticationError:
        endpoint_hint = f" (endpoint: {base_url})" if base_url else ""
        return _err(AIError.AUTH_FAILED,
            f"OpenAI API key was rejected{endpoint_hint}. "
            "Verify the key in your secrets file or environment variables.")
    except openai.NotFoundError:
        return _err(AIError.MODEL_NOT_FOUND,
            f"Model '{model}' was not found. "
            "Check the model name in your secrets file.")
    except openai.RateLimitError:
        return _err(AIError.RATE_LIMITED,
            "OpenAI rate limit reached. Wait a moment and try again.")
    except openai.APIConnectionError:
        endpoint_hint = base_url or "api.openai.com"
        return _err(AIError.NETWORK_ERROR,
            f"Could not connect to '{endpoint_hint}'. "
            "Check the base_url in your secrets file and your internet connection.")
    except openai.APITimeoutError:
        return _err(AIError.NETWORK_ERROR,
            "OpenAI request timed out. The endpoint may be overloaded.")
    except openai.APIStatusError as exc:
        return _err(AIError.SERVER_ERROR,
            f"OpenAI API error (HTTP {exc.status_code}): {exc.message}")
    except Exception as exc:
        return _err(AIError.UNEXPECTED,
            f"Unexpected error calling OpenAI: {type(exc).__name__}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_ai_explanation(
    *,
    prediction_min: float,
    distance_km: float,
    traffic: str,
    weather: str,
    vehicle_condition: int,
    multiple_deliveries: int,
    rider_rating: float,
    rider_age: int,
    city: str,
    festival: str,
    mae: float,
    credentials: dict,
) -> AIResult:
    """
    Returns an AIResult describing either the generated explanation or the
    specific reason it could not be produced.

    Parameters
    ----------
    prediction_min : float
        Delivery time predicted by the ML model (minutes).
    distance_km, traffic, weather, vehicle_condition, multiple_deliveries,
    rider_rating, rider_age, city, festival :
        Human-readable input values shown to the user.
    mae : float
        Model test MAE, used to frame the uncertainty statement.
    credentials : dict
        Shape — see load_credentials() docstring.

    Returns
    -------
    AIResult
        .text   : str | None — explanation text or None
        .error  : AIError    — OK if successful, otherwise a specific code
        .detail : str        — safe, user-displayable error description
    """
    if not credentials:
        return _err(AIError.NOT_CONFIGURED,
            "No AI credentials are configured. "
            "Add your API key to .streamlit/secrets.toml or set environment variables.")

    backend = credentials.get("backend", "").lower()
    api_key = credentials.get("api_key", "").strip()

    if not api_key:
        return _err(AIError.MISSING_FIELD,
            "AI credentials found but 'api_key' is empty. "
            "Check your .streamlit/secrets.toml file.")

    if not backend:
        return _err(AIError.MISSING_FIELD,
            "AI credentials found but 'backend' is not set. "
            "Set backend to 'watsonx' or 'openai'.")

    prompt = _build_prompt(
        prediction_min=prediction_min,
        distance_km=distance_km,
        traffic=traffic,
        weather=weather,
        vehicle_condition=vehicle_condition,
        multiple_deliveries=multiple_deliveries,
        rider_rating=rider_rating,
        rider_age=rider_age,
        city=city,
        festival=festival,
        mae=mae,
    )

    if backend == "watsonx":
        project_id = credentials.get("project_id", "").strip()
        url        = credentials.get("url", "https://us-south.ml.cloud.ibm.com").strip()
        model_id   = credentials.get("model_id", WX_DEFAULT_MODEL)
        if not project_id:
            return _err(AIError.MISSING_FIELD,
                "watsonx backend is selected but 'project_id' is empty. "
                "Add your watsonx project ID to .streamlit/secrets.toml.")
        if not url:
            return _err(AIError.MISSING_FIELD,
                "watsonx backend is selected but 'url' is empty. "
                "Example: https://us-south.ml.cloud.ibm.com")
        return _call_watsonx(prompt, api_key, project_id, url, model_id)

    if backend == "openai":
        model    = credentials.get("model", "gpt-4o-mini")
        base_url = credentials.get("base_url") or None
        return _call_openai(prompt, api_key, model, base_url)

    return _err(AIError.UNKNOWN_BACKEND,
        f"Unknown AI backend '{backend}'. Supported values: 'watsonx', 'openai'.")


# ---------------------------------------------------------------------------
# Credential loader — called from app.py
# ---------------------------------------------------------------------------

def load_credentials() -> dict:
    """
    Resolves AI credentials from Streamlit secrets or environment variables.
    Returns an empty dict if nothing is configured.

    Streamlit secrets (.streamlit/secrets.toml):
        [ai]
        backend    = "watsonx"
        api_key    = "your-ibm-cloud-api-key"
        project_id = "your-watsonx-project-id"
        url        = "https://us-south.ml.cloud.ibm.com"
        model_id   = "ibm/granite-3-3-8b-instruct"    # optional

        OR

        [ai]
        backend = "openai"
        api_key = "sk-..."
        model   = "gpt-4o-mini"    # optional
        base_url = ""              # optional — for compatible endpoints

    Environment variables (fallback):
        AI_BACKEND, AI_API_KEY, AI_PROJECT_ID, AI_URL, AI_MODEL_ID,
        AI_MODEL, AI_BASE_URL
    """
    # ---- Try Streamlit secrets first ----
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "ai" in st.secrets:
            creds = dict(st.secrets["ai"])
            if creds.get("api_key", "").strip():
                return creds
    except Exception:
        pass  # running outside Streamlit context

    # ---- Fall back to environment variables ----
    backend = os.environ.get("AI_BACKEND", "").strip().lower()
    api_key = os.environ.get("AI_API_KEY", "").strip()

    if not backend or not api_key:
        return {}

    creds: dict = {"backend": backend, "api_key": api_key}
    if backend == "watsonx":
        creds["project_id"] = os.environ.get("AI_PROJECT_ID", "").strip()
        creds["url"]        = os.environ.get("AI_URL", "https://us-south.ml.cloud.ibm.com").strip()
        creds["model_id"]   = os.environ.get("AI_MODEL_ID", WX_DEFAULT_MODEL).strip()
    elif backend == "openai":
        creds["model"]    = os.environ.get("AI_MODEL", "gpt-4o-mini").strip()
        creds["base_url"] = os.environ.get("AI_BASE_URL", "").strip() or None

    return creds


# ---------------------------------------------------------------------------
# AI status probe — used by the app sidebar to show a live status badge
# ---------------------------------------------------------------------------

def probe_ai_status(credentials: dict) -> AIResult:
    """
    Sends a minimal test request to verify that credentials and the endpoint
    are reachable.  Used only for the sidebar status indicator.

    Returns an AIResult whose .error field indicates the status:
      OK              — credentials validated and endpoint reachable
      NOT_CONFIGURED  — no credentials
      AUTH_FAILED     — key rejected
      NETWORK_ERROR   — cannot reach endpoint
      (any other)     — specific error
    """
    if not credentials:
        return _err(AIError.NOT_CONFIGURED, "No AI credentials configured.")

    backend = credentials.get("backend", "").lower()
    api_key = credentials.get("api_key", "").strip()

    if not api_key:
        return _err(AIError.MISSING_FIELD, "api_key is empty.")

    if backend == "watsonx":
        # Validate by exchanging the API key for an IAM token only — cheaper than a full inference call
        token, err = _get_iam_token(api_key)
        if err is not None:
            return err
        return AIResult(
            text="watsonx.ai credentials validated (IAM token obtained).",
            error=AIError.OK,
            detail="",
        )

    if backend == "openai":
        # Validate by listing models (lightweight GET, no tokens consumed)
        try:
            import openai
        except ImportError:
            return _err(AIError.SDK_MISSING, "openai package not installed.")
        try:
            kwargs: dict = {"api_key": api_key}
            base_url = credentials.get("base_url") or None
            if base_url:
                kwargs["base_url"] = base_url
            client = openai.OpenAI(**kwargs)
            client.models.list()
            model = credentials.get("model", "gpt-4o-mini")
            return AIResult(
                text=f"OpenAI credentials validated (model: {model}).",
                error=AIError.OK,
                detail="",
            )
        except openai.AuthenticationError:
            return _err(AIError.AUTH_FAILED, "OpenAI API key was rejected.")
        except openai.APIConnectionError:
            base_url = credentials.get("base_url") or "api.openai.com"
            return _err(AIError.NETWORK_ERROR,
                f"Cannot reach '{base_url}'. Check base_url and your connection.")
        except Exception as exc:
            return _err(AIError.UNEXPECTED, f"{type(exc).__name__}")

    return _err(AIError.UNKNOWN_BACKEND,
        f"Unknown backend '{backend}'. Supported: 'watsonx', 'openai'.")
