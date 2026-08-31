"""Minimal Streamlit frontend for the Fraud Detection FastAPI service.

Run with:
    streamlit run app.py

The FastAPI contract remains:
    POST {FRAUD_API_URL}/predict
    {
        "transaction_id": str,
        "features": [Time, V1, ..., V28, Amount]  # exactly 30 floats
    }
"""

from __future__ import annotations

import math
import os
import re
from typing import Any

import requests
import streamlit as st


# Keep the backend URL configurable without adding a sidebar or extra UI.
API_URL = os.getenv("FRAUD_API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("API_KEY", "")
API_HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}
PREDICT_URL = f"{API_URL}/predict"


st.set_page_config(
    page_title="Fraud Detection",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed",
)


# One UI accent color. Green/red are reserved for the semantic prediction state.
st.markdown(
    """
    <style>
        :root {
            --accent: #2563eb;
            --text: #172033;
            --muted: #667085;
            --border: #d9dee8;
            --success: #15803d;
            --success-bg: #ecfdf3;
            --danger: #b42318;
            --danger-bg: #fff1f0;
        }

        [data-testid="stAppViewContainer"] {
            background: #ffffff;
        }

        .block-container {
            max-width: 680px;
            padding: 2.5rem 1rem 3rem;
        }

        h1, h2, h3, p, label, input, textarea, button {
            font-family: Arial, sans-serif;
            color: var(--text);
        }

        h1 {
            font-size: 28px;
            line-height: 1.2;
            margin-bottom: 0.35rem;
        }

        h2, h3, p, label, input, textarea, button {
            font-size: 15px;
        }

        .subtitle {
            color: var(--muted);
            margin: 0 0 1.75rem;
        }

        .section-label {
            color: var(--text);
            font-weight: 700;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.65rem;
            margin: 1.5rem 0 1rem;
        }

        .status-card {
            border-radius: 12px;
            border: 1px solid;
            padding: 1.2rem 1.25rem;
            margin: 0.75rem 0 1rem;
        }

        .status-card.fraud {
            color: var(--danger);
            background: var(--danger-bg);
            border-color: #f3b5ae;
        }

        .status-card.safe {
            color: var(--success);
            background: var(--success-bg);
            border-color: #9be0b5;
        }

        .status-title {
            color: inherit;
            font-size: 15px;
            font-weight: 700;
            margin: 0 0 0.55rem;
        }

        .status-confidence {
            color: inherit;
            font-size: 15px;
            margin: 0;
        }

        [data-testid="stFormSubmitButton"] button {
            background: var(--accent);
            border-color: var(--accent);
            color: #ffffff;
            min-height: 2.75rem;
        }

        [data-testid="stFormSubmitButton"] button:hover {
            background: #1d4ed8;
            border-color: #1d4ed8;
            color: #ffffff;
        }

        [data-testid="stExpander"] {
            border-color: var(--border);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("Fraud Detection")

# Quiet API availability indicator; no sidebar or dashboard clutter.
try:
    health_response = requests.get(f"{API_URL}/health", headers=API_HEADERS, timeout=2)
    api_online = health_response.ok
except requests.RequestException:
    api_online = False
st.caption("API status: " + ("Online" if api_online else "Unavailable"))
st.markdown(
    '<p class="subtitle">Submit a transaction to receive a clear fraud decision.</p>',
    unsafe_allow_html=True,
)


def parse_pca_features(raw_value: str) -> list[float]:
    """Parse the 28 PCA features while accepting commas, spaces, or newlines."""
    tokens = [token for token in re.split(r"[,\s]+", raw_value.strip()) if token]

    if len(tokens) != 28:
        raise ValueError(f"Enter exactly 28 values for V1–V28; received {len(tokens)}.")

    try:
        values = [float(token) for token in tokens]
    except ValueError as exc:
        raise ValueError("V1–V28 must contain numbers only.") from exc

    if not all(math.isfinite(value) for value in values):
        raise ValueError("V1–V28 must contain finite numbers only.")

    return values


def show_prediction(result: dict[str, Any]) -> None:
    """Render the prediction as a simple status card, not raw JSON."""
    is_fraud = bool(result.get("is_fraud", False))
    fraud_probability = float(result.get("fraud_probability", 0.0))

    # Confidence means confidence in the displayed decision.
    confidence = fraud_probability if is_fraud else 1.0 - fraud_probability
    card_class = "fraud" if is_fraud else "safe"
    title = "Fraud Detected" if is_fraud else "Not Fraud"

    st.markdown(
        f"""
        <div class="status-card {card_class}">
            <p class="status-title">{title}</p>
            <p class="status-confidence">Confidence: {confidence:.1%}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Keep secondary details available without adding clutter to the main view.
    with st.expander("View explanation"):
        risk_level = result.get("risk_level")
        decision_ms = result.get("decision_ms")

        if risk_level is not None:
            st.write(f"Risk level: {str(risk_level).capitalize()}")
        recommended_action = result.get("recommended_action")
        if recommended_action:
            action_label = str(recommended_action).replace("_", " ").capitalize()
            st.write(f"Recommended action: {action_label}")
        if decision_ms is not None:
            st.write(f"Decision time: {float(decision_ms):.2f} ms")
        st.write(f"Fraud probability: {fraud_probability:.1%}")

        reasons = result.get("top_reasons") or []
        if reasons:
            st.write("Main reasons")
            for reason in reasons:
                feature = reason.get("feature", "Feature")
                impact = str(reason.get("impact", "")).replace("_", " ")
                importance = reason.get("importance")
                st.write(f"{feature}: {impact}" + (f" ({importance})" if importance is not None else ""))
        if not reasons:
            st.caption(
                "Feature-level reasons appear here when ENABLE_SHAP=true is enabled on the API."
            )


with st.form("transaction_form", clear_on_submit=False):
    st.markdown('<p class="section-label">Transaction details</p>', unsafe_allow_html=True)

    transaction_id = st.text_input(
        "Transaction ID",
        placeholder="e.g. tx_001",
        help="A unique identifier used for tracing the prediction.",
    )

    time_value = st.number_input(
        "Time",
        value=0.0,
        step=1.0,
        format="%.6f",
        help="The transaction Time feature from the training dataset.",
    )

    amount_value = st.number_input(
        "Amount",
        value=0.0,
        min_value=0.0,
        step=1.0,
        format="%.6f",
        help="The transaction Amount feature from the training dataset.",
    )

    pca_values = st.text_area(
        "V1–V28",
        height=130,
        placeholder="Paste 28 numeric values separated by commas, spaces, or new lines",
        help="The backend expects the complete ordered feature vector: Time, V1–V28, Amount.",
    )

    predict_clicked = st.form_submit_button("Predict", type="primary", use_container_width=True)


if predict_clicked:
    if not transaction_id.strip():
        st.error("Enter a transaction ID.")
    else:
        try:
            middle_features = parse_pca_features(pca_values)
            features = [float(time_value), *middle_features, float(amount_value)]

            response = requests.post(
                PREDICT_URL,
                json={
                    "transaction_id": transaction_id.strip(),
                    "features": features,
                },
                headers=API_HEADERS,
                timeout=10,
            )
            response.raise_for_status()
            show_prediction(response.json())

        except ValueError as exc:
            st.error(str(exc))
        except requests.exceptions.ConnectionError:
            st.error("The fraud API is unavailable. Check that FastAPI is running.")
        except requests.exceptions.Timeout:
            st.error("The fraud API took too long to respond. Please try again.")
        except requests.exceptions.HTTPError as exc:
            detail = "The API rejected the request."
            try:
                detail = exc.response.json().get("detail", detail)
            except (ValueError, AttributeError):
                pass
            st.error(str(detail))
        except requests.exceptions.RequestException:
            st.error("Could not reach the fraud API. Please try again.")
        except (KeyError, TypeError, OverflowError) as exc:
            st.error(f"Unexpected API response: {exc}")
