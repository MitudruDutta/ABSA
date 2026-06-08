"""
Streamlit demo — talks to the FastAPI service over HTTP (does NOT load the
model itself). The model lives only in the API; this is a thin client.

Run the API first:   uvicorn app.api:app --port 8000
Then the UI:         streamlit run app/streamlit_app.py
Or both via:         docker compose up
"""

import os
import requests
import streamlit as st

API_URL = os.environ.get("ABSA_API_URL", "http://localhost:8000")

st.set_page_config(page_title="ABSA Ticket Router", page_icon="🎫", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1100px;}
.chip {display:inline-block; padding:5px 13px; border-radius:14px; font-size:0.86rem;
       font-weight:600; margin:3px 4px;}
.neg {background:#fde8e8; color:#c0392b; border:1px solid #f5c6cb;}
.neu {background:#fff6e0; color:#9a6b00; border:1px solid #ffe2a8;}
.absent {background:#f1f3f5; color:#aeb4bb; border:1px solid #e6e8eb;}
.routecard {background:linear-gradient(135deg,#1e3a8a,#2563eb); color:#fff;
            padding:18px 22px; border-radius:14px;}
.routecard h2 {color:#fff; margin:0; font-size:1.5rem;}
.prio-high {color:#c0392b; font-weight:700;}
.prio-medium {color:#9a6b00; font-weight:700;}
.prio-low {color:#2e7d32; font-weight:700;}
.small {color:#8a9099; font-size:0.85rem;}
</style>
""", unsafe_allow_html=True)

st.title("🎫 Aspect-Based Support-Ticket Router")
st.markdown('<p class="small">Fine-tuned RoBERTa decomposes a ticket into per-aspect '
            'sentiment, then routes & prioritises it. The model is served by a '
            'FastAPI backend — this UI is a thin HTTP client.</p>',
            unsafe_allow_html=True)

# API health
try:
    h = requests.get(f"{API_URL}/health", timeout=3).json()
    st.success(f"🟢 Connected to API at {API_URL}", icon="✅")
    aspects = h.get("aspects", [])
except Exception:
    st.error(f"🔴 API not reachable at {API_URL}. Start it: `uvicorn app.api:app --port 8000`")
    aspects = []
    st.stop()

EXAMPLES = {
    "— example tickets —": "",
    "💳 Billing + crash": "I was overcharged on my invoice and the app keeps crashing during checkout.",
    "🔒 Security breach": "Unauthorized access detected on our network, sensitive customer data may be exposed.",
    "❓ Feature question": "Could you tell me how to integrate your REST API with our analytics dashboard?",
    "🖥️ Hardware fault": "My laptop screen keeps flickering after the latest firmware update, please help.",
}

col_in, col_out = st.columns([1, 1], gap="large")

with col_in:
    ex = st.selectbox("Load an example", list(EXAMPLES.keys()))
    body = st.text_area("Ticket text", value=EXAMPLES[ex], height=200,
                        placeholder="Paste a support ticket here…")
    go = st.button("🚀 Analyze ticket", type="primary", use_container_width=True)

with col_out:
    if go and body.strip():
        try:
            r = requests.post(f"{API_URL}/predict", json={"body": body}, timeout=30).json()
        except Exception as e:
            st.error(f"API request failed: {e}"); st.stop()

        prio = r["priority"]
        st.markdown(f"""
        <div class="routecard">
          <div style="font-size:0.8rem;opacity:.85;">ROUTE TO</div>
          <h2>{r['route']}</h2>
          <div style="margin-top:8px;">Priority:
            <span class="prio-{prio}">{prio.upper()}</span></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("##### Per-aspect sentiment")
        chips = ""
        cls = {"negative": "neg", "neutral": "neu", "not_present": "absent"}
        for a in aspects:
            s = r["aspects"][a]
            chips += f'<span class="chip {cls[s]}">{a}: {s}</span>'
        st.markdown(chips, unsafe_allow_html=True)

        if r["negatives"]:
            st.markdown(f"**🔴 Complaints:** {', '.join(r['negatives'])}")
        elif r["present"]:
            st.markdown(f"**🟡 Mentioned (neutral):** {', '.join(r['present'])}")
        else:
            st.markdown("**⚪ No specific aspect** — general inquiry.")
    else:
        st.info("Enter a ticket and click **Analyze** to see the routing decision.")

with st.expander("ℹ️  How it works · model & metrics"):
    st.markdown("""
- **Model:** RoBERTa sentence-pair classifier — `[ticket] [SEP] [aspect definition]`
  → `not_present / negative / neutral`, run once per aspect.
- **Test scores (held-out, sklearn-verified):** macro-F1 **0.843**, weighted-F1
  **0.944**, aspect-detection F1 **0.864**, polarity accuracy **0.986**.
- **Routing:** ticket → aspects → queue of the first negative aspect;
  priority = high (≥2 complaints) / medium (1) / low.
- **Architecture:** FastAPI hosts the model; this Streamlit UI calls it over HTTP.
""")
