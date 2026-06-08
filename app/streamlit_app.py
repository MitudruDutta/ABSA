"""
Streamlit demo - talks to the FastAPI service over HTTP. The model lives only in the API; this is a thin client.

Run the API first:   uvicorn app.api:app --port 8000
Then the UI:         streamlit run app/streamlit_app.py
Or both via:         docker compose up
"""

import os
import requests
import streamlit as st

API_URL = os.environ.get("ABSA_API_URL", "http://localhost:8000")

st.set_page_config(page_title="ABSA Ticket Router", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2.4rem; max-width: 1120px;}
.chip {display:inline-block; padding:5px 13px; border-radius:6px; font-size:0.85rem;
       font-weight:600; margin:3px 4px; letter-spacing:.2px;}
.neg {background:#fdecea; color:#b3261e; border:1px solid #f4c7c3;}
.neu {background:#fff5e0; color:#8a6100; border:1px solid #ffe3a3;}
.absent {background:#f3f4f6; color:#9aa0a6; border:1px solid #e5e7eb;}
.routecard {background:#0f2a52; color:#fff; padding:20px 24px; border-radius:10px;}
.routecard .label {font-size:0.72rem; letter-spacing:1.5px; opacity:.7;}
.routecard h2 {color:#fff; margin:.2rem 0 0; font-size:1.5rem; font-weight:700;}
.badge {display:inline-block; padding:3px 11px; border-radius:5px; font-weight:700; font-size:0.8rem;}
.b-high {background:#b3261e; color:#fff;}
.b-medium {background:#8a6100; color:#fff;}
.b-low {background:#1e7a3c; color:#fff;}
.subtle {color:#6b7280; font-size:0.88rem;}
.sectionlabel {font-size:0.78rem; letter-spacing:1px; color:#6b7280; font-weight:700;
               text-transform:uppercase; margin:1.1rem 0 .4rem;}
</style>
""", unsafe_allow_html=True)

st.title("Aspect-Based Support-Ticket Router")
st.markdown('<p class="subtle">A fine-tuned RoBERTa model decomposes each ticket into '
            'per-aspect sentiment, then routes and prioritises it. The model is served '
            'by a FastAPI backend; this interface is a thin HTTP client.</p>',
            unsafe_allow_html=True)

try:
    h = requests.get(f"{API_URL}/health", timeout=3).json()
    aspects = h.get("aspects", [])
except Exception:
    st.error("Backend unavailable. The model service is not responding - please try again shortly.")
    st.stop()

EXAMPLES = {
    "Select an example ticket": "",
    "Billing and crash": "I was overcharged on my invoice and the app keeps crashing during checkout.",
    "Security breach": "Unauthorized access detected on our network, sensitive customer data may be exposed.",
    "Feature question": "Could you tell me how to integrate your REST API with our analytics dashboard?",
    "Hardware fault": "My laptop screen keeps flickering after the latest firmware update, please help.",
}

col_in, col_out = st.columns([1, 1], gap="large")

with col_in:
    ex = st.selectbox("Load an example", list(EXAMPLES.keys()))
    body = st.text_area("Ticket text", value=EXAMPLES[ex], height=210,
                        placeholder="Paste a support ticket here...")
    go = st.button("Analyze ticket", type="primary", use_container_width=True)

with col_out:
    if go and body.strip():
        try:
            r = requests.post(f"{API_URL}/predict", json={"body": body}, timeout=30).json()
        except Exception as e:
            st.error(f"API request failed: {e}")
            st.stop()

        prio = r["priority"]
        st.markdown(f"""
        <div class="routecard">
          <div class="label">ROUTE TO</div>
          <h2>{r['route']}</h2>
          <div style="margin-top:10px;">Priority
            <span class="badge b-{prio}">{prio.upper()}</span></div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sectionlabel">Per-aspect sentiment</div>', unsafe_allow_html=True)
        cls = {"negative": "neg", "neutral": "neu", "not_present": "absent"}
        chips = "".join(
            f'<span class="chip {cls[r["aspects"][a]]}">{a}: {r["aspects"][a]}</span>'
            for a in aspects)
        st.markdown(chips, unsafe_allow_html=True)

        st.markdown('<div class="sectionlabel">Summary</div>', unsafe_allow_html=True)
        if r["negatives"]:
            st.markdown(f"**Complaints about:** {', '.join(r['negatives'])}")
        elif r["present"]:
            st.markdown(f"**Mentioned (neutral):** {', '.join(r['present'])}")
        else:
            st.markdown("**No specific aspect detected** - general inquiry.")
    else:
        st.info("Enter a ticket and select Analyze to see the routing decision.")

with st.expander("How it works - model and metrics"):
    st.markdown("""
- **Model:** RoBERTa sentence-pair classifier - `[ticket] [SEP] [aspect definition]`
  to `not_present / negative / neutral`, run once per aspect.
- **Test scores (held-out, sklearn-verified):** macro-F1 0.843, weighted-F1 0.944,
  aspect-detection F1 0.864, polarity accuracy 0.986.
- **Routing:** ticket to aspects to the queue of the first negative aspect;
  priority is high (two or more complaints), medium (one), or low.
- **Architecture:** FastAPI hosts the model; this Streamlit interface calls it over HTTP.
""")
