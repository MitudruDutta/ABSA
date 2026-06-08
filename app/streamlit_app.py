"""
Streamlit demo for the ABSA support-ticket router.

Paste a ticket -> see per-aspect sentiment, the routing decision, and priority.

Run: streamlit run app/streamlit_app.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from app.predict import predict
from taxonomy import ASPECTS, ASPECT_DEFINITIONS

st.set_page_config(page_title="ABSA Ticket Router", page_icon="🎫", layout="centered")
st.title("🎫 Aspect-Based Support-Ticket Router")
st.caption("RoBERTa sentence-pair ABSA — per-aspect sentiment, routing, and priority.")

SENT_COLOR = {"negative": "🔴", "neutral": "🟡", "not_present": "⚪"}

EXAMPLES = {
    "— pick an example —": "",
    "Billing + crash": "I was overcharged on my invoice and the app keeps crashing during checkout.",
    "Security breach": "Unauthorized access detected on our network, sensitive data may be exposed.",
    "Feature inquiry": "Could you tell me how to integrate your API with our dashboard?",
    "Hardware issue": "My laptop screen keeps flickering after the latest firmware update.",
}

ex = st.selectbox("Try an example", list(EXAMPLES.keys()))
body = st.text_area("Ticket text", value=EXAMPLES[ex], height=140,
                    placeholder="Paste a support ticket...")

if st.button("Analyze", type="primary") and body.strip():
    with st.spinner("Analyzing..."):
        r = predict(body)

    c1, c2 = st.columns(2)
    c1.metric("Route to", r["route"])
    c2.metric("Priority", r["priority"].upper())

    st.subheader("Per-aspect sentiment")
    cols = st.columns(3)
    for i, a in enumerate(ASPECTS):
        s = r["aspects"][a]
        with cols[i % 3]:
            st.markdown(f"{SENT_COLOR[s]} **{a}** — {s}")

    if r["negatives"]:
        st.error("Complaints about: " + ", ".join(r["negatives"]))
    elif r["present"]:
        st.info("Mentioned (neutral): " + ", ".join(r["present"]))
    else:
        st.success("No specific aspect detected — general inquiry.")

with st.expander("Aspect taxonomy"):
    for a in ASPECTS:
        st.markdown(f"**{a}** — {ASPECT_DEFINITIONS[a]}")
