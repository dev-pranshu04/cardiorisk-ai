"""
Footer section.
"""
import streamlit as st


def render_footer() -> None:
    st.markdown("<hr style='margin-top:40px;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="max-width:580px;margin:0 auto;
        background:linear-gradient(135deg,#04080e,#070d1c);
        border:1px solid #0a1428;border-radius:20px;padding:28px 36px;text-align:center;">
        <div style="font-size:9px;font-weight:700;letter-spacing:2.5px;
            text-transform:uppercase;color:#1d4ed8;margin-bottom:8px;">Crafted by</div>
        <div style="font-size:22px;font-weight:700;color:#e2e8f0;letter-spacing:-0.3px;margin-bottom:3px;">Pranshu Kumar</div>
        <div style="font-size:11px;color:#1e3a5f;margin-bottom:22px;">ML Engineer &nbsp;\u00b7&nbsp; Healthcare AI &nbsp;\u00b7&nbsp; Python</div>
        <div style="display:flex;justify-content:center;gap:10px;flex-wrap:wrap;">
            <a href="https://github.com/dev-pranshu04" target="_blank"
               style="display:inline-flex;align-items:center;gap:7px;padding:9px 20px;
               border-radius:9px;font-size:12px;font-weight:500;text-decoration:none;
               background:#0a0f18;border:1px solid #141e30;color:#94a3b8;">
               GitHub
            </a>
            <a href="https://www.linkedin.com/in/dev-pranshu" target="_blank"
               style="display:inline-flex;align-items:center;gap:7px;padding:9px 20px;
               border-radius:9px;font-size:12px;font-weight:500;text-decoration:none;
               background:#0a66c2;color:#fff;">
               LinkedIn
            </a>
            <a href="https://dev-pranshu04.github.io/" target="_blank"
               style="display:inline-flex;align-items:center;gap:7px;padding:9px 20px;
               border-radius:9px;font-size:12px;font-weight:500;text-decoration:none;
               background:linear-gradient(135deg,#1d4ed8,#3b82f6);color:#fff;">
               Portfolio
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)
