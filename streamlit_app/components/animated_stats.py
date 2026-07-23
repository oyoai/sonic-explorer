"""A real count-up animation for the Overview page's headline numbers.

st.markdown(..., unsafe_allow_html=True) strips <script> tags entirely, so a
real animation needs an actual iframe -- st.components.v1.html renders one,
with real JS execution. Colors are hardcoded to Streamlit's own dark-theme
tokens (page background #0e1117, primary text #fafafa, muted label #a3a8b8)
since the app is dark-theme-only (.streamlit/config.toml sets base="dark")
and an iframe doesn't inherit the host page's theme automatically."""

import html
import json

_BG = "#0e1117"
_TEXT = "#fafafa"
_LABEL = "#a3a8b8"
_FONT_STACK = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"


def animated_stat_row(stats: list[tuple[str, int]], duration_ms: int = 1200) -> str:
    """stats is a list of (label, value) pairs, rendered left-to-right.
    Returns a self-contained HTML string -- pass it to
    st.components.v1.html(result, height=...)."""
    columns_html = []
    animate_calls = []
    for i, (label, value) in enumerate(stats):
        elem_id = f"animated-stat-{i}"
        columns_html.append(f"""
        <div style="flex: 1; min-width: 0;">
          <div style="color: {_LABEL}; font-size: 0.85rem; font-family: {_FONT_STACK};">
            {html.escape(label)}
          </div>
          <div id="{elem_id}" style="color: {_TEXT}; font-size: 2.1rem; font-weight: 600;
                                      font-family: {_FONT_STACK}; line-height: 1.4;">0</div>
        </div>""")
        animate_calls.append(f"animateCount({json.dumps(elem_id)}, {int(value)}, {int(duration_ms)});")

    return f"""
    <div style="display: flex; gap: 2rem; background: {_BG}; padding: 0.25rem 0 0.75rem 0;">
      {"".join(columns_html)}
    </div>
    <script>
      function animateCount(elemId, target, duration) {{
        const el = document.getElementById(elemId);
        if (!el) return;
        const start = performance.now();
        function tick(now) {{
          const t = Math.min((now - start) / duration, 1);
          const eased = 1 - Math.pow(1 - t, 3);
          el.textContent = Math.floor(eased * target).toLocaleString();
          if (t < 1) {{
            requestAnimationFrame(tick);
          }} else {{
            el.textContent = target.toLocaleString();
          }}
        }}
        requestAnimationFrame(tick);
      }}
      {"".join(animate_calls)}
    </script>
    """
