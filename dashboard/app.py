import gradio as gr
import pandas as pd
from cachetools import TTLCache
import plotly.graph_objects as go

from pipeline import run_once
from chatbot.agent import build

# ────────────────────── Cache & helpers ─────────────────────
cache = TTLCache(maxsize=1, ttl=60)


def refresh():
    """Run the ETL pipeline once per minute & cache frames."""
    if "frames" not in cache:
        cache["frames"] = run_once()

    # Normalise month to YYYY-MM
    cf = cache["frames"]["cashflow"]
    if not cf.empty:
        cache["frames"]["cashflow"]["month"] = (
            pd.to_datetime(cf["month"], errors="coerce")
              .dt.to_period("M")
              .astype(str)
        )
    return cache["frames"]


# ────────────────────── Chat wrapper ────────────────────────
def chat_fn(message, history):
    bot = build(refresh())          # pass full metrics dict
    return bot.run(message)


# ────────────────────── Plotly gauge builder ────────────────
def gauge_fig(value: float, title: str, min_=0, max_=1, fmt="{:.2f}"):
    value = round(float(value), 2)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"valueformat": fmt},
        gauge={
            "axis": {"range": [min_, max_]},
            "bar": {"thickness": 0.35},
            "bgcolor": "rgba(0,0,0,0)",
            "threshold": {"line": {"width": 4}, "thickness": 0.75, "value": value},
        },
        title={"text": title},
    ))
    fig.update_layout(
        paper_bgcolor="#1e1e1e",
        font_color="#ffffff",
        margin=dict(t=30, b=10, l=10, r=10),
        height=260,
    )
    return fig


# ────────────────────── Gradio UI ───────────────────────────
empty_df = pd.DataFrame({"month": [""], "Value": [0], "Category": [""]})

css_dark = r"""
.chatbot{background-color:#121212!important;color:white!important;}
.chatbot .chat-message{background-color:#1e1e1e!important;color:white!important;}
.chatbot::-webkit-scrollbar{background-color:#121212!important;}
.chatbot .message.user,.chatbot .message.bot{max-width:90%;}
textarea{background-color:#222!important;color:#fff!important;}
#chatbot-container{display:flex;flex-direction:column;height:700px;}
#chatbot-container .chatbot{flex:1 1 auto;min-height:700px;}
.plot-container{background-color:#1e1e1e!important;}
"""

demo = gr.Blocks(css=css_dark, title="MSME Financial Dashboard")

with demo:
    with gr.Row(equal_height=True):

        # ───────── LEFT PANE ─────────
        with gr.Column(scale=2):
            refresh_btn = gr.Button("🔄 Refresh Data")
            kpi = gr.Markdown("Loading…")

            line = gr.LinePlot(
                value=empty_df,
                x="month",
                y="Value",
                color="Category",
                color_legend_title="Category",
                color_mapping={
                    "cash_in": "#4CAF50",
                    "cash_out": "#F44336",
                    "cum_cash": "#2196F3",
                },
                title="Cash Flow Over Time",
                height=400,
                tooltip=["month", "Value", "Category"],
                label="Monthly Cash Flow",
            )

            with gr.Row():
                cur_ratio_plot = gr.Plot(show_label=False)
                quick_ratio_plot = gr.Plot(show_label=False)
                dte_plot = gr.Plot(show_label=False)

            gr.Markdown("""
            ### 🧾 Legend  
            - <span style='color:#2196F3'>🔵 cash_in</span>  
            - <span style='color:#FFEB3B'>🟡 cash_out</span>  
            - <span style='color:#F44336'>🔴 cum_cash</span>
            """)

        # ───────── RIGHT PANE ─────────
        with gr.Column(scale=1, elem_id="chatbot-container"):
            gr.ChatInterface(
                chat_fn,
                title="💬 MSME Finance Copilot",
                type="messages",
            )

    # ───────────────── dashboard updater ─────────────────
    def dash_update():
        data = refresh()
        cf = data["cashflow"].copy()

        if cf.empty:
            msg = "⚠️ No data"
            return msg, empty_df, *[gauge_fig(0, "", 0, 1)]*3

        cf = cf.tail(6)  # always show latest six rows

        # KPI
        net, runway = cf["net_cash"].iloc[-1], data["burn_rate_months"]
        kpi_md = f"**Net burn:** RM {net:,.2f}<br>**Runway:** {runway:.2f} months"

        # line plot
        cf_plot = (
            cf[["month", "cash_in", "cash_out", "cum_cash"]]
            .round(2)
            .melt(id_vars="month", var_name="Category", value_name="Value")
        )

        # gauges
        r = data["ratios"]
        g1 = gauge_fig(r["current_ratio"],  "Current Ratio", 0, 3)
        g2 = gauge_fig(r["quick_ratio"],    "Quick Ratio",   0, 3)
        g3 = gauge_fig(r["debt_to_equity"], "Debt / Equity", 0, 3)

        return kpi_md, cf_plot, g1, g2, g3

    # UI event wiring
    refresh_btn.click(
        dash_update,
        inputs=[],
        outputs=[kpi, line, cur_ratio_plot, quick_ratio_plot, dte_plot],
    )
    demo.load(
        dash_update,
        inputs=[],
        outputs=[kpi, line, cur_ratio_plot, quick_ratio_plot, dte_plot],
    )

# ────────────────────── Launch ───────────────────────────────
if __name__ == "__main__":
    demo.launch(share=False)
