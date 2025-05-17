import gradio as gr
import pandas as pd
from cachetools import TTLCache
# switch back to your real pipeline when ready
# from pipeline.dummy import run_once
from pipeline import run_once

cache = TTLCache(maxsize=1, ttl=60)


def refresh():
    if "frames" not in cache:
        cache["frames"] = run_once()
    return cache["frames"]

# ─────────────────────────  Chat function  ──


def chat_fn(message, history):
    bot = agent_builder.build(refresh()["ratios"])
    return bot.run(message)


# ─────────────────────────  UI  ─────────────
empty_df = pd.DataFrame({"x": [0], "y": [0]})      # harmless starter plot

with gr.Blocks(title="MSME Financial Dashboard") as demo:
    with gr.Tabs(elem_id="tabs"):
        # ---------- Cash-flow tab ----------
        with gr.TabItem("Cash Flow"):
            kpi = gr.Markdown("Loading…")
            line = gr.LinePlot(value=empty_df, x="x", y="y")  # ⇦ FIX A
        # ---------- Chatbot tab ------------
        with gr.TabItem("Chatbot"):
            # instantiate directly; no .render()  ← FIX B
            gr.ChatInterface(
                chat_fn,
                title="💬 MSME Finance Copilot",
                type="messages"
            )

    # ----- server-side updater -------------
    # dashboard/app.py  (keep the rest of the file the same)

    def dash_update():
        print("⋯ dash_update start")
        data = refresh()
        print("✔ data loaded")

        cf = data["cashflow"]
        if cf.empty:
            return "⚠️ No data", line.update(value=None)

        net = cf["net_cash"].iloc[-1]
        runway = data["burn_rate_months"]
        kpi_md = f"**Net burn:** RM {net:,.0f}<br>**Runway:** {runway:.1f} months"

        cf_plot = cf.astype({
            "cash_in":  float, "cash_out": float,
            "net_cash": float, "cum_cash": float
        })

        # 👉 use the *instance* method, not the class
        return kpi_md, line.update(
            value=cf_plot,
            x="month",
            y=["cash_in", "cash_out", "cum_cash"]
        )

    demo.load(dash_update, outputs=[kpi, line], every=5)

if __name__ == "__main__":
    demo.launch(share=False)
