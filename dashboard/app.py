import gradio as gr
import pandas as pd
from cachetools import TTLCache
from pipeline import run_once
from chatbot.agent import build

# ─────────────────────── Cache setup ─────────────────────────
cache = TTLCache(maxsize=1, ttl=60)

def refresh():
    if "frames" not in cache:
        cache["frames"] = run_once()
    return cache["frames"]

# ─────────────────────── Chat function ───────────────────────
def chat_fn(message, history):
    bot = build(refresh()["ratios"])
    return bot.run(message)

# ─────────────────────── Gradio UI ───────────────────────────
empty_df = pd.DataFrame({"month": [""], "Value": [0], "Category": [""]})  # Placeholder with correct columns

demo = gr.Blocks(css="""
.chatbot {
    background-color: #121212 !important;
    color: white !important;
    transition: none !important;
}
.chatbot .chat-message {
    background-color: #1e1e1e !important;
    color: white !important;
}
.chatbot::-webkit-scrollbar {
    background-color: #121212 !important;
}

/* Center the chatbot container horizontally and vertically */
#chatbot-container {
    display: flex;
    justify-content: center;  /* horizontal centering */
    align-items: center;      /* vertical centering */
    height: 700px;            /* container height */
    width: 100%;
}

/* Size the inner chatbot box */
#chatbot-container > div {
    width: 700px;
    height: 600px;
    box-sizing: border-box;
}
""", title="MSME Financial Dashboard")

with demo:

    with gr.Tabs(elem_id="tabs"):
        # ---------- Cash-flow tab ----------
        with gr.TabItem("Cash Flow"):
            kpi = gr.Markdown("Loading…")
            line = gr.LinePlot(
                value=empty_df,
                x="month",
                y="Value",
                color="Category",  # Matches melted dataframe column
                color_legend_title="Category",  # ✅ Ensure legend title appears
                color_mapping={  # ✅ Matches exact melted category names
                    "cash_in": "#4CAF50",
                    "cash_out": "#F44336",
                    "cum_cash": "#2196F3"
                },
                title="Cash Flow Over Time",
                height=400,
                tooltip=["month", "Value", "Category"],
                label="Monthly Cash Flow"
            )

            # 👇 Manual legend
            gr.Markdown("""
            ### 🧾 Legend
            - <span style='color:#2196F3'>🔵 cash_in</span>  
            - <span style='color:#FFEB3B'>🟡 cash_out</span>  
            - <span style='color:#F44336'>🔴 cum_cash</span>
            """)    

            refresh_btn = gr.Button("🔄 Refresh Data")  

        # ---------- Chatbot tab ------------
        with gr.TabItem("Chatbot"):
            with gr.Column(elem_id="chatbot-container"):
                gr.ChatInterface(
                    chat_fn,
                    title="💬 MSME Finance Copilot",
                    type="messages"
                )

    # ----- dashboard updater function -----
    def dash_update():
        print("⋯ dash_update start")
        data = refresh()
        print("✔ data loaded")

        cf = data["cashflow"]
        if cf.empty:
            return "⚠️ No data", empty_df

        net = cf["net_cash"].iloc[-1]
        runway = data["burn_rate_months"]
        kpi_md = f"**Net burn:** RM {net:,.0f}<br>**Runway:** {runway:.1f} months"

        # Format month and ensure numeric types
        cf["month"] = pd.to_datetime(cf["month"]).dt.strftime("%Y-%m")
        cf[["cash_in", "cash_out", "net_cash", "cum_cash"]] = cf[
            ["cash_in", "cash_out", "net_cash", "cum_cash"]
        ].apply(pd.to_numeric, errors='coerce')

        # Reshape data for multi-line plotting
        cols_to_plot = ["cash_in", "cash_out", "cum_cash"]
        cf_plot = cf[["month"] + cols_to_plot].melt(
            id_vars="month",
            value_vars=cols_to_plot,
            var_name="Category",  # ✅ Matches `color` argument in plot
            value_name="Value"
        )

        print(cf_plot.head())  # Optional debug

        return kpi_md, cf_plot  # Return the reshaped DataFrame directly


    refresh_btn.click(dash_update, outputs=[kpi, line])  
    demo.load(dash_update, outputs=[kpi, line])

# ─────────────────────── Launch App ──────────────────────────
if __name__ == "__main__":
    demo.launch(share=False)
