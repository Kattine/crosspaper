"""
CrossPaper: Cross-field paper recommender with before/after model comparison.
Supports diversity-aware ranking and web UI via Gradio.
"""

import argparse
import json
import os

import gradio as gr
import plotly.graph_objects as go

from scripts.artifacts import ensure_artifacts
from scripts.recommender import CrossPaperRecommender


# Field color scheme
FIELD_COLORS = {
    "computer_science": "#6366f1",        # Indigo
    "neuroscience": "#a855f7",            # Purple
    "psychology": "#14b8a6",              # Teal
    "biochemistry_genetics": "#22c55e",   # Green
    "physics": "#f59e0b",                 # Amber
}

FIELD_LABELS = {
    "computer_science": "Computer Science",
    "neuroscience": "Neuroscience",
    "psychology": "Psychology",
    "biochemistry_genetics": "Biochem & Genetics",
    "physics": "Physics",
}


class CrossPaperApp:
    """Gradio app for the recommender with two models (base and fine-tuned)."""

    def __init__(self):
        """Load both base and fine-tuned models."""
        ensure_artifacts()

        print("Loading models...")
        self.recommender_ft = CrossPaperRecommender()
        self.recommender_ft.load(index_name="finetuned")

        self.recommender_base = CrossPaperRecommender()
        self.recommender_base.load(index_name="base")
        print("Models loaded.")

    def search(self, query, use_finetuned, lambda_param, top_n):
        """Run a query and return formatted results with diversity metrics."""
        if not query.strip():
            return "Please enter a research query.", None, ""

        recommender = self.recommender_ft if use_finetuned else self.recommender_base
        result = recommender.recommend(
            query, top_n=int(top_n), lambda_param=lambda_param
        )

        results_html = self._format_results_html(result["recommendations"])
        diversity_chart = self._build_diversity_chart(result["diversity"])
        metrics_text = self._format_metrics(result["diversity"], use_finetuned)

        return results_html, diversity_chart, metrics_text

    def compare(self, query, lambda_param, top_n):
        """Compare results from base and fine-tuned models on the same query."""
        if not query.strip():
            empty = "Please enter a research query."
            return empty, empty, None, ""

        base_result = self.recommender_base.recommend(
            query, top_n=int(top_n), lambda_param=lambda_param
        )
        ft_result = self.recommender_ft.recommend(
            query, top_n=int(top_n), lambda_param=lambda_param
        )

        before_html = self._format_results_html(base_result["recommendations"])
        after_html = self._format_results_html(ft_result["recommendations"])
        comparison_chart = self._build_comparison_chart(
            base_result["diversity"], ft_result["diversity"]
        )
        delta_text = self._format_delta(
            base_result["diversity"], ft_result["diversity"]
        )

        return before_html, after_html, comparison_chart, delta_text

    def _format_results_html(self, recommendations):
        """Format recommendations as styled HTML cards."""
        if not recommendations:
            return "<p>No results found.</p>"

        cards = []
        for i, rec in enumerate(recommendations, 1):
            field = rec["field"]
            color = FIELD_COLORS.get(field, "#6b7280")
            label = FIELD_LABELS.get(field, field)
            score_pct = int(rec["relevance_score"] * 100)

            card = f"""
            <div style="border:1px solid #e2e8f0; border-radius:10px;
                        padding:16px; margin:8px 0; border-left:4px solid {color};
                        background:#fafafa;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="background:{color}; color:white; padding:2px 10px;
                                 border-radius:12px; font-size:12px; font-weight:600;">
                        {label}
                    </span>
                    <span style="color:#64748b; font-size:12px;">
                        relevance {score_pct}% &middot; cited {rec['cited_by_count']}x
                        &middot; {rec['year']}
                    </span>
                </div>
                <h4 style="margin:8px 0 4px 0; font-size:15px; color:#1e293b;">
                    {i}. {rec['title']}
                </h4>
                <p style="color:#475569; font-size:13px; line-height:1.5; margin:4px 0 0 0;">
                    {rec['abstract']}...
                </p>
            </div>
            """
            cards.append(card)

        return "\n".join(cards)

    def _build_diversity_chart(self, diversity):
        """Build field distribution radar chart."""
        dist = diversity.get("distribution", {})
        all_fields = list(FIELD_LABELS.keys())

        values = [dist.get(d, 0) * 100 for d in all_fields]
        labels = [FIELD_LABELS[d] for d in all_fields]
        colors = [FIELD_COLORS[d] for d in all_fields]

        # Close the radar polygon
        values_closed = values + [values[0]]
        labels_closed = labels + [labels[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            fill="toself",
            fillcolor="rgba(99,102,241,0.15)",
            line=dict(color="#6366f1", width=2),
            name="Distribution",
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            margin=dict(l=40, r=40, t=30, b=30),
            height=300,
        )
        return fig

    def _build_comparison_chart(self, base_div, ft_div):
        """Build bar chart comparing base vs. fine-tuned model metrics."""
        metrics = ["Entropy", "Cross-Disc Rate", "Num Fields"]
        base_vals = [
            base_div["entropy"],
            base_div["cross_field_rate"] * 100,
            base_div["num_fields"],
        ]
        ft_vals = [
            ft_div["entropy"],
            ft_div["cross_field_rate"] * 100,
            ft_div["num_fields"],
        ]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Before (Base)", x=metrics, y=base_vals, marker_color="#94a3b8"
        ))
        fig.add_trace(go.Bar(
            name="After (Fine-tuned)", x=metrics, y=ft_vals, marker_color="#6366f1"
        ))
        fig.update_layout(
            barmode="group",
            margin=dict(l=40, r=40, t=30, b=30),
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        return fig

    def _format_metrics(self, diversity, is_finetuned):
        """Format diversity metrics as readable text."""
        model_label = "Fine-tuned" if is_finetuned else "Base"
        return (
            f"Model: {model_label}\n"
            f"Field Diversity (Shannon Entropy): {diversity['entropy']:.2f}\n"
            f"Cross-Field Rate: {diversity['cross_field_rate']:.0%}\n"
            f"Unique Fields in Top-10: {diversity['num_fields']}"
        )

    def _format_delta(self, base_div, ft_div):
        """Format improvement metrics comparing before and after."""
        e_delta = ft_div["entropy"] - base_div["entropy"]
        c_delta = ft_div["cross_field_rate"] - base_div["cross_field_rate"]
        n_delta = ft_div["num_fields"] - base_div["num_fields"]

        return (
            f"Entropy:     {base_div['entropy']:.2f} -> {ft_div['entropy']:.2f}  "
            f"({'+'if e_delta>=0 else ''}{e_delta:.2f})\n"
            f"Cross Rate:  {base_div['cross_field_rate']:.0%} -> "
            f"{ft_div['cross_field_rate']:.0%}  "
            f"({'+'if c_delta>=0 else ''}{c_delta:.0%})\n"
            f"Fields: {base_div['num_fields']} -> {ft_div['num_fields']}  "
            f"({'+'if n_delta>=0 else ''}{n_delta})"
        )

    def build_ui(self):
        """Build the Gradio interface with search and comparison tabs."""
        with gr.Blocks(
            title="CrossPaper",
            theme=gr.themes.Soft(primary_hue="indigo"),
        ) as app:
            gr.Markdown(
                "# CrossPaper\n"
                "Discover research across disciplinary boundaries. "
                "Fine-tuned on cross-disciplinary citation patterns to "
                "recommend papers your field might never show you."
            )

            with gr.Tabs():
                # Tab 1: Main search
                with gr.TabItem("Search"):
                    with gr.Row():
                        query_input = gr.Textbox(
                            label="Research interest",
                            placeholder="e.g. attention mechanism in visual processing",
                            scale=4,
                        )
                        search_btn = gr.Button("Recommend", variant="primary", scale=1)

                    with gr.Row():
                        model_toggle = gr.Checkbox(
                            label="Use fine-tuned model", value=True
                        )
                        lambda_slider = gr.Slider(
                            0.0, 1.0, value=0.6, step=0.05,
                            label="Relevance vs. Diversity (lambda)",
                        )
                        topn_slider = gr.Slider(
                            5, 20, value=10, step=1, label="Number of results"
                        )

                    with gr.Row():
                        results_html = gr.HTML(label="Recommendations")

                    with gr.Row():
                        diversity_chart = gr.Plot(label="Field Distribution")
                        metrics_box = gr.Textbox(
                            label="Diversity Metrics", lines=4, interactive=False
                        )

                    search_btn.click(
                        fn=self.search,
                        inputs=[query_input, model_toggle, lambda_slider, topn_slider],
                        outputs=[results_html, diversity_chart, metrics_box],
                    )

                # Tab 2: Before/After comparison
                with gr.TabItem("Before / After"):
                    with gr.Row():
                        compare_query = gr.Textbox(
                            label="Research interest",
                            placeholder="e.g. decision making under uncertainty",
                            scale=4,
                        )
                        compare_btn = gr.Button("Compare", variant="primary", scale=1)

                    with gr.Row():
                        cmp_lambda = gr.Slider(
                            0.0, 1.0, value=0.6, step=0.05,
                            label="Relevance vs. Diversity (lambda)",
                        )
                        cmp_topn = gr.Slider(
                            5, 20, value=10, step=1, label="Number of results"
                        )

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Before (Base Model)")
                            before_html = gr.HTML()
                        with gr.Column():
                            gr.Markdown("### After (Fine-tuned)")
                            after_html = gr.HTML()

                    with gr.Row():
                        comparison_chart = gr.Plot(label="Metric Comparison")
                        delta_box = gr.Textbox(
                            label="Improvement", lines=4, interactive=False
                        )

                    compare_btn.click(
                        fn=self.compare,
                        inputs=[compare_query, cmp_lambda, cmp_topn],
                        outputs=[before_html, after_html, comparison_chart, delta_box],
                    )

        return app


def main():
    """Launch the web app."""
    parser = argparse.ArgumentParser(description="CrossPaper web app")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app_instance = CrossPaperApp()
    ui = app_instance.build_ui()

    # Bind to 0.0.0.0 on Spaces for the proxy, localhost otherwise
    on_spaces = os.environ.get("SPACE_ID") is not None
    server_name = "0.0.0.0" if on_spaces else "127.0.0.1"

    ui.queue()
    ui.launch(
        server_name=server_name,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
