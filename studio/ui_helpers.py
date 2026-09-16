"""
UI Helpers and Visual Components for FastVideo AI Studio Pro.
Provides timing metrics, dynamic progress bar rendering, and safe progress hooks.
"""

from typing import List, Optional


def create_timing_display(
    inference_time: float,
    total_time: float,
    stage_execution_times: List[float],
    num_frames: int,
) -> str:
    """Generate rich HTML summary of inference timing and hardware throughput."""
    dit_denoising_time = f"{stage_execution_times[5]:.2f}s" if len(stage_execution_times) > 5 else "N/A"

    timing_html = f"""
    <div style="margin: 10px 0;">
        <h3 style="text-align: center; margin-bottom: 10px; color: #f0f6fc;">⏱️ Timing Breakdown</h3>
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-bottom: 10px;">
            <div class="timing-card timing-card-highlight">
                <div style="font-size: 20px;">🚀</div>
                <div style="font-weight: bold; margin: 3px 0; font-size: 14px; color: #f0f6fc;">DiT Denoising</div>
                <div style="font-size: 16px; color: #ffa200; font-weight: bold;">{dit_denoising_time}</div>
            </div>
            <div class="timing-card">
                <div style="font-size: 20px;">🧠</div>
                <div style="font-weight: bold; margin: 3px 0; font-size: 14px; color: #f0f6fc;">E2E (w. VAE/Text)</div>
                <div style="font-size: 16px; color: #2563eb;">{inference_time:.2f}s</div>
            </div>
            <div class="timing-card">
                <div style="font-size: 20px;">🎬</div>
                <div style="font-weight: bold; margin: 3px 0; font-size: 14px; color: #f0f6fc;">Video Encoding</div>
                <div style="font-size: 16px; color: #dc2626;">N/A</div>
            </div>
            <div class="timing-card">
                <div style="font-size: 20px;">🌐</div>
                <div style="font-weight: bold; margin: 3px 0; font-size: 14px; color: #f0f6fc;">Network Transfer</div>
                <div style="font-size: 16px; color: #059669;">N/A</div>
            </div>
            <div class="timing-card">
                <div style="font-size: 20px;">📊</div>
                <div style="font-weight: bold; margin: 3px 0; font-size: 14px; color: #f0f6fc;">Total Processing</div>
                <div style="font-size: 18px; color: #0277bd;">{total_time:.2f}s</div>
            </div>
        </div>"""

    if inference_time > 0:
        fps = num_frames / inference_time
        timing_html += f"""
        <div class="performance-card" style="margin-top: 15px;">
            <span style="font-weight: bold; color: #f0f6fc;">Generation Speed: </span>
            <span style="font-size: 18px; color: #6366f1; font-weight: bold;">{fps:.1f} frames/second</span>
        </div>"""

    return timing_html + "</div>"


def safe_progress(prog, frac: float, desc: str = "") -> None:
    """Safely update Gradio progress bar without raising exceptions."""
    if prog is not None:
        try:
            prog(frac, desc=desc)
        except Exception:
            pass


def make_progress_bar_html(pct: int, text: str, status: str = "running") -> str:
    """
    Render a sleek dark-mode progress card item.
    Status can be 'running', 'success', 'error', or 'idle'.
    """
    if status == "running":
        color = "#f5a623"
        bg_bar = "linear-gradient(90deg, #f5a623, #ffbe4d)"
    elif status == "success":
        color = "#10b981"
        bg_bar = "linear-gradient(90deg, #10b981, #34d399)"
    elif status == "error":
        color = "#ef4444"
        bg_bar = "#ef4444"
    else:
        color = "#8b9bb4"
        bg_bar = "#243142"

    return f"""
    <div style="margin: 6px 0 10px 0; background: #0b0f15; border-radius: 6px; padding: 6px 10px; border: 1px solid #243142;">
        <div style="display: flex; justify-content: space-between; font-size: 12px; color: #8b9bb4; margin-bottom: 5px;">
            <span>{text}</span>
            <span style="font-weight: bold; color: {color};">{pct}%</span>
        </div>
        <div style="background: #161f2c; height: 8px; border-radius: 4px; overflow: hidden;">
            <div style="background: {bg_bar}; width: {pct}%; height: 100%; border-radius: 4px; transition: width 0.3s ease;"></div>
        </div>
    </div>
    """


STUDIO_CSS = """
/* FastVideo AI Studio Pro Theme */
body {
    background-color: #0f141c !important;
    color: #f0f6fc !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.gradio-container {
    max-width: 1680px !important;
    margin: auto !important;
}
.studio-card {
    background-color: #161f2c !important;
    border: 1px solid #243142 !important;
    border-radius: 10px !important;
    padding: 16px !important;
    margin-bottom: 14px !important;
}
.timing-card {
    background: #161f2c;
    border: 1px solid #243142;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
}
.timing-card-highlight {
    background: linear-gradient(135deg, #1e293b 0%, #161f2c 100%);
    border: 1px solid #ffa200;
}
.performance-card {
    background: #161f2c;
    border: 1px solid #6366f1;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
    margin: 10px 0;
}
button.primary {
    background: linear-gradient(135deg, #e63946 0%, #ff4d5e 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(230, 57, 70, 0.4) !important;
}
button.primary:hover {
    background: linear-gradient(135deg, #ff4d5e 0%, #ff6b7a 100%) !important;
}
"""
