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
        <div style="font-size: 12px; font-weight: 700; color: #8b9bb4; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">⏱️ Phân Bổ Thời Gian Chi Tiết</div>
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin-bottom: 10px;">
            <div class="timing-card timing-card-highlight">
                <div style="font-size: 16px;">🚀</div>
                <div style="font-weight: 600; margin: 2px 0; font-size: 11px; color: #f0f6fc;">DiT Denoise</div>
                <div style="font-size: 14px; color: #ffa200; font-weight: 700;">{dit_denoising_time}</div>
            </div>
            <div class="timing-card">
                <div style="font-size: 16px;">🧠</div>
                <div style="font-weight: 600; margin: 2px 0; font-size: 11px; color: #f0f6fc;">E2E (VAE/Text)</div>
                <div style="font-size: 14px; color: #388bfd; font-weight: 700;">{inference_time:.2f}s</div>
            </div>
            <div class="timing-card">
                <div style="font-size: 16px;">🎬</div>
                <div style="font-weight: 600; margin: 2px 0; font-size: 11px; color: #f0f6fc;">Encoding</div>
                <div style="font-size: 14px; color: #e63946; font-weight: 700;">OK</div>
            </div>
            <div class="timing-card">
                <div style="font-size: 16px;">💾</div>
                <div style="font-weight: 600; margin: 2px 0; font-size: 11px; color: #f0f6fc;">VRAM Offload</div>
                <div style="font-size: 14px; color: #10b981; font-weight: 700;">Tiered</div>
            </div>
            <div class="timing-card">
                <div style="font-size: 16px;">📊</div>
                <div style="font-weight: 600; margin: 2px 0; font-size: 11px; color: #f0f6fc;">Tổng Thời Gian</div>
                <div style="font-size: 14px; color: #ffbe4d; font-weight: 700;">{total_time:.2f}s</div>
            </div>
        </div>"""

    if inference_time > 0:
        fps = num_frames / inference_time
        timing_html += f"""
        <div class="performance-card">
            <span style="font-weight: 600; color: #8b9bb4; font-size: 12px;">Tốc độ xử lý GPU: </span>
            <span style="font-size: 14px; color: #388bfd; font-weight: 700; font-family: monospace;">{fps:.2f} frames/giây</span>
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
    """Render a sleek dark-mode progress card item."""
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
    <div style="margin: 6px 0 10px 0; background: #0b0f15; border-radius: 6px; padding: 8px 12px; border: 1px solid #243142;">
        <div style="display: flex; justify-content: space-between; font-size: 12px; color: #8b9bb4; margin-bottom: 6px;">
            <span style="color: #f0f6fc; font-weight: 500;">{text}</span>
            <span style="font-weight: 700; font-family: monospace; color: {color};">{pct}%</span>
        </div>
        <div style="background: #161f2c; height: 7px; border-radius: 4px; overflow: hidden;">
            <div style="background: {bg_bar}; width: {pct}%; height: 100%; border-radius: 4px; transition: width 0.3s ease;"></div>
        </div>
    </div>
    """


STUDIO_CSS = """
/* ========================================================
   FastVideo AI Studio Pro — Ultra Dark Obsidian Theme
   ======================================================== */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Force dark theme everywhere */
:root, html, body, .gradio-container, .dark {
    --bg-main: #0a0d14;
    --bg-card: #141b26;
    --bg-input: #0b0f15;
    --border-color: #243142;
    --border-hover: #3a4b63;
    --border-focus: #ff4d5e;
    --accent-red: #e63946;
    --accent-red-hover: #ff4d5e;
    --accent-amber: #f5a623;
    --accent-amber-light: #ffbe4d;
    --accent-blue: #388bfd;
    --accent-green: #10b981;
    --text-primary: #f0f6fc;
    --text-secondary: #8b9bb4;
    --text-muted: #57657a;
    background-color: #0a0d14 !important;
    color: #f0f6fc !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    color-scheme: dark !important;
}

/* Page Layout */
.gradio-container {
    max-width: 1720px !important;
    margin: 0 auto !important;
    padding: 10px 16px !important;
    background: #0a0d14 !important;
}

/* Custom Studio Header */
.studio-hero-header {
    background: linear-gradient(180deg, #16202e 0%, #101622 100%);
    border: 1px solid #243142;
    border-radius: 10px;
    padding: 12px 18px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
}
.hero-title-group {
    display: flex;
    align-items: center;
    gap: 14px;
}
.hero-badges-group {
    display: flex;
    align-items: center;
    gap: 8px;
}
.hero-badge {
    background: #0b0f15;
    border: 1px solid #243142;
    padding: 4px 10px;
    border-radius: 16px;
    font-size: 11px;
    font-weight: 600;
    color: #8b9bb4;
    display: inline-flex;
    align-items: center;
    gap: 5px;
}
.hero-badge.badge-highlight {
    border-color: rgba(245, 166, 35, 0.4);
    color: #ffbe4d;
}
.hero-badge.badge-active {
    border-color: rgba(16, 185, 129, 0.4);
    color: #34d399;
}

/* Panels & Cards */
.studio-card, .block, .gr-box, .gr-panel, div[data-testid="block"] {
    background-color: #141b26 !important;
    border: 1px solid #243142 !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}
.studio-card {
    padding: 12px 14px !important;
    margin-bottom: 10px !important;
}

/* Card Titles */
.card-title {
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
    color: #f0f6fc !important;
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
    padding-bottom: 6px !important;
    margin-bottom: 10px !important;
    border-bottom: 1px solid #243142 !important;
}

/* Form Controls & Inputs - Zero White Backgrounds */
input, textarea, select, 
.gr-input, .gr-box input, .gr-box textarea,
.wrap, .secondary-wrap, .wrap-inner,
div[data-testid="textbox"] textarea,
div[data-testid="dropdown"] select,
.gr-dropdown {
    background-color: #0b0f15 !important;
    border: 1px solid #243142 !important;
    border-radius: 6px !important;
    color: #f0f6fc !important;
    font-size: 13px !important;
    padding: 6px 10px !important;
}
input:focus, textarea:focus, select:focus, 
.gr-input:focus, .wrap:focus-within {
    border-color: #ff4d5e !important;
    box-shadow: 0 0 0 2px rgba(230, 57, 70, 0.25) !important;
    outline: none !important;
}

/* Textarea Monospace */
textarea {
    font-family: 'JetBrains Mono', 'Consolas', monospace !important;
    line-height: 1.5 !important;
}

/* Dropdown Menu styling */
.dropdown, .options, ul.options, .dropdown-menu {
    background-color: #0b0f15 !important;
    border: 1px solid #243142 !important;
    color: #f0f6fc !important;
    border-radius: 6px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5) !important;
}
ul.options li, .options .item, .dropdown-item {
    background-color: #141b26 !important;
    color: #f0f6fc !important;
    padding: 7px 12px !important;
    border-bottom: 1px solid #1c2635 !important;
    font-size: 12px !important;
}
ul.options li:hover, .options .item:hover, ul.options li.selected {
    background-color: #243142 !important;
    color: #ffbe4d !important;
}

/* Labels */
label, span.label-text, .block-title, .form label, .block label span {
    color: #8b9bb4 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    margin-bottom: 3px !important;
    letter-spacing: 0.3px !important;
}

/* Radio Group Redesign (Horizontal Dark Pills) */
.gr-radio, fieldset, div[data-testid="radio-group"] {
    background-color: #0b0f15 !important;
    border: 1px solid #243142 !important;
    border-radius: 6px !important;
    padding: 4px 6px !important;
    margin-bottom: 6px !important;
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 6px !important;
}
.gr-radio label, fieldset label {
    background: #141b26 !important;
    border: 1px solid #243142 !important;
    border-radius: 4px !important;
    padding: 4px 10px !important;
    color: #8b9bb4 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}
.gr-radio label:hover, fieldset label:hover {
    border-color: #3a4b63 !important;
    color: #f0f6fc !important;
}
.gr-radio label.selected, fieldset label.selected,
.gr-radio input[type="radio"]:checked + span {
    background: linear-gradient(135deg, rgba(230, 57, 70, 0.25), rgba(255, 77, 94, 0.15)) !important;
    border-color: #e63946 !important;
    color: #ff4d5e !important;
    font-weight: 600 !important;
}

/* Checkbox */
input[type="checkbox"] {
    accent-color: #e63946 !important;
    width: 14px !important;
    height: 14px !important;
}

/* Accordions */
.accordion, .label-wrap {
    background-color: #141b26 !important;
    border: 1px solid #243142 !important;
    border-radius: 6px !important;
    padding: 6px 10px !important;
}
.accordion .label-wrap span {
    color: #f0f6fc !important;
    font-weight: 600 !important;
    font-size: 12px !important;
}

/* Primary Generate Button */
button.btn-generate-main, button.primary {
    background: linear-gradient(135deg, #e63946 0%, #ff4d5e 100%) !important;
    border: 1px solid #ff4d5e !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    letter-spacing: 0.6px !important;
    padding: 10px 18px !important;
    width: 100% !important;
    box-shadow: 0 4px 14px rgba(230, 57, 70, 0.4) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    text-transform: uppercase !important;
    margin-top: 4px !important;
}
button.btn-generate-main:hover, button.primary:hover {
    background: linear-gradient(135deg, #ff4d5e 0%, #ff6b7a 100%) !important;
    box-shadow: 0 6px 20px rgba(230, 57, 70, 0.55) !important;
    transform: translateY(-1px) !important;
}
button.secondary {
    background: #1c2635 !important;
    border: 1px solid #243142 !important;
    color: #f0f6fc !important;
    border-radius: 6px !important;
    padding: 6px 12px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
}
button.secondary:hover {
    background: #243142 !important;
    border-color: #f5a623 !important;
    color: #ffbe4d !important;
}

/* Status Banner */
.banner-info {
    background: #0b0f15;
    border: 1px solid #243142;
    border-left: 3px solid #f5a623;
    border-radius: 6px;
    padding: 8px 12px;
    color: #8b9bb4;
    font-size: 12px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.banner-info strong {
    color: #f0f6fc;
}

/* Status Badges */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.3px;
}
.badge-idle {
    background: rgba(139, 155, 180, 0.15);
    color: #8b9bb4;
    border: 1px solid #243142;
}
.badge-running {
    background: rgba(245, 166, 35, 0.2);
    color: #ffbe4d;
    border: 1px solid #f5a623;
}
.badge-success {
    background: rgba(16, 185, 129, 0.2);
    color: #34d399;
    border: 1px solid #10b981;
}
.badge-error {
    background: rgba(239, 68, 68, 0.2);
    color: #f87171;
    border: 1px solid #ef4444;
}

/* Video Card in Queue */
.video-row-card {
    background-color: #141b26 !important;
    border: 1px solid #243142 !important;
    border-radius: 8px !important;
    padding: 12px !important;
    margin-bottom: 12px !important;
}
.card-header-row {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    margin-bottom: 6px !important;
}
.card-header-row h3 {
    margin: 0 !important;
    font-size: 14px !important;
    color: #f0f6fc !important;
    font-weight: 600 !important;
}
.video-row-card video {
    border-radius: 6px !important;
    background: #000000 !important;
    border: 1px solid #243142 !important;
}

/* Timing & Performance */
.timing-card {
    background: #0b0f15;
    border: 1px solid #243142;
    border-radius: 6px;
    padding: 8px 6px;
    text-align: center;
}
.timing-card-highlight {
    border-color: rgba(245, 166, 35, 0.5);
    background: linear-gradient(135deg, #141b26 0%, #0b0f15 100%);
}
.performance-card {
    background: #0b0f15;
    border: 1px solid #243142;
    border-radius: 6px;
    padding: 6px 10px;
    text-align: center;
}

/* Sliders */
.slider-wrap input[type="range"] {
    accent-color: #e63946 !important;
}

/* Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #0a0d14;
}
::-webkit-scrollbar-thumb {
    background: #243142;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #3a4b63;
}
"""
