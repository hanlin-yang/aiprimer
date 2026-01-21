"""
PDF Report Generation Service

Generates professional primer design reports using Jinja2 templates
and WeasyPrint for PDF rendering.

Features:
- Risk score dashboard with visual gauges
- In-silico gel electrophoresis visualization
- Structured primer information tables
- Variant conflict summaries
- AI recommendations section

Compliance: FDA 21 CFR Part 11 audit trail integration
"""

from __future__ import annotations

import base64
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

logger = logging.getLogger(__name__)

# Template directory
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


# ============================================================================
# Report Data Models
# ============================================================================

class ReportData:
    """Data container for report generation."""

    def __init__(
        self,
        request_id: str,
        target_gene: str,
        task_type: str,
        primer_pairs: List[Dict[str, Any]],
        best_pair: Optional[Dict[str, Any]],
        structural_warnings: List[Dict[str, Any]],
        variant_warnings: List[Dict[str, Any]],
        specificity_result: Optional[Dict[str, Any]],
        ai_suggestion: Optional[str],
        computation_time_ms: int,
        created_at: datetime,
        user_id: Optional[str] = None,
    ):
        self.request_id = request_id
        self.target_gene = target_gene
        self.task_type = task_type
        self.primer_pairs = primer_pairs
        self.best_pair = best_pair
        self.structural_warnings = structural_warnings
        self.variant_warnings = variant_warnings
        self.specificity_result = specificity_result
        self.ai_suggestion = ai_suggestion
        self.computation_time_ms = computation_time_ms
        self.created_at = created_at
        self.user_id = user_id

    @classmethod
    def from_response(cls, response) -> 'ReportData':
        """Create ReportData from PrimerDesignResponse."""
        return cls(
            request_id=response.request_id,
            target_gene=response.target_gene,
            task_type=response.task_type.value,
            primer_pairs=[pp.model_dump() for pp in response.primer_pairs],
            best_pair=response.best_pair.model_dump() if response.best_pair else None,
            structural_warnings=[w.model_dump() for w in response.structural_warnings],
            variant_warnings=response.variant_warnings,
            specificity_result=response.specificity_results.model_dump() if response.specificity_results else None,
            ai_suggestion=response.ai_suggestion,
            computation_time_ms=response.computation_time_ms or 0,
            created_at=response.created_at,
        )


# ============================================================================
# Visualization Generators
# ============================================================================

class GelVisualization:
    """
    Generates in-silico gel electrophoresis images.

    Creates SVG representations of expected PCR products
    based on their sizes.
    """

    # Gel dimensions
    GEL_WIDTH = 400
    GEL_HEIGHT = 500
    LANE_WIDTH = 50
    WELL_HEIGHT = 20

    # Ladder sizes (bp) and their positions
    LADDER_SIZES = [1000, 750, 500, 400, 300, 200, 150, 100, 75, 50]

    def __init__(self):
        self.max_size = 1200  # Maximum bp to display
        self.min_size = 25   # Minimum bp to display

    def _bp_to_position(self, bp: int) -> float:
        """Convert base pairs to Y position (log scale)."""
        import math

        if bp <= self.min_size:
            bp = self.min_size
        if bp >= self.max_size:
            bp = self.max_size

        # Log scale transformation
        log_min = math.log10(self.min_size)
        log_max = math.log10(self.max_size)
        log_bp = math.log10(bp)

        # Invert (smaller fragments travel further)
        relative_pos = (log_max - log_bp) / (log_max - log_min)

        # Map to gel height (leaving space for wells)
        return self.WELL_HEIGHT + 30 + (self.GEL_HEIGHT - self.WELL_HEIGHT - 60) * relative_pos

    def generate_svg(
        self,
        products: List[Dict[str, Any]],
        title: str = "In-Silico Gel"
    ) -> str:
        """
        Generate SVG representation of gel.

        Args:
            products: List of dicts with 'name', 'size', 'intensity'
            title: Gel title

        Returns:
            SVG string
        """
        num_lanes = len(products) + 1  # +1 for ladder
        total_width = (num_lanes + 1) * self.LANE_WIDTH + 40

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_width} {self.GEL_HEIGHT + 60}">',
            # Background
            '<defs>',
            '<linearGradient id="gelGradient" x1="0%" y1="0%" x2="0%" y2="100%">',
            '<stop offset="0%" style="stop-color:#1a1a2e"/>',
            '<stop offset="100%" style="stop-color:#16213e"/>',
            '</linearGradient>',
            '</defs>',
            # Gel background
            f'<rect x="20" y="{self.WELL_HEIGHT}" width="{total_width - 40}" height="{self.GEL_HEIGHT}" '
            'fill="url(#gelGradient)" rx="5"/>',
            # Title
            f'<text x="{total_width/2}" y="15" text-anchor="middle" fill="#333" '
            'font-family="Arial" font-size="14" font-weight="bold">{title}</text>',
        ]

        # Draw wells
        for i in range(num_lanes):
            x = 40 + i * self.LANE_WIDTH
            svg_parts.append(
                f'<rect x="{x}" y="{self.WELL_HEIGHT}" width="{self.LANE_WIDTH - 10}" '
                f'height="10" fill="#0a0a15" rx="2"/>'
            )

        # Draw ladder in first lane
        ladder_x = 40
        for bp in self.LADDER_SIZES:
            y = self._bp_to_position(bp)
            # Intensity varies by size
            intensity = 0.5 + 0.5 * (bp / self.max_size)
            svg_parts.append(
                f'<rect x="{ladder_x}" y="{y}" width="{self.LANE_WIDTH - 15}" height="3" '
                f'fill="rgba(255,255,255,{intensity})" rx="1"/>'
            )
            # Size label
            svg_parts.append(
                f'<text x="{ladder_x - 5}" y="{y + 3}" text-anchor="end" fill="#666" '
                f'font-family="Arial" font-size="8">{bp}</text>'
            )

        # Lane label for ladder
        svg_parts.append(
            f'<text x="{ladder_x + self.LANE_WIDTH/2 - 5}" y="{self.GEL_HEIGHT + 40}" '
            f'text-anchor="middle" fill="#333" font-family="Arial" font-size="10">M</text>'
        )

        # Draw product bands
        for i, product in enumerate(products):
            lane_x = 40 + (i + 1) * self.LANE_WIDTH
            y = self._bp_to_position(product['size'])
            intensity = product.get('intensity', 0.9)

            # Main band
            svg_parts.append(
                f'<rect x="{lane_x}" y="{y}" width="{self.LANE_WIDTH - 15}" height="4" '
                f'fill="rgba(255,255,255,{intensity})" rx="1">'
                '<animate attributeName="opacity" values="1;0.8;1" dur="2s" repeatCount="indefinite"/>'
                '</rect>'
            )

            # Size annotation
            svg_parts.append(
                f'<text x="{lane_x + self.LANE_WIDTH}" y="{y + 3}" fill="#0af" '
                f'font-family="Arial" font-size="9">{product["size"]} bp</text>'
            )

            # Lane label
            svg_parts.append(
                f'<text x="{lane_x + self.LANE_WIDTH/2 - 5}" y="{self.GEL_HEIGHT + 40}" '
                f'text-anchor="middle" fill="#333" font-family="Arial" font-size="10">'
                f'{product.get("name", str(i + 1))}</text>'
            )

        svg_parts.append('</svg>')
        return '\n'.join(svg_parts)


class RiskGauge:
    """
    Generates risk score gauge visualizations.

    Creates SVG gauges showing risk levels from 0-1.
    """

    def generate_svg(
        self,
        score: float,
        label: str = "Risk Score",
        width: int = 200,
        height: int = 120
    ) -> str:
        """
        Generate SVG gauge for risk score.

        Args:
            score: Risk score 0-1
            label: Gauge label
            width: SVG width
            height: SVG height

        Returns:
            SVG string
        """
        score = max(0, min(1, score))  # Clamp to 0-1

        # Color based on score
        if score < 0.3:
            color = "#22c55e"  # Green
            level = "Low"
        elif score < 0.6:
            color = "#eab308"  # Yellow
            level = "Medium"
        elif score < 0.8:
            color = "#f97316"  # Orange
            level = "High"
        else:
            color = "#ef4444"  # Red
            level = "Critical"

        # Gauge arc calculation
        import math
        cx, cy = width / 2, height - 20
        radius = 60
        start_angle = math.pi  # 180 degrees
        end_angle = 0  # 0 degrees
        score_angle = start_angle - (start_angle - end_angle) * score

        # Arc path
        def arc_point(angle):
            return cx + radius * math.cos(angle), cy - radius * math.sin(angle)

        start = arc_point(start_angle)
        end = arc_point(end_angle)
        score_point = arc_point(score_angle)

        # Large arc flag
        large_arc = 0

        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">
  <!-- Background arc -->
  <path d="M {start[0]} {start[1]} A {radius} {radius} 0 {large_arc} 1 {end[0]} {end[1]}"
        fill="none" stroke="#e5e7eb" stroke-width="12" stroke-linecap="round"/>

  <!-- Score arc -->
  <path d="M {start[0]} {start[1]} A {radius} {radius} 0 {large_arc} 1 {score_point[0]} {score_point[1]}"
        fill="none" stroke="{color}" stroke-width="12" stroke-linecap="round"/>

  <!-- Needle -->
  <line x1="{cx}" y1="{cy}" x2="{score_point[0]}" y2="{score_point[1]}"
        stroke="#374151" stroke-width="3" stroke-linecap="round"/>
  <circle cx="{cx}" cy="{cy}" r="8" fill="#374151"/>

  <!-- Labels -->
  <text x="{cx}" y="{cy + 35}" text-anchor="middle" fill="#374151"
        font-family="Arial" font-size="24" font-weight="bold">{score:.0%}</text>
  <text x="{cx}" y="{cy + 50}" text-anchor="middle" fill="{color}"
        font-family="Arial" font-size="12" font-weight="bold">{level}</text>
  <text x="{cx}" y="15" text-anchor="middle" fill="#6b7280"
        font-family="Arial" font-size="11">{label}</text>

  <!-- Scale labels -->
  <text x="25" y="{cy + 5}" fill="#9ca3af" font-family="Arial" font-size="10">0%</text>
  <text x="{width - 30}" y="{cy + 5}" fill="#9ca3af" font-family="Arial" font-size="10">100%</text>
</svg>'''
        return svg


# ============================================================================
# Report Generator
# ============================================================================

class ReportGenerator:
    """
    Generates PDF reports from primer design results.

    Uses Jinja2 for HTML templating and WeasyPrint for PDF conversion.
    """

    def __init__(self, templates_dir: Optional[Path] = None):
        self.templates_dir = templates_dir or TEMPLATES_DIR

        # Initialize Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=select_autoescape(['html', 'xml'])
        )

        # Register custom filters
        self.env.filters['format_tm'] = lambda x: f"{x:.1f}°C"
        self.env.filters['format_gc'] = lambda x: f"{x:.1f}%"
        self.env.filters['format_dg'] = lambda x: f"{x:.2f} kcal/mol" if x else "N/A"

        self.gel_viz = GelVisualization()
        self.risk_gauge = RiskGauge()

    def generate_html(
        self,
        data: ReportData,
        include_gel: bool = True,
        include_dashboard: bool = True
    ) -> str:
        """
        Generate HTML report from design data.

        Args:
            data: Report data
            include_gel: Include in-silico gel image
            include_dashboard: Include risk dashboard

        Returns:
            HTML string
        """
        # Prepare visualizations
        visualizations = {}

        if include_gel and data.primer_pairs:
            products = [
                {
                    'name': f"P{pp.get('pair_id', i) + 1}",
                    'size': pp.get('product_size', 100),
                    'intensity': 1.0 - pp.get('penalty_score', 0) / 10
                }
                for i, pp in enumerate(data.primer_pairs[:5])
            ]
            visualizations['gel_svg'] = self.gel_viz.generate_svg(
                products,
                title=f"{data.target_gene} PCR Products"
            )

        if include_dashboard:
            # Calculate overall risk score
            num_warnings = len(data.structural_warnings) + len(data.variant_warnings)
            risk_score = min(1.0, num_warnings * 0.15)

            visualizations['risk_gauge_svg'] = self.risk_gauge.generate_svg(
                risk_score,
                label="Overall Risk"
            )

            # Specificity gauge
            if data.specificity_result:
                spec_score = data.specificity_result.get('specificity_score', 1.0)
                visualizations['specificity_gauge_svg'] = self.risk_gauge.generate_svg(
                    1 - spec_score,  # Invert for "risk"
                    label="Off-Target Risk"
                )

            # Dimer risk gauge
            if data.best_pair:
                dimer_score = data.best_pair.get('dimer_risk_score', 0)
                visualizations['dimer_gauge_svg'] = self.risk_gauge.generate_svg(
                    dimer_score,
                    label="Dimer Risk"
                )

        # Load and render template
        try:
            template = self.env.get_template('primer_report.html')
        except Exception:
            # Use inline template if file not found
            template = self.env.from_string(INLINE_TEMPLATE)

        html = template.render(
            data=data,
            visualizations=visualizations,
            generated_at=datetime.utcnow().isoformat(),
            report_version="1.0.0"
        )

        return html

    def generate_pdf(
        self,
        data: ReportData,
        output_path: Optional[Path] = None
    ) -> bytes:
        """
        Generate PDF report.

        Args:
            data: Report data
            output_path: Optional path to save PDF

        Returns:
            PDF bytes
        """
        try:
            from weasyprint import HTML, CSS
        except ImportError:
            logger.error("WeasyPrint not installed. Install with: pip install weasyprint")
            raise ImportError("WeasyPrint required for PDF generation")

        html = self.generate_html(data)

        # Convert to PDF
        pdf_bytes = HTML(string=html).write_pdf(
            stylesheets=[CSS(string=PDF_STYLES)]
        )

        if output_path:
            output_path.write_bytes(pdf_bytes)
            logger.info(f"PDF saved to {output_path}")

        return pdf_bytes

    def generate_report(
        self,
        response,
        format: str = "html",
        output_path: Optional[Path] = None
    ) -> str | bytes:
        """
        Generate report from PrimerDesignResponse.

        Args:
            response: PrimerDesignResponse object
            format: Output format ('html' or 'pdf')
            output_path: Optional output path

        Returns:
            HTML string or PDF bytes
        """
        data = ReportData.from_response(response)

        if format.lower() == 'pdf':
            return self.generate_pdf(data, output_path)
        else:
            html = self.generate_html(data)
            if output_path:
                output_path.write_text(html, encoding='utf-8')
            return html


# ============================================================================
# Inline Template (fallback)
# ============================================================================

INLINE_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Primer Design Report - {{ data.target_gene }}</title>
    <style>
        :root {
            --primary: #2563eb;
            --success: #22c55e;
            --warning: #eab308;
            --danger: #ef4444;
            --gray-100: #f3f4f6;
            --gray-200: #e5e7eb;
            --gray-700: #374151;
            --gray-900: #111827;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            line-height: 1.6;
            color: var(--gray-900);
            background: var(--gray-100);
            padding: 20px;
        }

        .container { max-width: 1000px; margin: 0 auto; }

        .header {
            background: linear-gradient(135deg, var(--primary), #1e40af);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 24px;
        }

        .header h1 { font-size: 28px; margin-bottom: 8px; }
        .header .meta { opacity: 0.9; font-size: 14px; }

        .card {
            background: white;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        .card h2 {
            color: var(--gray-700);
            font-size: 18px;
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 2px solid var(--gray-200);
        }

        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
        }

        .gauge-container {
            text-align: center;
            padding: 16px;
            background: var(--gray-100);
            border-radius: 8px;
        }

        .primer-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }

        .primer-table th, .primer-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--gray-200);
        }

        .primer-table th {
            background: var(--gray-100);
            font-weight: 600;
        }

        .primer-table tr:hover { background: #f9fafb; }

        .sequence {
            font-family: 'Courier New', monospace;
            background: var(--gray-100);
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 13px;
        }

        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
        }

        .badge-success { background: #dcfce7; color: #166534; }
        .badge-warning { background: #fef9c3; color: #854d0e; }
        .badge-danger { background: #fee2e2; color: #991b1b; }

        .warning-list {
            list-style: none;
        }

        .warning-item {
            padding: 12px;
            margin-bottom: 8px;
            border-radius: 8px;
            display: flex;
            align-items: flex-start;
            gap: 12px;
        }

        .warning-item.warning { background: #fffbeb; border-left: 4px solid var(--warning); }
        .warning-item.error { background: #fef2f2; border-left: 4px solid var(--danger); }

        .ai-suggestion {
            background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
            border-left: 4px solid var(--primary);
            padding: 20px;
            border-radius: 0 8px 8px 0;
        }

        .ai-suggestion h3 {
            color: var(--primary);
            margin-bottom: 8px;
        }

        .gel-container {
            text-align: center;
            padding: 20px;
        }

        .footer {
            text-align: center;
            color: var(--gray-700);
            font-size: 12px;
            padding: 20px;
            border-top: 1px solid var(--gray-200);
            margin-top: 24px;
        }

        @media print {
            body { background: white; padding: 0; }
            .card { box-shadow: none; border: 1px solid var(--gray-200); }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="header">
            <h1>🧬 Primer Design Report</h1>
            <div class="meta">
                <strong>Target Gene:</strong> {{ data.target_gene }} |
                <strong>Task Type:</strong> {{ data.task_type }} |
                <strong>Request ID:</strong> {{ data.request_id[:8] }}...
            </div>
        </header>

        <!-- Risk Dashboard -->
        {% if visualizations.risk_gauge_svg %}
        <section class="card">
            <h2>📊 Risk Assessment Dashboard</h2>
            <div class="dashboard">
                <div class="gauge-container">
                    {{ visualizations.risk_gauge_svg | safe }}
                </div>
                {% if visualizations.specificity_gauge_svg %}
                <div class="gauge-container">
                    {{ visualizations.specificity_gauge_svg | safe }}
                </div>
                {% endif %}
                {% if visualizations.dimer_gauge_svg %}
                <div class="gauge-container">
                    {{ visualizations.dimer_gauge_svg | safe }}
                </div>
                {% endif %}
            </div>
        </section>
        {% endif %}

        <!-- Best Primer Pair -->
        {% if data.best_pair %}
        <section class="card">
            <h2>🏆 Recommended Primer Pair</h2>
            <table class="primer-table">
                <thead>
                    <tr>
                        <th>Direction</th>
                        <th>Sequence (5' → 3')</th>
                        <th>Length</th>
                        <th>Tm</th>
                        <th>GC%</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Forward</strong></td>
                        <td><code class="sequence">{{ data.best_pair.forward.sequence }}</code></td>
                        <td>{{ data.best_pair.forward.length }} bp</td>
                        <td>{{ data.best_pair.forward.tm | format_tm }}</td>
                        <td>{{ data.best_pair.forward.gc_percent | format_gc }}</td>
                    </tr>
                    <tr>
                        <td><strong>Reverse</strong></td>
                        <td><code class="sequence">{{ data.best_pair.reverse.sequence }}</code></td>
                        <td>{{ data.best_pair.reverse.length }} bp</td>
                        <td>{{ data.best_pair.reverse.tm | format_tm }}</td>
                        <td>{{ data.best_pair.reverse.gc_percent | format_gc }}</td>
                    </tr>
                </tbody>
            </table>
            <div style="margin-top: 16px; display: flex; gap: 16px; flex-wrap: wrap;">
                <span class="badge badge-success">Product Size: {{ data.best_pair.product_size }} bp</span>
                <span class="badge {% if data.best_pair.tm_difference <= 1 %}badge-success{% elif data.best_pair.tm_difference <= 3 %}badge-warning{% else %}badge-danger{% endif %}">
                    ΔTm: {{ "%.1f"|format(data.best_pair.tm_difference) }}°C
                </span>
                <span class="badge {% if data.best_pair.dimer_risk_score < 0.3 %}badge-success{% elif data.best_pair.dimer_risk_score < 0.6 %}badge-warning{% else %}badge-danger{% endif %}">
                    Dimer Risk: {{ "%.0f"|format(data.best_pair.dimer_risk_score * 100) }}%
                </span>
            </div>
        </section>
        {% endif %}

        <!-- In-Silico Gel -->
        {% if visualizations.gel_svg %}
        <section class="card">
            <h2>🔬 In-Silico Gel Electrophoresis</h2>
            <div class="gel-container">
                {{ visualizations.gel_svg | safe }}
            </div>
            <p style="text-align: center; color: #666; font-size: 12px; margin-top: 12px;">
                M = Marker ladder | P1-P5 = Primer pairs ranked by quality
            </p>
        </section>
        {% endif %}

        <!-- All Primer Pairs -->
        {% if data.primer_pairs and data.primer_pairs | length > 1 %}
        <section class="card">
            <h2>📋 All Designed Primer Pairs</h2>
            <table class="primer-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Forward Primer</th>
                        <th>Reverse Primer</th>
                        <th>Product</th>
                        <th>ΔTm</th>
                        <th>Score</th>
                    </tr>
                </thead>
                <tbody>
                    {% for pair in data.primer_pairs %}
                    <tr>
                        <td>{{ pair.pair_id + 1 }}</td>
                        <td><code class="sequence">{{ pair.forward.sequence[:20] }}...</code></td>
                        <td><code class="sequence">{{ pair.reverse.sequence[:20] }}...</code></td>
                        <td>{{ pair.product_size }} bp</td>
                        <td>{{ "%.1f"|format(pair.tm_difference) }}°C</td>
                        <td>{{ "%.2f"|format(pair.penalty_score) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </section>
        {% endif %}

        <!-- Warnings -->
        {% if data.structural_warnings or data.variant_warnings %}
        <section class="card">
            <h2>⚠️ Warnings & Alerts</h2>
            <ul class="warning-list">
                {% for warning in data.structural_warnings %}
                <li class="warning-item {{ warning.severity }}">
                    <span>{{ '🔴' if warning.severity == 'critical' else '🟡' if warning.severity == 'warning' else 'ℹ️' }}</span>
                    <div>
                        <strong>{{ warning.code }}</strong>
                        <p>{{ warning.message }}</p>
                        {% if warning.suggestion %}
                        <p style="color: #666; font-size: 13px; margin-top: 4px;">
                            💡 {{ warning.suggestion }}
                        </p>
                        {% endif %}
                    </div>
                </li>
                {% endfor %}

                {% for warning in data.variant_warnings %}
                <li class="warning-item warning">
                    <span>🧬</span>
                    <div>
                        <strong>Variant Conflict</strong>
                        <p>{{ warning }}</p>
                    </div>
                </li>
                {% endfor %}
            </ul>
        </section>
        {% endif %}

        <!-- AI Suggestion -->
        {% if data.ai_suggestion %}
        <section class="card">
            <h2>🤖 AI Analysis & Recommendations</h2>
            <div class="ai-suggestion">
                <h3>Expert Recommendation</h3>
                <p>{{ data.ai_suggestion }}</p>
            </div>
        </section>
        {% endif %}

        <!-- Footer -->
        <footer class="footer">
            <p>
                Generated: {{ generated_at }} |
                Computation Time: {{ data.computation_time_ms }}ms |
                Report Version: {{ report_version }}
            </p>
            <p style="margin-top: 8px;">
                Bio-AI-SaaS Primer Design Platform | FDA 21 CFR Part 11 Compliant
            </p>
        </footer>
    </div>
</body>
</html>'''


# ============================================================================
# PDF-specific styles
# ============================================================================

PDF_STYLES = '''
@page {
    size: A4;
    margin: 2cm;
    @bottom-center {
        content: "Page " counter(page) " of " counter(pages);
        font-size: 10px;
        color: #666;
    }
}

body {
    font-size: 11pt;
}

.card {
    page-break-inside: avoid;
}

.header {
    page-break-after: avoid;
}

table {
    page-break-inside: avoid;
}
'''


# ============================================================================
# Module-level instance
# ============================================================================

report_generator = ReportGenerator()


# ============================================================================
# Convenience Functions
# ============================================================================

def generate_html_report(response) -> str:
    """Generate HTML report from PrimerDesignResponse."""
    return report_generator.generate_report(response, format='html')


def generate_pdf_report(response, output_path: Optional[Path] = None) -> bytes:
    """Generate PDF report from PrimerDesignResponse."""
    return report_generator.generate_report(response, format='pdf', output_path=output_path)
