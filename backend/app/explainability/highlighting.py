"""
Text highlighting utilities for PhishNetra
Provides visual highlighting of suspicious content
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from ..core.logging import LoggerMixin


class TextHighlighter(LoggerMixin):
    """
    Advanced text highlighting for scam detection
    Provides visual markers for suspicious content
    """

    def __init__(self):
        self.logger.info("Initializing Text Highlighter")

        # Highlight colors/styles for different risk levels
        self.highlight_styles = {
            "high": {
                "color": "#dc3545",  # Red
                "background": "#ffe6e6",
                "border": "2px solid #dc3545"
            },
            "medium": {
                "color": "#fd7e14",  # Orange
                "background": "#fff3cd",
                "border": "2px solid #fd7e14"
            },
            "low": {
                "color": "#ffc107",  # Yellow
                "background": "#fff8e1",
                "border": "1px solid #ffc107"
            }
        }

    def highlight_text(self, text: str, highlights: List[Dict[str, Any]],
                      format_type: str = "html") -> str:
        """
        Highlight suspicious phrases in text

        Args:
            text: Original text
            highlights: List of phrases to highlight with metadata
            format_type: Output format ("html", "markdown", "json")

        Returns:
            Highlighted text in specified format
        """

        if not highlights:
            return text if format_type != "json" else {"original_text": text, "highlights": []}

        # Sort highlights by position to avoid conflicts
        positioned_highlights = self._position_highlights(text, highlights)

        if format_type == "html":
            return self._highlight_html(text, positioned_highlights)
        elif format_type == "markdown":
            return self._highlight_markdown(text, positioned_highlights)
        elif format_type == "json":
            return self._highlight_json(text, positioned_highlights)
        else:
            return text

    def _position_highlights(self, text: str, highlights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Position highlights in text to avoid overlaps"""

        text_lower = text.lower()
        positioned = []

        for highlight in highlights:
            phrase = highlight["phrase"].lower()
            severity = highlight.get("severity", "medium")

            # Find all occurrences
            start = 0
            while True:
                pos = text_lower.find(phrase, start)
                if pos == -1:
                    break

                positioned.append({
                    "start": pos,
                    "end": pos + len(phrase),
                    "phrase": text[pos:pos + len(phrase)],  # Get original case
                    "severity": severity,
                    "category": highlight.get("category", "unknown"),
                    "explanation": highlight.get("explanation", "")
                })

                start = pos + 1

        # Remove overlapping highlights (keep higher severity)
        positioned = self._resolve_overlaps(positioned)

        return positioned

    def _resolve_overlaps(self, highlights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resolve overlapping highlights by keeping higher severity"""

        if not highlights:
            return highlights

        # Sort by start position
        highlights.sort(key=lambda x: x["start"])

        resolved = []
        current = highlights[0]

        severity_order = {"high": 0, "medium": 1, "low": 2}

        for highlight in highlights[1:]:
            # Check for overlap
            if highlight["start"] < current["end"]:
                # Overlap - keep the higher severity one
                current_severity = severity_order.get(current["severity"], 2)
                highlight_severity = severity_order.get(highlight["severity"], 2)

                if highlight_severity < current_severity:
                    current = highlight  # Replace with higher severity
            else:
                # No overlap - add current and start new
                resolved.append(current)
                current = highlight

        resolved.append(current)
        return resolved

    def _highlight_html(self, text: str, highlights: List[Dict[str, Any]]) -> str:
        """Generate HTML with highlighted text"""

        if not highlights:
            return f'<div class="text-content">{text}</div>'

        # Sort highlights by position
        highlights.sort(key=lambda x: x["start"])

        result = []
        last_end = 0

        for highlight in highlights:
            start, end = highlight["start"], highlight["end"]

            # Add text before highlight
            if start > last_end:
                result.append(text[last_end:start])

            # Add highlighted text
            highlighted_text = text[start:end]
            style = self.highlight_styles.get(highlight["severity"], self.highlight_styles["medium"])

            tooltip = f'title="{highlight.get("explanation", "")}"' if highlight.get("explanation") else ""

            highlight_html = f'''
                <span class="highlight highlight-{highlight["severity"]}"
                      style="color: {style['color']};
                             background-color: {style['background']};
                             border: {style['border']};
                             padding: 2px 4px;
                             border-radius: 3px;
                             margin: 0 1px;"
                      {tooltip}>
                    {highlighted_text}
                </span>
            '''

            result.append(highlight_html)
            last_end = end

        # Add remaining text
        if last_end < len(text):
            result.append(text[last_end:])

        # Wrap in container
        full_html = f'''
            <div class="highlighted-text">
                {"".join(result)}
            </div>
        '''

        return full_html

    def _highlight_markdown(self, text: str, highlights: List[Dict[str, Any]]) -> str:
        """Generate Markdown with highlighted text"""

        if not highlights:
            return text

        # Sort highlights by position
        highlights.sort(key=lambda x: x["start"])

        result = []
        last_end = 0

        for highlight in highlights:
            start, end = highlight["start"], highlight["end"]

            # Add text before highlight
            if start > last_end:
                result.append(text[last_end:start])

            # Add highlighted text with markdown formatting
            highlighted_text = text[start:end]
            severity = highlight["severity"]

            if severity == "high":
                markdown_highlight = f"**🔴 {highlighted_text}**"
            elif severity == "medium":
                markdown_highlight = f"*🟠 {highlighted_text}*"
            else:
                markdown_highlight = f"🟡 {highlighted_text}"

            result.append(markdown_highlight)
            last_end = end

        # Add remaining text
        if last_end < len(text):
            result.append(text[last_end:])

        return "".join(result)

    def _highlight_json(self, text: str, highlights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate JSON with highlight positions"""

        return {
            "original_text": text,
            "highlights": [
                {
                    "start": h["start"],
                    "end": h["end"],
                    "text": h["phrase"],
                    "severity": h["severity"],
                    "category": h.get("category", "unknown"),
                    "explanation": h.get("explanation", "")
                }
                for h in highlights
            ]
        }

    def create_risk_visualization(self, risk_score: float, confidence: float) -> str:
        """
        Create a visual representation of risk score

        Args:
            risk_score: Risk score (0-100)
            confidence: Confidence score (0-1)

        Returns:
            HTML visualization
        """

        # Determine risk level and color
        if risk_score >= 70:
            risk_level = "High Risk"
            color = "#dc3545"
            bg_color = "#ffe6e6"
        elif risk_score >= 40:
            risk_level = "Medium Risk"
            color = "#fd7e14"
            bg_color = "#fff3cd"
        else:
            risk_level = "Low Risk"
            color = "#28a745"
            bg_color = "#d4edda"

        # Create confidence indicator
        confidence_percent = int(confidence * 100)

        visualization = f'''
            <div class="risk-visualization" style="font-family: Arial, sans-serif; max-width: 400px; margin: 10px 0;">
                <div style="margin-bottom: 10px;">
                    <strong>Risk Assessment: {risk_level}</strong>
                </div>

                <div class="risk-meter" style="background: #f0f0f0; border-radius: 10px; height: 20px; overflow: hidden;">
                    <div style="width: {risk_score}%; background: {color}; height: 100%; transition: width 0.3s ease;">
                    </div>
                </div>

                <div style="display: flex; justify-content: space-between; font-size: 12px; margin-top: 5px;">
                    <span>0</span>
                    <span style="font-weight: bold; color: {color};">{risk_score:.1f}</span>
                    <span>100</span>
                </div>

                <div style="margin-top: 10px; font-size: 14px;">
                    Confidence: <strong>{confidence_percent}%</strong>
                </div>

                <div class="confidence-bar" style="background: #e9ecef; border-radius: 5px; height: 8px; margin-top: 5px;">
                    <div style="width: {confidence_percent}%; background: #007bff; height: 100%; border-radius: 5px;">
                    </div>
                </div>
            </div>
        '''

        return visualization

    def generate_summary_report(self, analysis_result: Dict[str, Any]) -> str:
        """
        Generate a comprehensive summary report

        Args:
            analysis_result: Complete analysis result

        Returns:
            HTML summary report
        """

        risk_score = analysis_result.get("risk_score", 0)
        threat_category = analysis_result.get("threat_category", "unknown")
        highlights = analysis_result.get("highlighted_phrases", [])
        risk_factors = analysis_result.get("risk_factors", [])

        # Risk visualization
        risk_viz = self.create_risk_visualization(
            risk_score,
            analysis_result.get("confidence", 0)
        )

        # Highlights summary
        highlights_summary = ""
        if highlights:
            high_count = sum(1 for h in highlights if h.get("severity") == "high")
            medium_count = sum(1 for h in highlights if h.get("severity") == "medium")
            low_count = sum(1 for h in highlights if h.get("severity") == "low")

            highlights_summary = f"""
                <div class="highlights-summary" style="margin: 15px 0;">
                    <h4>Suspicious Elements Found:</h4>
                    <ul>
                        {"<li>" + f"{high_count} high-risk phrases" + "</li>" if high_count else ""}
                        {"<li>" + f"{medium_count} medium-risk phrases" + "</li>" if medium_count else ""}
                        {"<li>" + f"{low_count} low-risk phrases" + "</li>" if low_count else ""}
                    </ul>
                </div>
            """

        # Risk factors
        factors_html = ""
        if risk_factors:
            factors_html = """
                <div class="risk-factors" style="margin: 15px 0;">
                    <h4>Key Risk Factors:</h4>
                    <ul>
            """

            for factor in risk_factors[:5]:  # Top 5
                factors_html += f'<li><strong>{factor["description"]}</strong> (confidence: {factor.get("score", 0):.2f})</li>'

            factors_html += "</ul></div>"

        # Build complete report
        report = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>RiskAnalyzer AI - Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                    .section {{ margin: 20px 0; padding: 15px; border: 1px solid #dee2e6; border-radius: 5px; }}
                    .highlight-high {{ color: #dc3545; background: #ffe6e6; padding: 2px 4px; border-radius: 3px; }}
                    .highlight-medium {{ color: #fd7e14; background: #fff3cd; padding: 2px 4px; border-radius: 3px; }}
                    .highlight-low {{ color: #ffc107; background: #fff8e1; padding: 2px 4px; border-radius: 3px; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h2>RiskAnalyzer AI Analysis Report</h2>
                    <p><strong>Threat Category:</strong> {threat_category.replace('_', ' ').title()}</p>
                </div>

                <div class="section">
                    <h3>Risk Assessment</h3>
                    {risk_viz}
                </div>

                {highlights_summary}

                {factors_html}

                <div class="section">
                    <h3>Analysis Summary</h3>
                    <p>{analysis_result.get("narrative_explanation", "Analysis completed.")}</p>
                </div>
            </body>
            </html>
        """

        return report


# Global text highlighter instance
text_highlighter = TextHighlighter()