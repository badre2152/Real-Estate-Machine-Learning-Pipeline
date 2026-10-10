"""Generate HTML reports from pipeline evaluation artifacts."""

import base64
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from logger_setup import get_logger

logger = get_logger(__name__)

def _b64_image(path: str) -> Optional[str]:
    """Encode une image en base64 pour l'intégrer dans le HTML."""
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext = Path(path).suffix.lstrip(".").lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "svg": "image/svg+xml"}.get(ext, "image/png")
    return f"data:{mime};base64,{data}"

def _metric_card(label: str, value, unit: str = "", color: str = "#4f46e5") -> str:
    fmt_value = f"{value:,.4f}" if isinstance(value, float) else str(value)
    return f"""
    <div class="metric-card">
      <div class="metric-value" style="color:{color}">{fmt_value}{unit}</div>
      <div class="metric-label">{label}</div>
    </div>"""

def _table_from_dict(data: dict, title: str = "") -> str:
    if not data:
        return ""
    rows = ""
    for k, v in data.items():
        fmt = f"{v:,.4f}" if isinstance(v, float) else str(v)
        rows += f"<tr><td>{k}</td><td><strong>{fmt}</strong></td></tr>"
    return f"""
    <div class="table-block">
      {"<h4>" + title + "</h4>" if title else ""}
      <table><thead><tr><th>Métrique</th><th>Valeur</th></tr></thead>
      <tbody>{rows}</tbody></table>
    </div>"""

def _baseline_comparison_table(model_metrics: dict, baselines: dict, metric_key: str) -> str:
    if not baselines:
        return ""
    model_val = model_metrics.get(metric_key, 0)
    rows = f"<tr><td>MODEL Modèle réel</td><td><strong>{model_val:.4f}</strong></td><td>N/A</td></tr>"
    for name, m in baselines.items():
        bv   = m.get(metric_key, 0)
        diff = model_val - bv
        sign = "+" if diff >= 0 else ""
        clr  = "green" if diff >= 0 else "red"
        rows += (
            f"<tr><td>{name}</td><td>{bv:.4f}</td>"
            f"<td style='color:{clr}'>{sign}{diff:.4f}</td></tr>"
        )
    return f"""
    <table>
      <thead><tr><th>Modèle</th><th>{metric_key}</th><th>Δ vs réel</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>"""

class ReportGenerator:
    """
    Génère un rapport HTML complet du pipeline ML.
    """

    CSS = """
    <style>
      * { box-sizing: border-box; margin: 0; padding: 0; }
      body { font-family: 'Segoe UI', sans-serif; background: #f8f9fa; color: #1a1a2e; }
      .container { max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem; }
      header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
               color: white; padding: 2.5rem 2rem; border-radius: 12px; margin-bottom: 2rem; }
      header h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.3rem; }
      header p  { opacity: 0.7; font-size: 0.9rem; }
      .section { background: white; border-radius: 10px; padding: 1.75rem;
                 margin-bottom: 1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,0.07); }
      .section h2 { font-size: 1.15rem; font-weight: 600; border-bottom: 2px solid #e5e7eb;
                    padding-bottom: 0.6rem; margin-bottom: 1.2rem; color: #1a1a2e; }
      .section h3 { font-size: 1rem; margin: 1.2rem 0 0.6rem; color: #374151; }
      .section h4 { font-size: 0.9rem; color: #6b7280; margin-bottom: 0.5rem; }
      .metrics-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
                      gap: 1rem; margin-bottom: 1rem; }
      .metric-card { background: #f3f4f6; border-radius: 8px; padding: 1rem; text-align: center; }
      .metric-value { font-size: 1.5rem; font-weight: 700; }
      .metric-label { font-size: 0.78rem; color: #6b7280; margin-top: 0.25rem; }
      table { width: 100%; border-collapse: collapse; font-size: 0.88rem; margin: 0.5rem 0; }
      th { background: #f3f4f6; font-weight: 600; padding: 0.6rem 0.8rem; text-align: left; }
      td { padding: 0.55rem 0.8rem; border-bottom: 1px solid #f0f0f0; }
      tr:last-child td { border-bottom: none; }
      .table-block { margin-bottom: 1rem; }
      .plots-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
                    gap: 1rem; margin-top: 1rem; }
      .plot-wrap img { width: 100%; border-radius: 8px; border: 1px solid #e5e7eb; }
      .plot-wrap p { font-size: 0.8rem; color: #9ca3af; text-align: center; margin-top: 0.3rem; }
      .badge { display: inline-block; padding: 0.2rem 0.7rem; border-radius: 99px;
               font-size: 0.78rem; font-weight: 600; }
      .badge-pass { background: #dcfce7; color: #166534; }
      .badge-fail { background: #fee2e2; color: #991b1b; }
      .badge-warn { background: #fef3c7; color: #92400e; }
      .alert-box { border-left: 4px solid #f59e0b; background: #fffbeb;
                   padding: 0.75rem 1rem; border-radius: 4px; margin: 0.5rem 0;
                   font-size: 0.88rem; }
      .ok-box { border-left: 4px solid #10b981; background: #ecfdf5;
                padding: 0.75rem 1rem; border-radius: 4px; margin: 0.5rem 0;
                font-size: 0.88rem; }
      footer { text-align: center; color: #9ca3af; font-size: 0.8rem; margin-top: 2rem; }
    </style>"""

    def build(
        self,
        reg_metrics: dict       = None,
        clf_metrics: dict       = None,
        baseline_reg: dict      = None,
        baseline_clf: dict      = None,
        validation_report       = None,
        shap_plots: dict        = None,
        monitoring_path: str    = None,
        pi_examples: list       = None,
        smote_report: dict      = None,
        output_dir: str         = "reports",
        filename: str           = "ml_report.html",
    ) -> str:
        """
        Génère le rapport HTML complet.

        Returns:
            Chemin vers le fichier HTML généré.
        """
        os.makedirs(output_dir, exist_ok=True)

        try:
            from config_loader import cfg
            title  = cfg.report.title
            author = cfg.report.author
        except Exception:
            title  = "Rapport ML: Avito Real Estate"
            author = "ML Pipeline automatique"

        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")

        monitoring_data = {}
        if monitoring_path and os.path.exists(monitoring_path):
            with open(monitoring_path) as f:
                monitoring_data = json.load(f)

        html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  {self.CSS}
</head>
<body>
<div class="container">
  <header>
    <h1> {title}</h1>
    <p>{author} · Généré le {timestamp}</p>
  </header>

  {self._section_summary(reg_metrics, clf_metrics)}
  {self._section_baselines(reg_metrics, clf_metrics, baseline_reg, baseline_clf)}
  {self._section_validation(validation_report)}
  {self._section_smote(smote_report)}
  {self._section_shap(shap_plots)}
  {self._section_monitoring(monitoring_data)}
  {self._section_pi(pi_examples)}

  <footer>Rapport généré automatiquement par le ML Pipeline v2 · {timestamp}</footer>
</div>
</body>
</html>"""

        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"    Rapport HTML → {path}")
        return path

    def _section_summary(self, reg: dict, clf: dict) -> str:
        reg  = reg  or {}
        clf  = clf  or {}
        r2   = reg.get("R2", 0)
        mae  = reg.get("MAE", 0)
        mape = reg.get("MAPE", 0)
        f1   = clf.get("F1", 0)
        acc  = clf.get("Accuracy", 0)
        cards = (
            _metric_card("R²", r2, color="#4f46e5" if r2 > 0.7 else "#ef4444") +
            _metric_card("MAE", mae, " MAD") +
            _metric_card("MAPE", mape, "%") +
            _metric_card("F1 Score", f1, color="#10b981" if f1 > 0.7 else "#f59e0b") +
            _metric_card("Accuracy", acc)
        )
        return f"""
  <div class="section">
    <h2> Résumé exécutif</h2>
    <div class="metrics-grid">{cards}</div>
  </div>"""

    def _section_baselines(self, reg, clf, bl_reg, bl_clf) -> str:
        if not bl_reg:
            return ""
        reg = reg or {}
        clf = clf or {}
        reg_tbl = _baseline_comparison_table(reg, bl_reg, "R2")
        clf_tbl = _baseline_comparison_table(clf, bl_clf, "F1") if bl_clf else ""
        return f"""
  <div class="section">
    <h2> Comparaison avec les Baselines</h2>
    <h3>Régression (R²)</h3>{reg_tbl}
    {"<h3>Classification (F1)</h3>" + clf_tbl if clf_tbl else ""}
  </div>"""

    def _section_validation(self, report) -> str:
        if not report:
            return ""
        badge = '<span class="badge badge-pass">PASS PASSED</span>' if report.passed \
                else '<span class="badge badge-fail">FAIL FAILED</span>'
        rows = ""
        for r in report.results[:20]:   # limiter à 20 tests pour la lisibilité
            icon = "PASS" if r.passed else ("FAIL" if r.severity == "error" else "WARNING")
            rows += f"<tr><td>{icon} {r.name}</td><td>{r.message}</td></tr>"
        return f"""
  <div class="section">
    <h2> Validation des Données {badge}</h2>
    <p style="margin-bottom:1rem; color:#6b7280">
      {report.n_rows:,} lignes · {report.n_cols} colonnes ·
      {report.n_passed} OK · {report.n_failed} erreurs · {report.n_warnings} warnings
    </p>
    <table><thead><tr><th>Test</th><th>Message</th></tr></thead>
    <tbody>{rows}</tbody></table>
  </div>"""

    def _section_smote(self, report: dict) -> str:
        if not report or not report.get("smote_applied"):
            info = "SMOTE non appliqué (données équilibrées ou non requis)"
            return f"""
  <div class="section">
    <h2> SMOTE: Équilibrage des classes</h2>
    <div class="ok-box">{info}</div>
  </div>"""
        before = report.get("before_dist", {})
        after  = report.get("after_dist", {})
        rows_b = "".join(
            f"<tr><td>{c}</td><td>{v['n']:,}</td><td>{v['pct']}%</td></tr>"
            for c, v in before.items()
        )
        rows_a = "".join(
            f"<tr><td>{c}</td><td>{v['n']:,}</td><td>{v['pct']}%</td></tr>"
            for c, v in after.items()
        )
        return f"""
  <div class="section">
    <h2> SMOTE: Équilibrage des classes</h2>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem">
      <div><h3>Avant SMOTE</h3>
        <table><thead><tr><th>Classe</th><th>N</th><th>%</th></tr></thead>
        <tbody>{rows_b}</tbody></table></div>
      <div><h3>Après SMOTE</h3>
        <table><thead><tr><th>Classe</th><th>N</th><th>%</th></tr></thead>
        <tbody>{rows_a}</tbody></table></div>
    </div>
  </div>"""

    def _section_shap(self, plots: dict) -> str:
        if not plots:
            return ""
        imgs = ""
        labels = {
            "summary"  : "Summary Plot: Impact de chaque feature",
            "bar"      : "Bar Plot: Importance moyenne |SHAP|",
            "waterfall": "Waterfall: Explication d'une prédiction",
        }
        for key, label in labels.items():
            path = plots.get(key)
            b64  = _b64_image(path)
            if b64:
                imgs += f"""
        <div class="plot-wrap">
          <img src="{b64}" alt="{label}">
          <p>{label}</p>
        </div>"""
        if not imgs:
            return ""
        return f"""
  <div class="section">
    <h2> SHAP: Interprétabilité du modèle</h2>
    <div class="plots-grid">{imgs}</div>
  </div>"""

    def _section_monitoring(self, data: dict) -> str:
        if not data or "steps" not in data:
            return ""
        total = data.get("total_duration_s", 0)
        rows  = ""
        for s in data.get("steps", []):
            status = "PASS" if s.get("success") else "FAIL"
            dur    = f"{(s.get('duration_s') or 0):.2f}s"
            rows  += f"<tr><td>{status} {s['name']}</td><td>{dur}</td></tr>"
        alerts = data.get("alerts", [])
        alert_html = ""
        for a in alerts:
            alert_html += f'<div class="alert-box"> {a["metric"]}={a["value"]:.4f} &lt; seuil={a["threshold"]}</div>'
        if not alerts:
            alert_html = '<div class="ok-box">PASS Aucune alerte: toutes les métriques dans les seuils</div>'
        return f"""
  <div class="section">
    <h2> Monitoring: Exécution du pipeline</h2>
    <p style="margin-bottom:0.75rem;color:#6b7280">Durée totale : <strong>{total:.1f}s</strong></p>
    <table><thead><tr><th>Étape</th><th>Durée</th></tr></thead>
    <tbody>{rows}</tbody></table>
    <h3 style="margin-top:1rem">Alertes</h3>{alert_html}
  </div>"""

    def _section_pi(self, examples: list) -> str:
        if not examples:
            return ""
        rows = ""
        for i, ex in enumerate(examples[:10]):
            rows += (
                f"<tr><td>#{i+1}</td>"
                f"<td>{ex.get('prediction', 0):,.0f}</td>"
                f"<td>{ex.get('lower', 0):,.0f}</td>"
                f"<td>{ex.get('upper', 0):,.0f}</td>"
                f"<td>{ex.get('interval_width', 0):,.0f}</td></tr>"
            )
        return f"""
  <div class="section">
    <h2> Intervalles de Prédiction (95% CI)</h2>
    <table>
      <thead><tr>
        <th>#</th><th>Prédiction (MAD)</th>
        <th>Borne basse</th><th>Borne haute</th><th>Largeur</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>"""