"""Render the executed notebook + summary into one A4 report (HTML -> PDF with headless Edge/Chrome).

    python tools/build_report.py
"""
import datetime as dt
import html
import json
import os
import re
import shutil
import subprocess
import sys

import mistune
import nbformat
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(ROOT, "report")
NOTEBOOK = os.path.join(ROOT, "notebooks", "FlipItNews_News_Classification.ipynb")
AUTHOR = "Guntuku Hariprasad"
PDF_PATH = os.path.join(ROOT, "docs", "FlipItNews_NLP_GuntukuHariprasad.pdf")
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe", "google-chrome", "chromium"]

md = mistune.create_markdown(plugins=["table", "strikethrough"], escape=False)
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def render_output(o):
    t = o.get("output_type")
    if t == "stream":
        if o.get("name") == "stderr":
            return ""
        txt = ANSI.sub("", o["text"]).strip("\n").rstrip()
        return f'<pre class="out">{html.escape(txt)}</pre>' if txt else ""
    if t in ("execute_result", "display_data"):
        d = o.get("data", {})
        if "image/png" in d:
            return f'<div class="fig"><img src="data:image/png;base64,{d["image/png"]}"/></div>'
        if "text/html" in d:
            h = re.sub(r"<style[\s\S]*?</style>", "", d["text/html"])
            return f'<div class="tbl">{h}</div>'
        if "text/plain" in d:
            return f'<pre class="out">{html.escape(ANSI.sub("", d["text/plain"]))}</pre>'
    if t == "error":
        return f'<pre class="out err">{html.escape(o["ename"] + ": " + o["evalue"])}</pre>'
    return ""


def render_notebook(path):
    nb = nbformat.read(path, as_version=4)
    parts, n = [], 0
    for c in nb.cells:
        if c.cell_type == "markdown":
            h = md(c.source)
            h = re.sub(r"<h1>(.*?)</h1>", r'<h1 class="part">\1</h1>', h, count=1)
            parts.append(f'<div class="md">{h}</div>')
        elif c.cell_type == "code":
            n += 1
            code = highlight(c.source, PythonLexer(), HtmlFormatter(nowrap=True))
            outs = "".join(render_output(o) for o in c.get("outputs", []))
            parts.append(f'<div class="cell"><div class="cin"><span class="num">In [{n}]</span>'
                         f'<pre class="code">{code}</pre></div>{outs}</div>')
    return "\n".join(parts)


def fill(text, values):
    return re.sub(r"\{\{(\w+)\}\}", lambda m: str(values.get(m.group(1), m.group(0))), text)


def find_browser():
    for b in BROWSERS:
        if os.path.exists(b) or shutil.which(b):
            return b
    sys.exit("No Edge/Chrome found to print the PDF")


def main():
    card = json.load(open(os.path.join(ROOT, "models", "model_card.json")))
    m = card["metrics"]
    v = {"lr_acc": f"{100 * m['logistic_regression__tfidf']['accuracy']:.1f}%",
         "nb_acc": f"{100 * m['naive_bayes__tfidf']['accuracy']:.1f}%",
         "author": AUTHOR, "date": dt.date.today().strftime("%d %B %Y")}
    summary = md(fill(open(os.path.join(REPORT_DIR, "summary.md"), encoding="utf-8").read(), v))
    body = fill(open(os.path.join(REPORT_DIR, "template.html"), encoding="utf-8").read(), {
        **v, "summary": summary, "notebook": render_notebook(NOTEBOOK),
        "pygments_css": HtmlFormatter(style="friendly").get_style_defs(".code")})
    html_path = os.path.join(REPORT_DIR, "report.html")
    open(html_path, "w", encoding="utf-8").write(body)

    os.makedirs(os.path.dirname(PDF_PATH), exist_ok=True)
    subprocess.run([find_browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=20000", f"--print-to-pdf={PDF_PATH}",
                    "file:///" + html_path.replace("\\", "/")], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote", PDF_PATH, f"({os.path.getsize(PDF_PATH) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
