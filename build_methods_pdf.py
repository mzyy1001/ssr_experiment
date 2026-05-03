"""Render data_methods_doc.md → data_methods_doc.pdf via weasyprint with CJK font."""
import markdown
import weasyprint

with open("/home/mzyy1001/business/data_methods_doc.md", encoding="utf-8") as f:
    md_text = f.read()

html_body = markdown.markdown(
    md_text,
    extensions=["tables", "fenced_code", "toc", "sane_lists"],
)

CSS = """
@page {
  size: A4;
  margin: 1.6cm 1.7cm;
  @bottom-center {
    content: counter(page) " / " counter(pages);
    font-family: "Droid Sans Fallback", "DejaVu Sans", sans-serif;
    font-size: 9pt; color: #666;
  }
}
body {
  font-family: "Droid Sans Fallback", "DejaVu Sans", sans-serif;
  font-size: 9.5pt;
  line-height: 1.45;
  color: #1c1c1c;
}
h1 { font-size: 18pt; margin: 0 0 6pt 0; color: #1f3a5f; border-bottom: 2px solid #1f3a5f; padding-bottom: 4pt; }
h2 { font-size: 13pt; margin: 14pt 0 4pt 0; color: #1f3a5f; }
h3 { font-size: 11pt; margin: 10pt 0 3pt 0; color: #2a557a; }
h4 { font-size: 10pt; margin: 8pt 0 2pt 0; color: #444; }
p { margin: 4pt 0; }
ul, ol { margin: 4pt 0 4pt 16pt; padding: 0; }
li { margin: 1pt 0; }
strong { color: #2a2a2a; }
em { color: #444; }
table {
  border-collapse: collapse;
  margin: 6pt 0;
  font-size: 9pt;
  width: auto;
  page-break-inside: avoid;
}
th, td {
  border: 0.6pt solid #aaa;
  padding: 3pt 6pt;
  text-align: left;
  vertical-align: top;
}
th { background: #e8eef5; font-weight: 600; }
tr:nth-child(even) td { background: #fafafa; }
code {
  font-family: "DejaVu Sans Mono", "Droid Sans Fallback", monospace;
  font-size: 8.5pt;
  background: #f3f3f3;
  padding: 0 2pt;
  border-radius: 2pt;
}
pre {
  background: #f5f5f5;
  border: 0.5pt solid #d0d0d0;
  border-left: 3pt solid #2a557a;
  padding: 6pt 8pt;
  font-family: "DejaVu Sans Mono", "Droid Sans Fallback", monospace;
  font-size: 8.2pt;
  line-height: 1.35;
  page-break-inside: avoid;
  white-space: pre-wrap;
  word-wrap: break-word;
}
pre code { background: transparent; padding: 0; font-size: 8.2pt; }
hr { border: 0; border-top: 0.6pt solid #ccc; margin: 10pt 0; }
"""

html = f"""<!doctype html>
<html><head><meta charset="utf-8"></head>
<body>{html_body}</body></html>"""

weasyprint.HTML(string=html).write_pdf(
    "/home/mzyy1001/business/data_methods_doc.pdf",
    stylesheets=[weasyprint.CSS(string=CSS)],
)
print("Wrote data_methods_doc.pdf")
