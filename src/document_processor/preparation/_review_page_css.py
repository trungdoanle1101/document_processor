_CSS = """
:root {
  --text: #1f2328; --muted: #656d76; --line: #d0d7de; --page: #ffffff;
  --panel: #f6f8fa; --matter: #f3f0e8; --dropped: #fff4e5; --dropped-line: #d4a72c;
  --accent: #0969da; --over: #cf222e; --bar: #2da44e; --target: #fff8c5;
}
@media (prefers-color-scheme: dark) {
  :root {
    --text: #e6edf3; --muted: #8d96a0; --line: #30363d; --page: #0d1117;
    --panel: #161b22; --matter: #1f1d17; --dropped: #2a2112; --dropped-line: #9e7a1f;
    --accent: #4493f8; --over: #f85149; --bar: #3fb950; --target: #3a3212;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--text);
       font: 15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, "Noto Sans", sans-serif; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
pre { margin: 0; white-space: pre-wrap; font: 13px/1.45 ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace; }

/* summary */
header.summary { padding: 12px 20px; border-bottom: 1px solid var(--line); background: var(--panel); }
header.summary ul { margin: 8px 0 0; padding: 0; list-style: none; }
header.summary li { padding: 2px 0; }

/* layout: outline on the left, entries on the right; each scrolls on its own */
.layout { display: grid; grid-template-columns: minmax(220px, 320px) 1fr; }
nav.outline { position: sticky; top: 0; height: 100vh; overflow-y: auto; padding: 12px 12px 40px;
              border-right: 1px solid var(--line); font-size: 13px; }
main { padding: 16px 24px 80px; min-width: 0; }

/* outline tree */
nav.outline details > :not(summary), nav.outline details > div { margin-left: 14px; }
nav.outline summary { cursor: pointer; }
nav.outline .section { font-weight: 600; color: var(--text); }
nav.outline .item { color: var(--muted); }
nav.outline .count { color: var(--muted); font-size: 11px; margin-left: 4px; }
nav.outline .part { margin: 6px 0; font-style: italic; }

/* chunks */
section.chunk { border: 1px solid var(--line); border-radius: 6px; margin: 0 0 12px; background: var(--page); }
.chunk-header { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 10px; padding: 6px 10px;
                border-bottom: 1px solid var(--line); background: var(--panel); font-size: 12px; color: var(--muted);
                border-radius: 6px 6px 0 0; }
.chunk-id { font-weight: 700; color: var(--text); }
.breadcrumb { flex: 1 1 240px; }
.tokens { width: 80px; height: 6px; border-radius: 3px; background: var(--line); overflow: hidden; }
.bar { display: block; height: 100%; background: var(--bar); }
section.over .bar { background: var(--over); }
section.over .token-count { color: var(--over); font-weight: 700; }
section.over { border-color: var(--over); }
.segments { padding: 8px 12px; }
.segments p { margin: 4px 0; }
.heading { font-weight: 600; margin: 6px 0 2px; }
.image { color: var(--muted); font-style: italic; }
section.chunk > details { padding: 4px 12px 8px; font-size: 12px; color: var(--muted); }
section.chunk > details pre { margin-top: 6px; padding: 8px; background: var(--panel); border-radius: 4px; color: var(--text); }

/* front and end matter */
section.front_matter, section.end_matter { background: var(--matter); }

/* tables */
table { border-collapse: collapse; margin: 6px 0; font-size: 13px; width: 100%; }
th, td { border: 1px solid var(--line); padding: 4px 6px; vertical-align: top; text-align: left; }
th { background: var(--panel); }
td p, th p { margin: 0 0 2px; }

/* gaps */
section.gap.in-breadcrumb { margin: 18px 0 8px; padding: 0 2px; font-size: 17px; }
section.gap .heading { margin-left: calc(var(--depth, 0) * 1.2em); }
section.gap.dropped { border: 1px dashed var(--dropped-line); background: var(--dropped); border-radius: 6px;
                      margin: 0 0 12px; padding: 6px 12px; color: var(--muted); }
.gap-label { font-size: 12px; font-weight: 600; }

/* a warning link lands here */
section:target { box-shadow: 0 0 0 3px var(--target), 0 0 0 4px var(--dropped-line); }

@media (max-width: 760px) {
  .layout { grid-template-columns: 1fr; }
  nav.outline { position: static; height: auto; max-height: 40vh; border-right: 0; border-bottom: 1px solid var(--line); }
  main { padding: 12px 16px 60px; }
}
""".strip()
