"""
campus_calibrate_server.py
브라우저에서 지도 클릭 → 건물 픽셀 좌표 수집

실행: python campus_calibrate_server.py
접속: http://localhost:8765
"""
import base64, json
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

SAVE_PATH = '/home/sean429/swe3032/building_px.json'

MAP_PATH = '/home/sean429/swe3032/카카오맵확대.png'
PORT = 8765

CLICK_ORDER = ['85', '26', '23', '22', '21', '33', '40']
NAMES = {
    '85': '산학협력센터(85)', '26': '제2공학관(26)', '23': '공과대학(23)',
    '22': '제1공학관(22)',   '21': '정보통신대학(21)', '33': '화학관(33)',
    '40': '반도체관(40)',
}
OLD_PX = {
    '85': (1139, 342), '26': (1626, 596), '23': (1641, 826),
    '22': (1615, 1027), '21': (1571, 1087), '33': (1578, 1562), '40': (1717, 1663),
}

img_b64 = base64.b64encode(Path(MAP_PATH).read_bytes()).decode()

HTML = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Campus Calibration</title>
<style>
  body {{ margin: 0; background: #1a1a2e; color: #eee; font-family: monospace; display: flex; gap: 16px; padding: 12px; }}
  #panel {{ width: 280px; flex-shrink: 0; }}
  h2 {{ margin: 0 0 10px; color: #7ecfff; font-size: 15px; }}
  #status {{ background: #0f3460; border-radius: 8px; padding: 10px; margin-bottom: 10px; font-size: 13px; }}
  #target {{ color: #ffcc00; font-size: 15px; font-weight: bold; }}
  #coords {{ color: #aaffaa; font-size: 13px; margin-top: 6px; }}
  #log {{ background: #0d0d1a; border-radius: 8px; padding: 8px; font-size: 11px; line-height: 1.7; max-height: 320px; overflow-y: auto; }}
  #output {{ margin-top: 10px; background: #0d1a0d; border-radius: 8px; padding: 10px; font-size: 11px; white-space: pre; display: none; }}
  button {{ margin-top: 8px; background: #e94560; border: none; color: #fff; padding: 7px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; }}
  button:hover {{ background: #c73652; }}
  #wrap {{ position: relative; overflow: auto; flex: 1; }}
  #map {{ display: block; cursor: crosshair; }}
  .dot {{ position: absolute; width: 18px; height: 18px; border-radius: 50%; border: 2px solid white; transform: translate(-50%,-50%); pointer-events: none; display: flex; align-items: center; justify-content: center; font-size: 9px; font-weight: bold; color: white; }}
  .dot-old {{ background: rgba(150,150,150,0.5); }}
  .dot-new {{ background: rgba(220,50,50,0.85); }}
</style>
</head>
<body>
<div id="panel">
  <h2>Campus Calibration</h2>
  <div id="status">
    <div>Click building:</div>
    <div id="target">— </div>
    <div id="coords">hover to see coordinates</div>
  </div>
  <div id="log"></div>
  <div id="output"></div>
  <button onclick="undo()">↩ Undo</button>
  <button onclick="reset()">Reset</button>
</div>
<div id="wrap">
  <img id="map" src="data:image/png;base64,{img_b64}">
</div>

<script>
const ORDER  = {json.dumps(CLICK_ORDER)};
const NAMES  = {json.dumps(NAMES)};
const OLD_PX = {json.dumps(OLD_PX)};
let collected = {{}};

const map  = document.getElementById('map');
const wrap = document.getElementById('wrap');

// Draw old positions
map.onload = () => {{
  for (const [bld, [x, y]] of Object.entries(OLD_PX)) {{
    addDot(x, y, bld, 'old');
  }}
  updateStatus();
}};
if (map.complete) map.onload();

map.addEventListener('mousemove', e => {{
  const r = map.getBoundingClientRect();
  const sx = map.naturalWidth / map.clientWidth;
  const sy = map.naturalHeight / map.clientHeight;
  const x = Math.round((e.clientX - r.left) * sx);
  const y = Math.round((e.clientY - r.top)  * sy);
  document.getElementById('coords').textContent = `x=${{x}}, y=${{y}}`;
}});

map.addEventListener('click', e => {{
  const done = Object.keys(collected).length;
  if (done >= ORDER.length) return;
  const bld = ORDER[done];
  const r   = map.getBoundingClientRect();
  const sx  = map.naturalWidth  / map.clientWidth;
  const sy  = map.naturalHeight / map.clientHeight;
  const x   = Math.round((e.clientX - r.left) * sx);
  const y   = Math.round((e.clientY - r.top)  * sy);
  collected[bld] = [x, y];
  addDot(x, y, bld, 'new');
  appendLog(bld, x, y);
  updateStatus();
}});

function addDot(x, y, label, type) {{
  const sx = map.clientWidth  / map.naturalWidth;
  const sy = map.clientHeight / map.naturalHeight;
  const d  = document.createElement('div');
  d.className = `dot dot-${{type}}`;
  d.id = `dot-${{type}}-${{label}}`;
  d.textContent = label;
  d.style.left = (map.offsetLeft + x * sx) + 'px';
  d.style.top  = (map.offsetTop  + y * sy) + 'px';
  wrap.appendChild(d);
}}

function appendLog(bld, x, y) {{
  const el = document.getElementById('log');
  el.innerHTML += `<span style="color:#7ecfff">${{bld}}</span> ${{NAMES[bld]}}<br><span style="color:#aaffaa">  → (${{x}}, ${{y}})</span><br>`;
  el.scrollTop = el.scrollHeight;
}}

function updateStatus() {{
  const done = Object.keys(collected).length;
  const tgt  = document.getElementById('target');
  if (done < ORDER.length) {{
    tgt.textContent = NAMES[ORDER[done]];
  }} else {{
    tgt.textContent = '✅ Done!';
    showOutput();
  }}
}}

function showOutput() {{
  const lines = ["BUILDING_PX = {{"];
  for (const bld of ORDER) {{
    const [x, y] = collected[bld];
    lines.push(`    '${{bld}}': (${{x}}, ${{y}}),`);
  }}
  lines.push("}}");
  const out = document.getElementById('output');
  out.textContent = lines.join('\\n');
  out.style.display = 'block';
}}

function undo() {{
  const done = Object.keys(collected).length;
  if (done === 0) return;
  const bld = ORDER[done - 1];
  delete collected[bld];
  const d = document.getElementById(`dot-new-${{bld}}`);
  if (d) d.remove();
  document.getElementById('log').innerHTML = '';
  for (const b of ORDER.slice(0, done - 1)) appendLog(b, ...collected[b]);
  document.getElementById('output').style.display = 'none';
  updateStatus();
}}

function reset() {{
  collected = {{}};
  document.querySelectorAll('.dot-new').forEach(d => d.remove());
  document.getElementById('log').innerHTML = '';
  document.getElementById('output').style.display = 'none';
  updateStatus();
}}
</script>
</body>
</html>
"""

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(HTML.encode())
    def log_message(self, *a): pass

print(f'http://localhost:{PORT}  을 브라우저에서 열어주세요')
print('Ctrl+C 로 종료')
HTTPServer(('0.0.0.0', PORT), Handler).serve_forever()
