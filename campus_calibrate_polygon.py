"""
campus_calibrate_polygon.py
브라우저에서 건물 외곽선(폴리곤) 좌표 수집

실행: python campus_calibrate_polygon.py
접속: http://localhost:8765
"""
import base64, json
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

MAP_PATH = '/home/sean429/swe3032/maps/카카오맵확대.png'
PORT = 8765

# 따야 할 건물 목록
BUILDING_LIST = [
    '05', '21', '22', '23', '24', '25', '26', '27', '31', '32', '33', '40',
    '51', '53', '61', '62', '71', '83', '85', '86'
]

img_b64 = base64.b64encode(Path(MAP_PATH).read_bytes()).decode()

HTML = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Campus Polygon Calibration</title>
<style>
  body {{ margin: 0; background: #1a1a2e; color: #eee; font-family: monospace; display: flex; gap: 16px; padding: 12px; height: 100vh; overflow: hidden; }}
  #panel {{ width: 320px; flex-shrink: 0; display: flex; flex-direction: column; gap: 10px; overflow-y: auto; }}
  h2 {{ margin: 0; color: #7ecfff; font-size: 16px; }}
  #status {{ background: #0f3460; border-radius: 8px; padding: 12px; }}
  #target {{ color: #ffcc00; font-size: 18px; font-weight: bold; margin-bottom: 5px; }}
  #instr {{ font-size: 12px; color: #ccc; }}
  #log {{ background: #0d0d1a; border-radius: 8px; padding: 8px; font-size: 11px; flex: 1; overflow-y: auto; }}
  #output {{ background: #0d1a0d; border-radius: 8px; padding: 10px; font-size: 11px; white-space: pre; display: none; max-height: 200px; overflow: auto; }}
  .btns {{ display: flex; gap: 5px; flex-wrap: wrap; }}
  button {{ background: #e94560; border: none; color: #fff; padding: 8px 12px; border-radius: 6px; cursor: pointer; font-size: 13px; }}
  button:hover {{ background: #c73652; }}
  button.secondary {{ background: #4e4e6a; }}
  #wrap {{ position: relative; overflow: auto; flex: 1; border: 2px solid #333; background: #000; }}
  #map {{ display: block; cursor: crosshair; }}
  canvas {{ position: absolute; top: 0; left: 0; pointer-events: none; }}
</style>
</head>
<body>
<div id="panel">
  <h2>Building Polygon Tool</h2>
  <div id="status">
    <div id="instr">Click corners of the building:</div>
    <div id="target">Loading...</div>
    <div id="coords">x=0, y=0</div>
  </div>
  <div class="btns">
    <button onclick="finishBuilding()">✅ Finish & Next</button>
    <button class="secondary" onclick="undoPoint()">↩ Undo Point</button>
    <button class="secondary" onclick="resetCurrent()">Reset Current</button>
  </div>
  <div id="log"></div>
  <div id="output"></div>
  <button style="background:#28a745" onclick="exportData()">💾 Generate Code</button>
</div>
<div id="wrap">
  <img id="map" src="data:image/png;base64,{img_b64}">
  <canvas id="canvas"></canvas>
</div>

<script>
const BUILDINGS = {json.dumps(BUILDING_LIST)};
let currentIdx = 0;
let polygons = {{}}; // {{ '85': [[x1,y1], [x2,y2], ...], ... }}
let currentPoints = [];

const map = document.getElementById('map');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

map.onload = () => {{
  canvas.width = map.naturalWidth;
  canvas.height = map.naturalHeight;
  updateStatus();
  draw();
}};

map.addEventListener('mousemove', e => {{
  const r = map.getBoundingClientRect();
  const sx = map.naturalWidth / map.clientWidth;
  const sy = map.naturalHeight / map.clientHeight;
  const x = Math.round((e.clientX - r.left) * sx);
  const y = Math.round((e.clientY - r.top)  * sy);
  document.getElementById('coords').textContent = `x=${{x}}, y=${{y}}`;
}});

map.addEventListener('click', e => {{
  if (currentIdx >= BUILDINGS.length) return;
  const r = map.getBoundingClientRect();
  const sx = map.naturalWidth / map.clientWidth;
  const sy = map.naturalHeight / map.clientHeight;
  const x = Math.round((e.clientX - r.left) * sx);
  const y = Math.round((e.clientY - r.top)  * sy);
  currentPoints.push([x, y]);
  draw();
}});

function draw() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Draw already finished polygons
  ctx.lineWidth = 3;
  for (const [id, pts] of Object.entries(polygons)) {{
    ctx.strokeStyle = 'rgba(100, 200, 255, 0.8)';
    ctx.fillStyle = 'rgba(100, 200, 255, 0.3)';
    drawPoly(pts, true, id);
  }}
  
  // Draw current points
  ctx.strokeStyle = '#ffcc00';
  ctx.fillStyle = 'rgba(255, 204, 0, 0.4)';
  ctx.lineWidth = 4;
  drawPoly(currentPoints, false, BUILDINGS[currentIdx]);
}}

function drawPoly(pts, close, label) {{
  if (pts.length === 0) return;
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) {{
    ctx.lineTo(pts[i][0], pts[i][1]);
  }}
  if (close) ctx.closePath();
  ctx.stroke();
  if (close) ctx.fill();
  
  // Points
  ctx.fillStyle = 'white';
  pts.forEach(p => {{
    ctx.beginPath();
    ctx.arc(p[0], p[1], 4, 0, Math.PI*2);
    ctx.fill();
  }});

  // Label at center
  if (pts.length > 0) {{
    const cx = pts.reduce((a,b)=>a+b[0], 0) / pts.length;
    const cy = pts.reduce((a,b)=>a+b[1], 0) / pts.length;
    ctx.font = 'bold 20px monospace';
    ctx.fillStyle = 'white';
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 1;
    ctx.textAlign = 'center';
    ctx.strokeText(label, cx, cy);
    ctx.fillText(label, cx, cy);
  }}
}}

function finishBuilding() {{
  if (currentPoints.length < 3) {{
    alert('Need at least 3 points for a polygon!');
    return;
  }}
  const id = BUILDINGS[currentIdx];
  polygons[id] = [...currentPoints];
  appendLog(id, currentPoints.length);
  currentPoints = [];
  currentIdx++;
  updateStatus();
  draw();
}}

function undoPoint() {{
  currentPoints.pop();
  draw();
}}

function resetCurrent() {{
  currentPoints = [];
  draw();
}}

function updateStatus() {{
  const tgt = document.getElementById('target');
  if (currentIdx < BUILDINGS.length) {{
    tgt.textContent = `${{currentIdx + 1}}/${{BUILDINGS.length}}: Building ${{BUILDINGS[currentIdx]}}`;
  }} else {{
    tgt.textContent = '✅ All Buildings Done!';
  }}
}}

function appendLog(id, count) {{
  const el = document.getElementById('log');
  el.innerHTML = `<div><b>${{id}}</b>: ${{count}} points</div>` + el.innerHTML;
}}

function exportData() {{
  let lines = ["BUILDING_POLY = {{"];
  for (const id of BUILDINGS) {{
    if (polygons[id]) {{
      const ptsStr = polygons[id].map(p => `(${{p[0]}}, ${{p[1]}})`).join(', ');
      lines.push(`    '${{id}}': [${{ptsStr}}],`);
    }}
  }}
  lines.push("}}");
  const out = document.getElementById('output');
  out.textContent = lines.join('\\n');
  out.style.display = 'block';
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
print('1. 마우스로 건물의 각 모서리를 클릭하세요.')
print('2. 한 건물을 다 그렸으면 [Finish & Next]를 누르세요.')
print('3. 모든 건물을 마쳤으면 [Generate Code]를 눌러 결과를 복사하세요.')
HTTPServer(('0.0.0.0', PORT), Handler).serve_forever()
