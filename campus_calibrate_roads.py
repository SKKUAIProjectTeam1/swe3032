"""
campus_calibrate_roads.py
브라우저에서 도로 길목(Node) 및 연결(Edge) 좌표 수집

실행: python campus_calibrate_roads.py
접속: http://localhost:8765
"""
import base64, json
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

MAP_PATH = '/home/sean429/swe3032/maps/카카오맵확대.png'
PORT = 8765

img_b64 = base64.b64encode(Path(MAP_PATH).read_bytes()).decode()

HTML = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Campus Road Network Tool</title>
<style>
  body {{ margin: 0; background: #1a1a2e; color: #eee; font-family: monospace; display: flex; gap: 16px; padding: 12px; height: 100vh; overflow: hidden; }}
  #panel {{ width: 320px; flex-shrink: 0; display: flex; flex-direction: column; gap: 10px; }}
  h2 {{ margin: 0; color: #7ecfff; font-size: 16px; }}
  #status {{ background: #0f3460; border-radius: 8px; padding: 12px; font-size: 13px; line-height: 1.5; }}
  .mode-indicator {{ font-weight: bold; padding: 4px 8px; border-radius: 4px; display: inline-block; margin-top: 5px; }}
  .mode-node {{ background: #28a745; color: white; }}
  .mode-link {{ background: #ffc107; color: black; }}
  #log {{ background: #0d0d1a; border-radius: 8px; padding: 8px; font-size: 11px; flex: 1; overflow-y: auto; }}
  #output {{ background: #0d1a0d; border-radius: 8px; padding: 10px; font-size: 11px; white-space: pre; display: none; max-height: 250px; overflow: auto; }}
  button {{ background: #e94560; border: none; color: #fff; padding: 10px; border-radius: 6px; cursor: pointer; }}
  button.secondary {{ background: #4e4e6a; }}
  #wrap {{ position: relative; overflow: auto; flex: 1; border: 2px solid #444; background: #000; }}
  #map {{ display: block; cursor: crosshair; }}
  canvas {{ position: absolute; top: 0; left: 0; pointer-events: none; }}
  .dot {{ position: absolute; width: 14px; height: 14px; background: #00ff00; border: 2px solid white; border-radius: 50%; transform: translate(-50%,-50%); cursor: pointer; pointer-events: auto; zorder: 10; }}
  .dot.selected {{ background: #ff00ff; box-shadow: 0 0 10px #ff00ff; }}
</style>
</head>
<body>
<div id="panel">
  <h2>Road Network Tool</h2>
  <div id="status">
    <b>[현재 모드]</b> <span id="modeLab" class="mode-indicator mode-node">Node 추가</span><br>
    - <b>Node 모드</b>: 지도 클릭 시 길목 생성<br>
    - <b>Link 모드</b>: 점 2개를 순서대로 클릭해 연결<br><br>
    <button onclick="toggleMode()">🔄 모드 전환 (Space)</button>
  </div>
  <div class="btns" style="display:flex; gap:5px;">
    <button onclick="undo()" style="flex:1">↩ Undo</button>
    <button onclick="resetAll()" class="secondary">Reset</button>
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
let nodes = []; // [[x, y], ...]
let edges = []; // [[i, j], ...]
let mode = 'node'; // 'node' or 'link'
let selectedNodeIdx = null;

const map = document.getElementById('map');
const wrap = document.getElementById('wrap');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

map.onload = () => {{
  canvas.width = map.naturalWidth;
  canvas.height = map.naturalHeight;
}};

function toggleMode() {{
  mode = (mode === 'node' ? 'link' : 'node');
  const lab = document.getElementById('modeLab');
  lab.textContent = mode === 'node' ? 'Node 추가' : 'Link 연결';
  lab.className = 'mode-indicator ' + (mode === 'node' ? 'mode-node' : 'mode-link');
  selectedNodeIdx = null;
  render();
}}

window.addEventListener('keydown', e => {{ if(e.code === 'Space') {{ e.preventDefault(); toggleMode(); }} }});

map.addEventListener('click', e => {{
  if (mode !== 'node') return;
  const r = map.getBoundingClientRect();
  const sx = map.naturalWidth / map.clientWidth;
  const sy = map.naturalHeight / map.clientHeight;
  const x = Math.round((e.clientX - r.left) * sx);
  const y = Math.round((e.clientY - r.top)  * sy);
  nodes.push([x, y]);
  render();
}});

function onNodeClick(idx) {{
  if (mode !== 'link') return;
  if (selectedNodeIdx === null) {{
    selectedNodeIdx = idx;
  }} else if (selectedNodeIdx === idx) {{
    selectedNodeIdx = null;
  }} else {{
    // 점 두 개가 다르면 Edge 추가
    const exists = edges.some(e => (e[0]===selectedNodeIdx && e[1]===idx) || (e[0]===idx && e[1]===selectedNodeIdx));
    if (!exists) edges.push([selectedNodeIdx, idx]);
    selectedNodeIdx = null;
  }}
  render();
}}

function render() {{
  // 1. 기존 요소 제거
  document.querySelectorAll('.dot').forEach(el => el.remove());
  const sx = map.clientWidth / map.naturalWidth;
  const sy = map.naturalHeight / map.clientHeight;
  
  // 2. 캔버스에 선 그리기
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#ffc107';
  ctx.lineWidth = 5;
  edges.forEach(e => {{
    const p1 = nodes[e[0]];
    const p2 = nodes[e[1]];
    ctx.beginPath();
    ctx.moveTo(p1[0], p1[1]);
    ctx.lineTo(p2[0], p2[1]);
    ctx.stroke();
  }});

  // 3. 점(Node) 생성
  nodes.forEach((p, i) => {{
    const dot = document.createElement('div');
    dot.className = 'dot' + (selectedNodeIdx === i ? ' selected' : '');
    dot.style.left = (map.offsetLeft + p[0] * sx) + 'px';
    dot.style.top = (map.offsetTop + p[1] * sy) + 'px';
    dot.onclick = (e) => {{ e.stopPropagation(); onNodeClick(i); }};
    
    // 툴팁으로 이름 표시
    dot.title = 'R' + (i+1);
    wrap.appendChild(dot);
  }});

  updateLog();
}}

function undo() {{
  if (mode === 'node') nodes.pop();
  else edges.pop();
  render();
}}

function resetAll() {{
  if(confirm('정말 초기화하시겠습니까?')) {{
    nodes = []; edges = []; render();
  }}
}}

function updateLog() {{
  const el = document.getElementById('log');
  el.innerHTML = `<div><b>Nodes:</b> ${{nodes.length}}</div>` + 
                 `<div><b>Edges:</b> ${{edges.length}}</div>` +
                 nodes.map((p, i) => `<div style="font-size:9px; color:#aaa;">R${{i+1}}: (${{p[0]}}, ${{p[1]}})</div>`).join('');
}}

function exportData() {{
  let lines = ["ROAD_NODES = {{"];
  nodes.forEach((p, i) => {{
    lines.push(`    'R${{i+1}}': (${{p[0]}}, ${{p[1]}}),`);
  }});
  lines.push("}}\\n\\nROAD_EDGES = [");
  edges.forEach(e => {{
    lines.push(`    ('R${{e[0]+1}}', 'R${{e[1]+1}}'),`);
  }});
  lines.push("]");
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

print(f'http://localhost:{PORT} 접속')
print('1. Node 모드에서 길목들을 클릭해 찍으세요.')
print('2. Space키를 눌러 Link 모드로 바꾼 후, 점 2개를 순서대로 클릭해 길을 이으세요.')
print('3. Generate Code 버튼을 눌러 전체 네트워크 데이터를 복사하세요.')
HTTPServer(('0.0.0.0', PORT), Handler).serve_forever()
