"""
성균관대학교 킹고정보 수강인원 크롤러
- SKKU_ID / SKKU_PW 환경변수로 로그인
- 자연과학캠퍼스 전체 학부/학과 순회
- 교과목개요(상세) 팝업 → 최근 학기 수강인원합(명)
- sugang_full.csv 저장
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import os, time, re, json, random
import pandas as pd
import requests as req_lib

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException  # noqa: F401 — kept for WebDriverWait internals
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager

LOGIN_URL = "https://kingoinfo.skku.edu/gaia/nxui/index.html"

# .env 파일 우선, 없으면 환경변수
def _load_env():
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())

_load_env()
SKKU_ID = os.environ.get('SKKU_ID', '')
SKKU_PW = os.environ.get('SKKU_PW', '')

_EMPTY = {'', '-', '미지정', 'tba', '강의실미지정', '온라인', '원격', '없음', '온라인강의', '미배정'}

# JS XHR 인터셉터 — 모든 XHR 응답 본문을 window._xhrQ에 누적
_XHR_INIT = """
if (!window._xhrQ) {
    window._xhrQ = [];
    const _oo = XMLHttpRequest.prototype.open;
    const _os = XMLHttpRequest.prototype.send;
    XMLHttpRequest.prototype.open = function(m, u) {
        this._xhrUrl = u;
        return _oo.apply(this, arguments);
    };
    XMLHttpRequest.prototype.send = function(b) {
        this._xhrReqBody = b || '';
        this.addEventListener('load', function() {
            window._xhrQ.push({url: this._xhrUrl, req: this._xhrReqBody, resp: this.responseText});
        });
        return _os.apply(this, arguments);
    };
}
"""


# ── 공통 유틸 (crawl_sugang.py 동일) ─────────────────────────────────────────

def parse_building(room: str):
    if not room:
        return 0
    s = room.strip()
    if s.lower() in _EMPTY or '미지정' in s:
        return 0
    m = re.search(r'[가-힣A-Za-z0-9]{1,12}(?:관|동\d*|홀|센터|빌딩)', s, re.IGNORECASE)
    if m:
        return m.group()
    for r in re.findall(r'【([^】]+)】', s):
        r = r.strip()
        if not r or '미지정' in r:
            return 0
        m2 = re.match(r'(\d{2})\d+', r)
        if m2:
            return m2.group(1)
        return r
    return 0


_TIME_RE = re.compile(r'[월화수목금토일]\d{1,2}:\d{2}')


def _find_time_field(record: dict) -> str:
    """SSV 레코드에서 시간대 패턴(월09:00)을 가진 필드명을 자동 감지"""
    for k, v in record.items():
        if isinstance(v, str) and _TIME_RE.search(v):
            return k
    return ''


def parse_ssv(content: str) -> list[dict]:
    RS, FS = '\x1e', '\x1f'
    columns = None
    records = []
    for sec in content.split(RS):
        if not sec:
            continue
        if sec.startswith('_RowType_') or sec.startswith('_RowType\x1f'):
            columns = [re.sub(r':.*$', '', c.strip()) for c in sec.split(FS)]
        elif columns and sec.startswith('N'):
            vals = sec.split(FS)
            if len(vals) < len(columns):
                vals += [''] * (len(columns) - len(vals))
            records.append(dict(zip(columns, vals[:len(columns)])))
    return records


def ac_click(driver, el):
    ActionChains(driver).move_to_element(el).click().perform()


def open_dropdown(driver, idx, wait=2):
    btns = driver.find_elements(By.XPATH, "//div[@class='ButtonControl dropbutton']")
    if idx >= len(btns):
        raise RuntimeError(f"dropbutton[{idx}] 없음 (총 {len(btns)}개)")
    ac_click(driver, btns[idx])
    time.sleep(wait)


def click_option(driver, text, timeout=10):
    el = WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.XPATH, f"//div[contains(text(),'{text}')]"))
    )
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
    time.sleep(0.3)
    try:
        ac_click(driver, el)
    except Exception:
        driver.execute_script("arguments[0].click();", el)


def delay():
    time.sleep(random.uniform(0.5, 1.0))


# ── XHR 큐 헬퍼 ──────────────────────────────────────────────────────────────

def flush_xhr(driver):
    driver.execute_script(_XHR_INIT)   # 인터셉터 소실 시 재주입
    driver.execute_script("window._xhrQ = [];")


def read_xhr(driver, url_substr=None) -> list[dict]:
    items = driver.execute_script("return window._xhrQ || [];")
    if url_substr:
        items = [x for x in items if url_substr in (x.get('url') or '')]
    return items


# ── 로그인 ────────────────────────────────────────────────────────────────────

def login(driver):
    """
    Nexacro 렌더링 로그인.
    input id: edtLOGIN_ID:input / edtLOGIN_PWD:input (둘 다 type=text)
    로그인 후 시스템 메시지 팝업 → 계속하기 클릭 → 메인화면 대기
    """
    print("[LOGIN] kingoinfo 접속...")
    driver.get(LOGIN_URL)
    print("  Nexacro 렌더링 대기 (15초)...")
    time.sleep(15)
    print(f"  현재 URL: {driver.current_url}")

    # ① ID / PW 입력
    try:
        id_el = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[id*='edtLOGIN_ID']"))
        )
        pw_el = driver.find_element(By.CSS_SELECTOR, "input[id*='edtLOGIN_PWD']")
        id_el.click(); id_el.clear(); id_el.send_keys(SKKU_ID)
        time.sleep(0.5)
        pw_el.click(); pw_el.clear(); pw_el.send_keys(SKKU_PW)
        time.sleep(0.5)
    except Exception as e:
        print(f"  [ERROR] 로그인 폼 입력 실패: {e}")
        print("  수동 로그인 후 Enter를 누르세요...")
        input()
        return

    # ② 로그인 버튼 클릭 (Nexacro div)
    clicked = False
    for xpath in [
        "//div[contains(@id,'btnLOGIN')]",
        "//div[contains(@id,'LOGIN') and contains(@class,'Button')]",
        "//div[text()='로그인']",
    ]:
        try:
            btn = driver.find_element(By.XPATH, xpath)
            if btn.is_displayed():
                ac_click(driver, btn)
                print(f"  로그인 버튼 클릭: id={btn.get_attribute('id')!r}")
                clicked = True
                break
        except Exception:
            continue
    if not clicked:
        pw_el.send_keys(Keys.RETURN)
        print("  Enter로 로그인 시도...")

    # ③ 시스템 메시지 팝업 → "계속하기" 클릭
    print("  시스템 팝업 대기 (15초)...")
    time.sleep(15)
    for xpath in [
        "//div[text()='계속하기']",
        "//div[contains(text(),'계속하기')]",
        "//div[contains(text(),'계속')]",
        "//div[contains(text(),'확인')]",
    ]:
        try:
            btn = driver.find_element(By.XPATH, xpath)
            if btn.is_displayed():
                print(f"  팝업 버튼 클릭: '{btn.text.strip()}'")
                ac_click(driver, btn)
                time.sleep(3)
                break
        except Exception:
            continue

    # ④ 메인화면 완전 로딩 대기
    print("  메인화면 대기 (10초)...")
    time.sleep(10)


# ── 드롭다운 항목 목록 수집 ───────────────────────────────────────────────────

def collect_dropdown(driver, idx) -> list[str]:
    """드롭다운 idx 열어 항목 텍스트 목록 반환, 닫은 뒤 리턴"""
    open_dropdown(driver, idx, wait=2)
    time.sleep(1.5)

    xpaths = [
        "//div[contains(@class,'ListBoxCell') and string-length(normalize-space(.))>0]",
        "//div[contains(@class,'listbox')]//div[string-length(normalize-space(.))>0 and not(.//div)]",
        "//div[contains(@class,'List')]//div[string-length(normalize-space(.))>0 and not(.//div)]",
    ]
    for xpath in xpaths:
        els = driver.find_elements(By.XPATH, xpath)
        texts = list(dict.fromkeys(e.text.strip() for e in els if e.text.strip()))
        if texts:
            driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.ESCAPE)
            time.sleep(0.8)
            return [t for t in texts if t != '전체']

    driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.ESCAPE)
    time.sleep(0.8)
    return []


# ── 강의 목록 조회 ────────────────────────────────────────────────────────────

_course_cols_printed = False

def query_courses(driver) -> list[dict]:
    """현재 선택 상태로 조회 클릭 → SSV 파싱된 강의 레코드 반환"""
    global _course_cols_printed
    flush_xhr(driver)
    ac_click(driver, driver.find_element(By.XPATH, "//div[text()='조회']"))
    time.sleep(8)

    for item in read_xhr(driver, url_substr='selectMain'):
        records = parse_ssv(item.get('resp', ''))
        if records:
            if not _course_cols_printed:
                print(f"  [SSV 전체 컬럼] {list(records[0].keys())}")
                print(f"  [첫 레코드 전체] {records[0]}")
                _course_cols_printed = True
            return records
    return []


# ── 그리드 행 더블클릭 (키보드 DOWN 방향키 내비게이션) ─────────────────────

_last_dbl_row = -1
_nx_grid_methods_printed = False


def _find_best_anchor(driver) -> tuple[int, object]:
    """
    팝업 닫힌 후 DOM에서 현재 렌더링된 가장 높은 gridrow를 앵커로 반환.
    팝업 닫히면 그리드가 어디로 리셋됐는지 동적으로 감지.
    """
    cells = driver.find_elements(By.XPATH,
        "//div[contains(@id,'.cell_') and contains(@id,'_4:text')]")
    best_row = 0
    best_el = None
    for cell in cells:
        eid = cell.get_attribute('id') or ''
        m = re.search(r'gridrow_(\d+)\.cell_', eid)
        if m:
            r = int(m.group(1))
            if r > best_row:
                best_row, best_el = r, cell
    # row 0 폴백 (항상 DOM에 있음)
    if best_el is None:
        els = driver.find_elements(By.XPATH,
            "//div[contains(@id,'gridrow_0.cell_0_4')]")
        if els:
            best_el = els[0]
    return best_row, best_el


_STAGE = 5   # 팝업 없이 한 번에 이동 가능한 최대 행 수 (실험적으로 5~6 작동)


def _nav_steps(driver, el, n: int):
    """el에 포커스 후 DOWN 키 n번 → 그리드 스크롤"""
    ac_click(driver, el)
    time.sleep(0.2)
    active = driver.switch_to.active_element
    for _ in range(n):
        active.send_keys(Keys.DOWN)
        time.sleep(0.04)
    time.sleep(0.35)


def _nexacro_grid_scroll(driver, row_idx: int) -> str:
    """
    단계적(Staged) 내비게이션:
    팝업 후 항상 anchor≈5 이므로, 6행씩 이동 후 앵커 재스캔을 반복.
    마지막 단계에서 target row에 도달.
    """
    anchor, anchor_el = _find_best_anchor(driver)
    if anchor_el is None:
        return 'no_anchor'

    stage_count = 0
    while row_idx - anchor > _STAGE:
        # 중간 지점까지 이동 (팝업 열지 않음)
        _nav_steps(driver, anchor_el, _STAGE)
        stage_count += 1
        new_anchor, new_el = _find_best_anchor(driver)
        if new_el is None or new_anchor <= anchor:
            return f'stage_stuck:anchor={anchor},stage={stage_count}'
        anchor, anchor_el = new_anchor, new_el

    # 최종 이동
    final_steps = row_idx - anchor
    if final_steps > 0:
        _nav_steps(driver, anchor_el, final_steps)

    target_suffix = f'gridrow_{row_idx}.cell_{row_idx}_4'
    found = bool(driver.find_elements(By.XPATH,
        f"//div[contains(@id,'{target_suffix}')]"))
    return f'{"OK" if found else "NOT_FOUND"}:anchor={anchor},stages={stage_count},final={final_steps}'




def dbl_click_course(driver, course_name: str, row_idx: int) -> bool:
    """
    행 더블클릭 → 팝업 열기.

    rows 0-6: 텍스트 div 직접 더블클릭
    rows 7+:  single-click(Nexacro 선택 설정) → 0.8s 대기 → double-click(팝업 열기)
              단, 화면에 없으면 스테이지 내비게이션 후 재시도
    """
    global _last_dbl_row, _nx_grid_methods_printed

    safe = course_name.replace("'", "\\'")

    def find_el(wait_sec: float):
        for xpath in [
            f"//div[normalize-space(text())='{safe}']",
            f"//div[contains(text(),'{safe[:12]}')]",
        ]:
            try:
                return WebDriverWait(driver, wait_sec).until(
                    EC.presence_of_element_located((By.XPATH, xpath))
                )
            except TimeoutException:
                continue
        return None

    if row_idx <= 6:
        el = find_el(5)
        if el:
            time.sleep(0.2)
            ActionChains(driver).move_to_element(el).double_click().perform()
            _last_dbl_row = row_idx
            return True
        return False

    # rows 7+: single-click 먼저 → Nexacro 선택 설정 → double-click
    el = find_el(2)
    if el is None:
        # 화면에 없으면 스테이지 내비게이션
        result = _nexacro_grid_scroll(driver, row_idx)
        if not _nx_grid_methods_printed or any(k in result for k in ('NOT_FOUND', 'stage_stuck', 'no_')):
            print(f"    [NxGrid] row={row_idx} → {result}")
        _nx_grid_methods_printed = True
        el = find_el(2)
        if el is None:
            return False

    # single-click → Nexacro 선택 업데이트
    try:
        ActionChains(driver).move_to_element(el).click().perform()
    except Exception:
        driver.execute_script("arguments[0].click();", el)
    time.sleep(0.8)  # Nexacro 선택 처리 대기
    # double-click → 팝업 열기
    try:
        ActionChains(driver).move_to_element(el).double_click().perform()
    except Exception:
        driver.execute_script(
            "arguments[0].dispatchEvent(new MouseEvent('dblclick',{bubbles:true}));", el)
    _last_dbl_row = row_idx
    return True


# ── 팝업 수강인원 추출 ────────────────────────────────────────────────────────

_enroll_debug = 3  # 처음 N개 과목은 XHR URL 출력


def enrollment_from_popup(driver, expected_course: str = '', gaesul_year: str = '', gaesul_term: str = '') -> tuple[int, int]:
    """
    XHR 큐에서 SUGANG_CNT, GAESUL_CNT 추출.
    gaesul_year/gaesul_term 지정 시 해당 학기 행만 사용.
    expected_course 지정 시 팝업 과목명(GYOGWAMOK_KOR_NM)과 비교해 불일치 시 (0,0) 반환.
    """
    global _enroll_debug
    items = driver.execute_script("return window._xhrQ || [];")

    if _enroll_debug > 0:
        urls = [x.get('url', '').split('gaia/')[-1] for x in items]
        print(f"         XHR: {urls}")
        _enroll_debug -= 1

    # ── 과목명 검증 (select01.do에서 GYOGWAMOK_KOR_NM 확인) ──────────────────
    if expected_course:
        for item in items:
            url = item.get('url') or ''
            if 'NHSSU020262P' in url and 'select01' in url:
                records = parse_ssv(item.get('resp', ''))
                if records:
                    popup_nm = records[0].get('GYOGWAMOK_KOR_NM', '').strip()
                    if popup_nm and expected_course not in popup_nm and popup_nm not in expected_course:
                        print(f"         [불일치] 예상={expected_course!r}, 팝업={popup_nm!r} → 0 처리")
                        return 0, 0

    # ── NHSSU020262P/select02.do → SUGANG_CNT, GAESUL_CNT ───────────────────
    for item in items:
        url = item.get('url') or ''
        if 'NHSSU020262P' in url and 'select02' in url:
            records = parse_ssv(item.get('resp', ''))
            if not records:
                continue
            # 학기 지정 시 해당 행 우선, 없으면 첫 행
            target = records[0]
            if gaesul_year and gaesul_term:
                for rec in records:
                    if rec.get('GAESUL_YEAR') == gaesul_year and rec.get('GAESUL_TERM') == gaesul_term:
                        target = rec
                        break
            sugang = target.get('SUGANG_CNT', '').strip()
            gaesul = target.get('GAESUL_CNT', '').strip()
            return (
                int(sugang) if sugang.isdigit() else 0,
                int(gaesul) if gaesul.isdigit() else 0,
            )

    # 폴백: 모든 SSV에서 SUGANG_CNT
    for item in items:
        records = parse_ssv(item.get('resp', ''))
        if not records:
            continue
        for col in ('SUGANG_CNT', 'SUHGANG_CNT', 'SUGANG_INW'):
            val = records[0].get(col, '').strip()
            if val.isdigit():
                return int(val), 0

    return 0, 0


# ── 팝업 닫기 ─────────────────────────────────────────────────────────────────

def close_popup(driver):
    """팝업 닫기: 닫기 버튼 → Escape 순서로 시도."""
    for xpath in [
        "//div[text()='닫기']",
        "//div[text()='X']",
        "//div[contains(@class,'close')]",
    ]:
        try:
            btn = driver.find_element(By.XPATH, xpath)
            if btn.is_displayed():
                ac_click(driver, btn)
                time.sleep(0.8)
                return
        except Exception:
            pass
    driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.ESCAPE)
    time.sleep(0.8)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def crawl():
    if not SKKU_ID or not SKKU_PW:
        print("[ERROR] SKKU_ID / SKKU_PW 환경변수를 설정하세요.")
        return

    opts = Options()
    opts.add_argument('--no-sandbox')
    opts.add_argument('--disable-dev-shm-usage')
    opts.add_argument('--disable-gpu')
    opts.add_argument('--start-maximized')
    opts.set_capability('goog:loggingPrefs', {'performance': 'ALL'})

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=opts
    )
    _COLS = ['학년도학기', '교과목명', '수업시간대', '수업요일및강의실', '수업형태', 'building', '수강인원', '개설강좌수']
    _SEMESTER_MAP = {
        '2024학년도 1학기': ('2024-1', '2024_1'),
        '2024학년도 2학기': ('2024-2', '2024_2'),
        '2025학년도 1학기': ('2025-1', '2025_1'),
        '2025학년도 2학기': ('2025-2', '2025_2'),
    }
    _CRAWL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'crawl')

    def _save_semester(rows, semester, dept):
        folder, prefix = _SEMESTER_MAP[semester]
        save_dir = os.path.join(_CRAWL_DIR, folder)
        os.makedirs(save_dir, exist_ok=True)
        df_s = pd.DataFrame(rows)[_COLS]
        path = os.path.join(save_dir, f'{prefix}_data_{dept}.csv')
        df_s.to_csv(path, index=False, encoding='utf-8-sig')
        print(f'  → 저장: {path}  ({len(df_s)}행)')

    all_rows = []

    try:
        # 1. 로그인
        login(driver)

        # 2. XHR 인터셉터 주입 (로그인 후 SPA 환경에서 유지됨)
        driver.execute_script(_XHR_INIT)

        # 3. 메뉴 진입: 수업영역 → 학사-전공과목
        print("[메뉴] 수업영역 클릭...")
        time.sleep(5)  # 메인화면 추가 대기 (Nexacro 메뉴 렌더링)
        area_el = None
        for xpath in [
            "//div[contains(text(),'수업영역')]",
            "//div[normalize-space(text())='수업영역']",
            "//div[contains(@id,'수업') and contains(text(),'영역')]",
            "//*[contains(text(),'수업영역')]",
        ]:
            try:
                area_el = WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.XPATH, xpath))
                )
                print(f"  수업영역 발견: xpath={xpath!r}, id={area_el.get_attribute('id')!r}")
                break
            except Exception:
                continue
        if area_el is None:
            # 진단: 현재 렌더링된 div 텍스트 목록 출력
            divs = driver.find_elements(By.XPATH, "//div[string-length(normalize-space(text()))>1 and string-length(normalize-space(text()))<20]")
            texts = list(dict.fromkeys(d.text.strip() for d in divs if d.text.strip()))[:30]
            print(f"  [DIAG] 현재 DOM 텍스트(상위30): {texts}")
            raise RuntimeError("수업영역 메뉴를 찾지 못했습니다. 위 DIAG 목록을 확인하세요.")
        ac_click(driver, area_el)
        time.sleep(3)

        print("[메뉴] 학사-전공과목 클릭...")
        menu_el = None
        for xpath in [
            "//div[contains(text(),'학사-전공과목')]",
            "//div[normalize-space(text())='학사-전공과목']",
            "//*[contains(text(),'학사-전공과목')]",
            "//div[contains(text(),'전공과목')]",
        ]:
            try:
                menu_el = WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.XPATH, xpath))
                )
                print(f"  학사-전공과목 발견: id={menu_el.get_attribute('id')!r}")
                break
            except Exception:
                continue
        if menu_el is None:
            divs = driver.find_elements(By.XPATH, "//div[string-length(normalize-space(text()))>1 and string-length(normalize-space(text()))<30]")
            texts = list(dict.fromkeys(d.text.strip() for d in divs if d.text.strip()))[:30]
            print(f"  [DIAG] 현재 DOM 텍스트(상위30): {texts}")
            raise RuntimeError("학사-전공과목 메뉴를 찾지 못했습니다.")
        ac_click(driver, menu_el)
        time.sleep(12)

        TARGET_COLLEGES = [
            '공과대학',
            '생명공학대학',
            '소프트웨어융합대학',
            '약학대학',
            '의과대학',
            '자연과학대학',
            '정보통신대학',
        ]
        TARGET_SEMESTERS = [
            '2024학년도 1학기',
            '2024학년도 2학기',
            '2025학년도 1학기',
            '2025학년도 2학기',
        ]

        # ── 단과대학별 학과 목록 수집 ────────────────────────────────────────
        print("\n[학과 목록 수집] 단과대학별 학과 목록 수집 중...")
        college_depts: dict[str, list[str]] = {}

        try:
            camp_el = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//div[text()='자연과학']"))
            )
            ac_click(driver, camp_el)
            time.sleep(2)
        except Exception:
            print("  [WARN] 자연과학 캠퍼스 선택 실패 (목록 수집)")

        for college in TARGET_COLLEGES:
            try:
                open_dropdown(driver, idx=4, wait=2)
                click_option(driver, college, timeout=10)
                time.sleep(1.5)
                depts = [d for d in collect_dropdown(driver, idx=5) if d not in ('선택', '전체', '')]
                college_depts[college] = depts
                print(f"  {college}: {len(depts)}개 → {depts}")
            except Exception as e:
                print(f"  [ERROR] {college} 학과 목록 수집 실패: {e}")
                college_depts[college] = []

        # ── 메인 크롤 루프: 단과대학 → 학과 → 학기 ──────────────────────────
        global _last_dbl_row, _nx_grid_methods_printed

        for college, depts in college_depts.items():
            for dept in depts:
                _time_field = ''
                for semester in TARGET_SEMESTERS:
                    # 이미 수집된 파일 스킵
                    folder, prefix = _SEMESTER_MAP[semester]
                    out_path = os.path.join(_CRAWL_DIR, folder, f'{prefix}_data_{dept}.csv')
                    if os.path.exists(out_path):
                        print(f"  [SKIP] {dept} {semester} — 이미 존재")
                        continue

                    print(f"\n{'='*60}")
                    print(f"[{college}] {dept} / {semester}")

                    try:
                        # 캠퍼스
                        try:
                            camp_el = WebDriverWait(driver, 10).until(
                                EC.presence_of_element_located((By.XPATH, "//div[text()='자연과학']"))
                            )
                            ac_click(driver, camp_el)
                            time.sleep(2)
                        except Exception:
                            print("  [WARN] 자연과학 캠퍼스 선택 실패")

                        # 학기
                        open_dropdown(driver, idx=2, wait=2)
                        click_option(driver, semester, timeout=10)
                        time.sleep(3)

                        # 도전학기
                        open_dropdown(driver, idx=3, wait=2)
                        click_option(driver, '도전학기 제외', timeout=10)
                        time.sleep(2)

                        # 단과대학 (discovery 후 idx=5 필터 리셋을 위해 필요)
                        open_dropdown(driver, idx=4, wait=2)
                        click_option(driver, college, timeout=10)
                        time.sleep(4)  # 학과 목록 로딩 대기

                        # 학과
                        open_dropdown(driver, idx=5, wait=2)
                        click_option(driver, dept, timeout=15)
                        time.sleep(2)

                    except Exception as e:
                        print(f"  [ERROR] 드롭다운 선택 실패, 스킵: {e.__class__.__name__}")
                        try:
                            driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.ESCAPE)
                        except Exception:
                            pass
                        time.sleep(2)
                        continue

                    # 조회
                    course_records = query_courses(driver)
                    print(f"  강의 {len(course_records)}개")

                    if not course_records:
                        print(f"  [SKIP] 강의 없음")
                        continue

                    # 시간대 필드 감지 (학과별 첫 유효 학기에서만)
                    if not _time_field:
                        _time_field = _find_time_field(course_records[0])
                        if _time_field:
                            print(f"  [시간대 필드 감지] '{_time_field}'")
                        else:
                            print("  [WARN] 시간대 필드 자동 감지 실패")

                    # 그리드 내비게이션 상태 초기화 (학과/학기 전환 시)
                    _last_dbl_row = -1
                    _nx_grid_methods_printed = False

                    enroll_cache: dict[str, tuple[int, int]] = {}
                    sem_rows = []
                    row_idx = 0
                    for r in course_records:
                        course    = r.get('GWAMOK_NAME', '').strip()
                        haksu_no  = r.get('HAKSU_NO', '').strip()
                        room      = r.get('GYOSI_NAME', '').strip()
                        ctype     = r.get('HYUNGTAE', '').strip()
                        time_slot = r.get(_time_field, '').strip() if _time_field else ''
                        g_year    = r.get('GAESUL_YEAR', '').strip()
                        g_term    = r.get('GAESUL_TERM', '').strip()

                        if not course:
                            row_idx += 1
                            continue
                        if re.search(r'i[-\s]?campus', ctype, re.IGNORECASE):
                            row_idx += 1
                            continue

                        if haksu_no in enroll_cache:
                            enrollment, gaesul = enroll_cache[haksu_no]
                            print(f"    [{row_idx}] {course}: 수강인원={enrollment} (캐시)")
                        else:
                            print(f"    [{row_idx}] {course} 더블클릭 중...", flush=True)
                            try:
                                flush_xhr(driver)
                                clicked = dbl_click_course(driver, course, row_idx)
                                if clicked:
                                    time.sleep(2)
                                    enrollment, gaesul = enrollment_from_popup(driver, expected_course=course, gaesul_year=g_year, gaesul_term=g_term)
                                    print(f"         팝업 닫는 중...", flush=True)
                                    close_popup(driver)
                                    delay()
                                else:
                                    print(f"    [SKIP] row={row_idx}: {course}")
                                    enrollment, gaesul = 0, 0
                            except Exception as e:
                                print(f"    [ERROR] row={row_idx}: {course} — {e.__class__.__name__}")
                                try:
                                    close_popup(driver)
                                except Exception:
                                    pass
                                enrollment, gaesul = 0, 0
                            enroll_cache[haksu_no] = (enrollment, gaesul)
                            print(f"    [{row_idx}] {course}: 수강인원={enrollment}, 개설강좌수={gaesul}")
                        row = {
                            '학년도학기': semester,
                            '교과목명': course,
                            '수업시간대': time_slot,
                            '수업요일및강의실': room,
                            '수업형태': ctype,
                            'building': parse_building(room),
                            '수강인원': enrollment,
                            '개설강좌수': gaesul,
                        }
                        all_rows.append(row)
                        sem_rows.append(row)
                        row_idx += 1

                    # 학기별 즉시 저장 (중간 크래시 대비)
                    if sem_rows:
                        _save_semester(sem_rows, semester, dept)

    finally:
        driver.quit()

    if not all_rows:
        print("[ERROR] 수집된 데이터 없음")
        return

    print(f"\n전체 {len(all_rows)}개 강의 수집 완료 (학기별 crawl/ 폴더에 저장됨)")


if __name__ == '__main__':
    crawl()