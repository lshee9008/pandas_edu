# 쉼자리 MVP 구현설계

**대상지 서울 관악구 · 기간 4주 · Python + Streamlit**

이 문서는 [기획서](기획서_쉼자리.md)의 4~7장을 실제 코드로 옮기기 위한 기술 명세다.

---

## 1. 모듈 구조

```
shimjari/
├── data/
│   ├── raw/                     # 원본 (주소DB, 건물DB, 전자지도, 쉼터, DEM)
│   └── processed/               # 정제 산출물 (parquet/gpkg)
├── src/shimjari/
│   ├── ingest/
│   │   ├── address_db.py        # 상세주소DB 파싱 → 지하·옥탑 플래그
│   │   ├── building_db.py       # 건물DB → 후보지 필터 + 높이 추정
│   │   ├── shelters.py          # 쉼터 표준데이터 지오코딩
│   │   └── terrain.py           # DEM → 보행망 경사도
│   ├── shade/
│   │   ├── solar.py             # 태양 고도각·방위각 (pvlib / NREL SPA)
│   │   ├── shadow.py            # 건물 폴리곤 → 그림자 폴리곤 union
│   │   └── network.py           # 보행망 엣지 가중치 λ_e, μ_e 부여
│   ├── risk/
│   │   ├── score.py             # 규칙기반 호 단위 취약도 (MVP)
│   │   └── pu_learn.py          # PU learning (2단계, 스켈레톤만)
│   ├── optimize/
│   │   ├── reach.py             # HWD 최단경로 → 커버 행렬 a_ij
│   │   ├── mclp.py              # 커버리지 최대화 (효율)
│   │   ├── pcenter.py           # 최악 도달시간 최소화 (형평)
│   │   └── pareto.py            # 두 목적 사이 프론트 탐색
│   ├── copilot/
│   │   ├── schema.py            # 최적화 스펙 Pydantic 모델
│   │   ├── nl2opt.py            # Claude tool use → 스펙 생성
│   │   └── brief.py             # 결과 → 근거 설명 + 보고서 초안
│   ├── validate/
│   │   └── backtest.py          # 과거재현 검증
│   └── app/
│       ├── main.py              # Streamlit 진입점
│       └── views/               # 지도 / 시나리오 / 파레토 / 보고서
└── tests/
```

---

## 2. 데이터 스키마 (processed)

### 2-1. `demand.parquet` — 수요 단위 (호 집계)

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `demand_id` | str | 건물 단위 집계 키 (건물관리번호) |
| `road_addr` | str | 도로명주소 |
| `x`, `y` | float | EPSG:5179 |
| `n_households` | int | 상세주소DB 호 수 |
| `n_basement` | int | 지하·반지하 호 수 |
| `n_rooftop` | int | 옥탑 호 수 |
| `bld_year` | int | 사용승인연도 |
| `floors` | int | 지상 층수 |
| `has_elevator` | bool | 승강기 유무 |
| `slope_deg` | float | 접도 경사도 |
| `w` | float | **취약도 가중치** (§4) |

> 지하·옥탑 판별은 상세주소DB의 층 표기(`지하`, `B1`, `옥탑`, `R`)를 정규식으로 파싱. 표기 변이 목록은 `ingest/address_db.py` 상단에 상수로 관리한다.

### 2-2. `candidates.parquet` — 후보지

건물DB에서 아래 조건으로 자동 추출한다.

```python
CANDIDATE_FILTER = {
    "용도": ["제1종근린생활시설", "제2종근린생활시설", "공공업무시설",
             "노유자시설", "교육연구시설", "문화및집회시설"],
    "소유": "국·공유",          # 별도 재산대장 연계 시
    "연면적_min_m2": 60,
    "지상층수_min": 1,
}
```

| 컬럼 | 설명 |
|---|---|
| `cand_id` | 건물관리번호 |
| `x`, `y` | 좌표 |
| `floor_area` | 연면적 |
| `use_code` | 용도 |
| `cost` | **전환 추정비용** = 연면적 × 단가 + 냉방설비 고정비 |
| `is_existing` | 기존 쉼터 여부 (기준선 비교용) |

### 2-3. `walk_network.gpkg` — 보행망

| 엣지 속성 | 설명 |
|---|---|
| `length` | 구간 길이(m) |
| `shade_ratio` | 그늘 비율 s_e ∈ [0,1] — 시각별 컬럼 (`s_10h`, `s_14h`, `s_16h`) |
| `slope` | 경사도(°) |
| `is_indoor` | 지하상가·실내 연결 여부 |
| `lambda_e` | 열노출계수 |
| `mu_e` | 보행부담계수 |
| `w_hwd` | **HWD 가중치** = length × λ_e × μ_e |

---

## 3. SHADE-NET — 그림자 시뮬레이션

### 3-1. 태양 위치

```python
# solar.py
import pvlib, pandas as pd

def solar_position(lat, lon, when):
    """when: tz-aware datetime. 반환 (고도각 deg, 방위각 deg)"""
    sp = pvlib.solarposition.get_solarposition(
        pd.DatetimeIndex([when]), lat, lon
    )
    return float(sp["apparent_elevation"].iloc[0]), float(sp["azimuth"].iloc[0])
```

기준 시각은 **폭염 대책기간 대표일(7월 하순) 10시 / 14시 / 16시** 세 개를 산출한다. 14시가 기본값.

### 3-2. 그림자 폴리곤

```python
# shadow.py
import numpy as np
from shapely.affinity import translate
from shapely.ops import unary_union

FLOOR_HEIGHT_M = 3.0

def shadow_polygons(buildings, elev_deg, azim_deg):
    """buildings: GeoDataFrame(geometry=폴리곤, floors=지상층수)"""
    if elev_deg <= 3:                      # 저고도각은 그림자 발산 → 상한 처리
        elev_deg = 3
    h = buildings["floors"].clip(lower=1) * FLOOR_HEIGHT_M
    L = h / np.tan(np.radians(elev_deg))

    # 그림자는 태양 반대 방향. 방위각은 북=0, 시계방향.
    theta = np.radians(azim_deg + 180.0)
    dx, dy = L * np.sin(theta), L * np.cos(theta)

    # 원 폴리곤과 평행이동 폴리곤의 convex hull union ≈ 압출 그림자
    shadows = [
        unary_union([g, translate(g, xoff=x, yoff=y)]).convex_hull
        for g, x, y in zip(buildings.geometry, dx, dy)
    ]
    return unary_union(shadows)
```

> **근사임을 명시한다.** 가로수·구조물·차양은 반영되지 않는다. 검증 목표는 절대 정확도가 아니라 **직선거리 대비 개선폭**이다(기획서 §12).

### 3-3. 엣지 가중치

```python
KAPPA = 0.8      # 열노출 민감도
INDOOR_LAMBDA = 0.3
SLOPE_PENALTY = 0.06   # 경사 1° 당 부담 증가율

def edge_weights(edges, shadow_union):
    inter = edges.geometry.intersection(shadow_union).length
    s = (inter / edges.geometry.length).clip(0, 1)

    lam = np.where(edges["is_indoor"], INDOOR_LAMBDA, 1 + KAPPA * (1 - s))
    mu  = 1 + SLOPE_PENALTY * edges["slope"].clip(lower=0)
    return edges.geometry.length * lam * mu
```

**겨울 모드**는 `s`(그늘) 대신 `wind_exposure`(건물 배치 기반 풍하지대 여부)를 넣고 부호를 뒤집는다. 같은 함수, 다른 계수.

### 3-4. HWD

```python
# reach.py
import networkx as nx

def coverage_matrix(G, demands, candidates, threshold_min, walk_speed_mpm=60):
    """a_ij = 1 if HWD(i,j) <= threshold. 반환 dict[cand_id] -> set(demand_id)"""
    budget = threshold_min * walk_speed_mpm      # HWD는 '가중 미터'
    cover = {}
    for c in candidates.itertuples():
        dist = nx.single_source_dijkstra_path_length(
            G, source=c.node_id, cutoff=budget, weight="w_hwd"
        )
        cover[c.cand_id] = {
            d.demand_id for d in demands.itertuples() if dist.get(d.node_id, 1e18) <= budget
        }
    return cover
```

> 후보지 수가 수백 개이므로 후보 기준 역방향 다익스트라가 훨씬 싸다. `cutoff`로 조기 종료.

---

## 4. HOUSEHOLD-RISK — 취약도 가중치 (MVP: 규칙 기반)

```python
DEFAULT_WEIGHTS = {          # 전부 UI 슬라이더로 노출한다
    "basement":   2.0,       # 지하·반지하 호
    "rooftop":    1.8,       # 옥탑 호
    "old_bld":    1.3,       # 사용승인 30년 초과
    "no_elev":    1.2,       # 승강기 없음 & 3층 이상
    "steep":      1.2,       # 접도 경사 8° 초과
}

def vulnerability(df, w=DEFAULT_WEIGHTS):
    base = df["n_households"] - df["n_basement"] - df["n_rooftop"]
    score = (base
             + df["n_basement"] * w["basement"]
             + df["n_rooftop"]  * w["rooftop"])
    mult = np.ones(len(df))
    mult *= np.where(df["bld_year"] < YEAR - 30, w["old_bld"], 1)
    mult *= np.where((~df["has_elevator"]) & (df["floors"] >= 3), w["no_elev"], 1)
    mult *= np.where(df["slope_deg"] > 8, w["steep"], 1)
    return score * mult
```

**가중치를 감추지 않는 것이 설계 의도다.** 담당자가 값을 바꿔가며 결과 민감도를 볼 수 있어야 결재가 통과된다.

2단계 PU 러닝(`pu_learn.py`)은 119 신고 주소 확보 시 이 함수를 대체한다. 인터페이스(`df → w 시리즈`)는 동일하게 유지한다.

---

## 5. 최적화

### 5-1. MCLP — 효율 우선

```python
from ortools.sat.python import cp_model

def solve_mclp(demands, candidates, cover, budget,
               min_dong_coverage=None, forbidden=()):
    m = cp_model.CpModel()
    x = {c: m.NewBoolVar(f"x_{c}") for c in candidates.cand_id}
    y = {d: m.NewBoolVar(f"y_{d}") for d in demands.demand_id}

    covered_by = defaultdict(list)
    for c, ds in cover.items():
        for d in ds:
            covered_by[d].append(x[c])

    for d, xs in covered_by.items():
        m.Add(y[d] <= sum(xs))
    for d in demands.demand_id:                    # 아무도 못 덮는 수요
        if d not in covered_by:
            m.Add(y[d] == 0)

    m.Add(sum(int(c.cost) * x[c.cand_id] for c in candidates.itertuples()) <= int(budget))
    for c in forbidden:
        m.Add(x[c] == 0)

    if min_dong_coverage:                          # 형평 제약
        for dong, alpha in min_dong_coverage.items():
            sub = demands[demands.dong == dong]
            tot = int(sub.w.sum() * SCALE)
            m.Add(sum(int(r.w * SCALE) * y[r.demand_id] for r in sub.itertuples())
                  >= int(alpha * tot))

    m.Maximize(sum(int(r.w * SCALE) * y[r.demand_id] for r in demands.itertuples()))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    solver.parameters.num_search_workers = 8
    return solver, x, y
```

`SCALE`은 CP-SAT가 정수만 다루므로 가중치를 100배 정수화하는 상수.

### 5-2. p-center — 형평 우선

이진탐색으로 임계 HWD `T`를 줄여가며 "예산 내에서 모든 수요를 T 안에 넣을 수 있는가"를 만족성 문제로 푼다. 가능한 최소 `T*`가 해.

### 5-3. 파레토 프론트

`min_dong_coverage`의 α를 0.0 → 0.7까지 스윕하며 (총커버, 최저동커버율) 쌍을 수집. 슬라이더로 탐색한다.

### 5-4. 한계효용 곡선 & 체감점

k = 1..K에 대해 예산 대신 개수 제약으로 풀어 커버 증가분을 얻고, **Kneedle 알고리즘**(또는 2차 차분 최대점)으로 knee를 잡는다.

---

## 6. NL2OPT — 정책문장 → 최적화 스펙

### 6-1. 스펙 스키마

```python
# schema.py
from pydantic import BaseModel, Field
from typing import Literal

class OptimizationSpec(BaseModel):
    objective: Literal["max_coverage", "minimax_time", "min_cost"]
    budget_krw: int | None = None
    max_facilities: int | None = None
    reach_threshold_min: float = 10.0
    time_of_day: Literal["10h", "14h", "16h"] = "14h"
    season: Literal["summer", "winter"] = "summer"
    weight_overrides: dict[str, float] = Field(default_factory=dict)
    min_dong_coverage: float | None = None
    exclude_candidates: list[str] = Field(default_factory=list)
    candidate_filters: dict = Field(default_factory=dict)
    rationale: str          # LLM이 각 항목을 왜 그렇게 뒀는지 한국어 설명
```

### 6-2. Claude 호출 — 도구 사용으로 구조 강제

```python
# nl2opt.py
import anthropic

MODEL = "claude-opus-5"

SYSTEM = """너는 지자체 쉼터 입지 최적화 시스템의 '스펙 컴파일러'다.
담당 공무원의 한국어 문장을 OptimizationSpec으로 변환하는 일만 한다.

절대 규칙:
- 최적해나 추천 건물을 제시하지 마라. 해는 솔버가 낸다.
- 문장에 없는 수치를 지어내지 마라. 불명확하면 해당 필드를 null로 두고
  rationale에 '담당자 확인 필요'로 적어라.
- rationale에는 각 필드를 그렇게 설정한 근거를 담당자가 검수할 수 있게 한국어로 써라.
"""

def compile_spec(user_text: str, context: dict) -> OptimizationSpec:
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM,
        tools=[{
            "name": "emit_spec",
            "description": "최적화 스펙을 확정한다",
            "input_schema": OptimizationSpec.model_json_schema(),
        }],
        tool_choice={"type": "tool", "name": "emit_spec"},
        messages=[{"role": "user", "content":
                   f"[가용 컨텍스트]\n{context}\n\n[담당자 입력]\n{user_text}"}],
    )
    block = next(b for b in resp.content if b.type == "tool_use")
    return OptimizationSpec(**block.input)
```

### 6-3. 승인 게이트 (필수)

```
compile_spec() → 화면에 수식 렌더링 → [담당자 승인] 버튼 → solve()
```

**승인 없이는 솔버가 호출되지 않는다.** 이 게이트가 기획서 §4-③의 "AI 책임 분리 원칙"을 코드로 강제하는 지점이다. `spec_hash`, 승인자, 타임스탬프, 솔버 버전을 `runs.jsonl`에 append-only로 기록한다.

**폐쇄망 폴백** — LLM 미사용 시 동일한 `OptimizationSpec`을 Streamlit 폼으로 직접 입력한다. NL2OPT가 없어도 전 기능이 동작해야 한다.

---

## 7. BRIEF — 근거 설명과 보고서

- **선정 기여도** — 선택된 각 후보에 대해 `신규커버세대`, `그중 지하·옥탑`, `기존쉼터 공백도`, `비용효율`, `경사 완화` 5개 항목의 정규화 기여율을 계산해 막대로 표시. (해석 가능한 선형 분해 — 별도 ML 모델 불필요)
- **탈락 사유** — 위반한 제약식을 그대로 인용: `"예산 초과 (전환비 1.2억, 잔여 예산 0.4억)"`, `"승강기 없음 ∧ 4층 → 후보 제외 규칙"`
- **보고서 초안** — 결과 JSON + 지도 PNG를 넣어 Claude로 마크다운 생성 → HWP 붙여넣기용. **수치는 반드시 솔버 출력만 인용하도록 프롬프트에서 강제**하고, 생성문 내 모든 숫자가 원본 JSON에 존재하는지 후검증한다.

```python
def verify_numbers(text: str, allowed: set[str]) -> list[str]:
    """생성문에서 원본에 없는 수치를 찾아낸다. 발견 시 재생성."""
    found = set(re.findall(r"[\d,]+\.?\d*", text))
    return [n for n in found if n.replace(",", "") not in allowed]
```

---

## 8. 백테스트

```python
# backtest.py
def run(cutoff_year=2020, horizon=(2021, 2025)):
    hist = load_state_as_of(cutoff_year)          # 그 시점 쉼터·건물·주소
    n = count_added_shelters(*horizon)            # 실제로 늘린 개수
    ai = solve_mclp(hist.demands, hist.candidates,
                    hist.cover, budget=None, max_facilities=n)
    actual = actual_added_shelters(*horizon)

    return {
        "ai_covered":     evaluate(ai, hist.demands),
        "actual_covered": evaluate(actual, hist.demands),
        "delta_pct": ...,
    }
```

**주의** — 실제 지자체 결정에는 소유권·협의·민원 등 모델이 모르는 제약이 있다. 결과는 *"AI가 더 낫다"* 가 아니라 *"주소 기반 수요 재계산 시 커버 가능했던 상한"* 으로 서술한다. 이 문구를 화면에도 그대로 띄운다.

---

## 9. Streamlit 화면

| 탭 | 내용 |
|---|---|
| **① 현황** | 관악구 사각지대 지도. **직선거리 vs HWD 좌우 비교** + 시각 슬라이더(10/14/16시) |
| **② 질의** | 자연어 입력 → 컴파일된 수식 표시 → 승인 → 해 |
| **③ 시뮬레이션** | 지도 클릭 → 반사실 즉시 재계산. 한계효용 곡선 + 체감점 표시 |
| **④ 대안 비교** | 효율안 / 형평안 / 파레토 슬라이더 |
| **⑤ 보고서** | 근거 분해 + 초안 다운로드 + 실행 로그 |

> Streamlit 앱 작성 시 `developing-with-streamlit` 스킬을 먼저 로드할 것.

---

## 10. 4주 체크리스트

**1주 — 데이터**
- [ ] 관악구 상세주소DB 파싱, 지하·옥탑 플래그 (정규식 커버리지 검수)
- [ ] 건물DB 정합 + 후보지 필터 → `candidates.parquet`
- [ ] 쉼터 표준데이터 지오코딩 (실패 건 목록화 → 품질 회신 자료)
- [ ] DEM → 보행망 경사도

**2주 — SHADE-NET**
- [ ] 태양 위치 계산 검증 (기상청 일출·일몰과 대조)
- [ ] 그림자 폴리곤 생성, 시각별 `shade_ratio`
- [ ] HWD 다익스트라 + 커버 행렬
- [ ] **좌우 비교 지도** (직선거리 vs HWD) — 첫 데모 산출물

**3주 — 최적화 + Copilot**
- [ ] MCLP / p-center / 파레토
- [ ] 한계효용 곡선 + knee 탐지
- [ ] NL2OPT + 승인 게이트 + 실행 로그
- [ ] 정책 문장 50건 테스트셋 → 변환 정확도 측정

**4주 — 통합**
- [ ] Streamlit 5개 탭
- [ ] 반사실 시뮬레이터 (응답 1초 이내 — 커버 행렬 사전 계산)
- [ ] 백테스트
- [ ] 보고서 생성 + 수치 후검증
- [ ] 시연 영상 3분

---

## 11. 성능 목표

| 항목 | 목표 |
|---|---|
| 커버 행렬 사전 계산 (관악구 전역) | < 5분 (배치) |
| 반사실 1회 재계산 | **< 1초** |
| MCLP 해 (후보 300 × 수요 20,000) | < 30초 |
| NL2OPT 컴파일 | < 10초 |
| 앱 콜드스타트 | < 15초 |

반사실 응답속도가 체감 품질을 좌우한다. 커버 집합을 `dict[cand_id] -> set[demand_id]`로 메모리에 상주시키고 합집합 연산만 수행한다.
