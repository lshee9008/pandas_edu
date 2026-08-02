"""
할리스 매장 상권 분석 대시보드
종합.ipynb 를 Streamlit 앱으로 옮긴 것.

실행:
    uv run streamlit run 종합.py

노트북과 달라진 점
    - 1~2단계(크롤링 / 좌표변환)는 외부 서버를 호출하므로 자동 실행하지 않는다.
      사이드바의 '데이터 수집' 메뉴에서 버튼을 눌러야 동작한다.
    - 3단계 이후는 source/ 에 저장된 CSV를 읽어 바로 보여준다.
    - DBSCAN 조건과 지도 종류는 화면에서 바꿔가며 볼 수 있다.
"""

import json
import os
import platform
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ---------------------------------------------------------------------------
# 경로 상수
# ---------------------------------------------------------------------------
SOURCE_DIR = "source"
OUTPUT_DIR = "output"

STORE_RAW = f"{SOURCE_DIR}/hollys_store.csv"                    # 크롤링 원본
STORE_GEO = f"{SOURCE_DIR}/hollys_store_geo_kakao_final.csv"    # 좌표 포함
POPULATION = f"{SOURCE_DIR}/population_sido.csv"                # 시도별 인구
GEOJSON = f"{SOURCE_DIR}/korea_sido.geojson"                    # 시도 경계

BASE_URL = "https://www.hollys.co.kr/store/korea/korStore2.do"

# GeoJSON 이 2018년 기준이라 개편 전 지명을 쓴다. 현재 지명으로 맞춘다.
NAME_FIX = {"강원도": "강원특별자치도", "전라북도": "전북특별자치도"}


# ---------------------------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------------------------
st.set_page_config(page_title="할리스 상권 분석", page_icon="☕", layout="wide")


def set_korean_font():
    """OS별 한글 폰트 설정. 없으면 그래프의 한글이 네모로 깨진다."""
    system = platform.system()
    if system == "Windows":
        family = "Malgun Gothic"
    elif system == "Darwin":
        family = "AppleGothic"
    else:
        family = "NanumBarunGothic"

    plt.rcParams["font.family"] = family
    plt.rcParams["axes.unicode_minus"] = False   # 음수 기호 깨짐 방지


set_korean_font()


# ---------------------------------------------------------------------------
# 데이터 로딩 (cache_data 를 붙이면 파일을 한 번만 읽는다)
# ---------------------------------------------------------------------------
@st.cache_data
def load_stores() -> pd.DataFrame:
    """좌표가 붙은 매장 데이터."""
    return pd.read_csv(STORE_GEO)


@st.cache_data
def load_population() -> pd.DataFrame:
    """시도별 인구."""
    return pd.read_csv(POPULATION)


@st.cache_data
def build_merged() -> pd.DataFrame:
    """매장 수 + 인구 → 10만명당 매장 수. (노트북 3·5단계)"""
    store_count = load_stores()["시도"].value_counts().reset_index()
    store_count.columns = ["시도", "매장수"]

    merged = store_count.merge(load_population(), on="시도", how="inner")
    merged["10만명당_매장수"] = merged["매장수"] / merged["인구"] * 100000
    return merged.sort_values("10만명당_매장수", ascending=False).reset_index(drop=True)


@st.cache_data
def load_geojson(step: int = 3) -> dict:
    """
    시도 경계 GeoJSON.

    원본이 7.2MB라 그대로 쓰면 브라우저 전송이 느리다.
    좌표를 step 간격으로 솎고 소수점을 4자리로 줄여 1MB 대로 낮춘다.
    시도 단위 지도에서는 모양 차이가 눈에 띄지 않는다.
    """
    def thin(ring):
        pts = ring if len(ring) <= 12 else ring[::step] + [ring[-1]]
        return [[round(float(x), 4), round(float(y), 4)] for x, y in pts]

    def simplify(geom):
        coords = geom["coordinates"]
        if geom["type"] == "Polygon":
            geom["coordinates"] = [thin(r) for r in coords]
        elif geom["type"] == "MultiPolygon":
            geom["coordinates"] = [[thin(r) for r in poly] for poly in coords]
        return geom

    with open(GEOJSON, encoding="utf-8") as f:
        geo = json.load(f)

    for feat in geo["features"]:
        feat["properties"]["name"] = NAME_FIX.get(
            feat["properties"]["name"], feat["properties"]["name"]
        )
        feat["geometry"] = simplify(feat["geometry"])

    return geo


def missing_files() -> list:
    """없는 데이터 파일 목록."""
    return [p for p in (STORE_GEO, POPULATION, GEOJSON) if not os.path.exists(p)]


def download_button(df: pd.DataFrame, filename: str, label: str):
    st.download_button(
        label,
        df.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# 1. 데이터 개요
# ---------------------------------------------------------------------------
def page_overview():
    st.header("데이터 개요")
    st.caption("할리스 공식 홈페이지 매장검색에서 수집한 매장 정보에 카카오 API로 좌표를 붙인 데이터")

    df = load_stores()
    ok = df["위도"].notnull()

    # 라벨은 짧게 유지한다. 4칸으로 나누면 긴 라벨이 잘린다.
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("전체 매장", f"{len(df):,}개")
    c2.metric("좌표 있음", f"{ok.sum():,}개")
    c3.metric("좌표 없음", f"{(~ok).sum()}개")
    c4.metric("시도 수", f"{df['시도'].nunique()}개")
    st.caption(f"좌표 변환 성공률 {ok.mean() * 100:.1f}%")

    if "검색방식" in df.columns:
        st.subheader("검색 방식별 건수")
        st.caption("주소검색이 실패하면 매장명으로 키워드검색을 시도한다 (노트북 2단계)")
        method = df["검색방식"].value_counts().reset_index()
        method.columns = ["검색방식", "건수"]
        st.dataframe(method, use_container_width=True, hide_index=True)

    st.subheader("원본 데이터")
    sido_pick = st.multiselect(
        "시도 필터", sorted(df["시도"].dropna().unique()), placeholder="전체 보기"
    )
    view = df[df["시도"].isin(sido_pick)] if sido_pick else df
    st.dataframe(view, use_container_width=True, height=380)
    st.caption(f"{len(view):,}행")

    if (~ok).any():
        with st.expander(f"좌표 변환에 실패한 매장 {(~ok).sum()}곳"):
            cols = [c for c in ("매장명", "주소", "주소_전처리") if c in df.columns]
            st.dataframe(df.loc[~ok, cols], use_container_width=True, hide_index=True)
            st.caption("건물 내부·휴게소·캠퍼스 안 매장은 주소만으로 좌표를 찾기 어렵다.")


# ---------------------------------------------------------------------------
# 2. 시도별 매장 수
# ---------------------------------------------------------------------------
def page_store_count():
    st.header("시도별 매장 수")
    st.caption("노트북 3단계 — value_counts() 로 집계")

    counted = build_merged().sort_values("매장수", ascending=False)

    left, right = st.columns([1, 1.4])

    with left:
        st.dataframe(
            counted[["시도", "매장수"]],
            use_container_width=True, hide_index=True, height=520,
        )
        download_button(counted[["시도", "매장수"]], "store_count.csv", "CSV 내려받기")

    with right:
        fig, ax = plt.subplots(figsize=(7, 8))
        sns.barplot(data=counted, y="시도", x="매장수", color="#c0392b", ax=ax)
        ax.set_title("시도별 할리스 매장 수")
        ax.set_xlabel("매장 수")
        ax.set_ylabel("")
        for p in ax.patches:
            ax.text(p.get_width() + 1, p.get_y() + p.get_height() / 2,
                    f"{int(p.get_width())}", va="center", fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    total = counted["매장수"].sum()
    metro = counted[counted["시도"].isin(["서울특별시", "경기도", "인천광역시"])]["매장수"].sum()
    # 굵게 표시한 뒤에 한글 조사가 바로 붙으면 마크다운이 적용되지 않는다. 어순을 바꿨다.
    st.info(
        f"수도권(서울·경기·인천)에 {metro}개 — 전체 {total}개 가운데 "
        f"**{metro / total * 100:.1f}%** 를 차지한다."
    )


# ---------------------------------------------------------------------------
# 3. 인구 대비 매장 밀도
# ---------------------------------------------------------------------------
def page_density():
    st.header("인구 대비 매장 밀도")
    st.caption("노트북 5·6단계 — 매장 수 ÷ 인구 × 100,000")

    merged = build_merged()

    report = merged.copy()
    report["인구(만명)"] = report["인구"] / 10000
    report = report[["시도", "매장수", "인구(만명)", "10만명당_매장수"]].round(2)

    # 시도명은 label 에, 숫자는 value 에 둔다.
    # 반대로 하면 '전북특별자치도' 같은 긴 이름이 큰 글씨에서 잘린다.
    # delta 자리는 쓰지 않는다. 증감이 아닌데 화살표가 붙어 오해를 부른다.
    top = merged.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric(f"1위 · {top['시도']} (매장 {top['매장수']}개)",
              f"{top['10만명당_매장수']:.2f}개")

    for col, name in ((c2, "서울특별시"), (c3, "경기도")):
        row = merged[merged["시도"] == name]
        if not row.empty:
            rank = int(row.index[0]) + 1
            col.metric(f"{rank}위 · {name} (매장 {int(row.iloc[0]['매장수'])}개)",
                       f"{row.iloc[0]['10만명당_매장수']:.2f}개")

    st.subheader("순위표")
    st.dataframe(report, use_container_width=True, hide_index=True, height=460)
    download_button(report, "hollys_report.csv", "순위표 CSV 내려받기")

    st.subheader("해석")
    st.markdown(
        f"""
매장 **수**가 아니라 인구 대비 **밀도**로 보면 순위가 뒤집힌다.
{top['시도']}가 10만명당 {top['10만명당_매장수']:.2f}개로 1위이고,
매장 수 1위인 서울특별시는 그보다 낮다.

인구가 가장 많은 경기도는 중하위권에 머문다.
거주 인구만으로는 프랜차이즈 입지를 설명할 수 없고,
유동인구·오피스 밀집도·역세권·대학가 같은 요인이 함께 작용한다는 뜻이다.
        """
    )


# ---------------------------------------------------------------------------
# 4. 그래프
# ---------------------------------------------------------------------------
def page_charts():
    st.header("그래프")
    st.caption("노트북 7단계")

    merged = build_merged()
    tab1, tab2 = st.tabs(["막대그래프 — 밀도 순위", "산점도 — 인구와 매장 수"])

    with tab1:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=merged, x="시도", y="10만명당_매장수",
                    hue="시도", palette="YlOrRd_r", legend=False, ax=ax)
        ax.set_title("시도별 인구 10만명당 할리스 매장 수")
        ax.set_xlabel("시도")
        ax.set_ylabel("10만명당 매장 수")
        ax.tick_params(axis="x", rotation=45)
        for p in ax.patches:
            ax.text(p.get_x() + p.get_width() / 2, p.get_height(),
                    f"{p.get_height():.2f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        show_label = st.checkbox("시도 이름 표시", value=True)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(merged["인구"], merged["매장수"], s=70,
                   color="#c0392b", alpha=.75, edgecolor="white")
        if show_label:
            for _, row in merged.iterrows():
                ax.text(row["인구"], row["매장수"], row["시도"], fontsize=9)
        ax.set_title("시도별 인구와 할리스 매장 수 관계")
        ax.set_xlabel("인구")
        ax.set_ylabel("매장 수")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        corr = merged["인구"].corr(merged["매장수"])
        st.info(
            f"인구와 매장 수의 상관계수 **{corr:.3f}** — "
            "인구가 많을수록 매장이 늘어나는 경향은 뚜렷하다. "
            "다만 추세에서 벗어난 지역이 입지 전략의 차이를 보여준다."
        )


# ---------------------------------------------------------------------------
# 5. 밀집 상권 (DBSCAN)
# ---------------------------------------------------------------------------
def page_cluster():
    st.header("밀집 상권 찾기 (DBSCAN)")
    st.caption("가까이 모여 있는 매장을 군집으로 묶는다. 조건을 바꿔가며 확인할 수 있다.")

    from sklearn.cluster import DBSCAN

    df = load_stores().dropna(subset=["위도", "경도"]).reset_index(drop=True)

    c1, c2 = st.columns(2)
    radius_km = c1.slider("반경 (km)", 0.2, 5.0, 0.8, 0.1,
                          help="이 거리 안에 있는 매장을 이웃으로 본다")
    min_samples = c2.slider("최소 매장 수", 2, 10, 5, 1,
                            help="군집으로 인정할 최소 매장 개수")

    kms_per_radian = 6371.0088
    db = DBSCAN(eps=radius_km / kms_per_radian, min_samples=min_samples,
                algorithm="ball_tree", metric="haversine")
    df["cluster"] = db.fit_predict(np.radians(df[["위도", "경도"]].values))

    n_cluster = df.loc[df["cluster"] != -1, "cluster"].nunique()
    n_noise = int((df["cluster"] == -1).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("군집 수", f"{n_cluster}개")
    c2.metric("군집에 속한 매장", f"{len(df) - n_noise}개")
    c3.metric(f"단독 매장 · 노이즈 ({n_noise / len(df) * 100:.1f}%)", f"{n_noise}개")

    if n_cluster == 0:
        st.warning("조건을 만족하는 군집이 없다. 반경을 넓히거나 최소 매장 수를 줄여보자.")
        return

    st.subheader("군집 요약")
    summary = (
        df[df["cluster"] != -1]
        .groupby("cluster")
        .agg(매장수=("매장명", "size"),
             시도=("시도", lambda s: s.mode()[0]),
             대표매장=("매장명", "first"))
        .reset_index()
        .rename(columns={"cluster": "군집"})
        .sort_values("매장수", ascending=False)
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.subheader("군집 매장 지도")
    st.caption("색이 있는 점은 군집, 회색은 단독 매장")

    import folium
    from streamlit_folium import st_folium

    palette = ["red", "blue", "green", "purple", "orange",
               "darkred", "cadetblue", "darkgreen", "pink", "black"]

    m = folium.Map(location=[df["위도"].mean(), df["경도"].mean()], zoom_start=7)
    for _, row in df.iterrows():
        c = int(row["cluster"])
        color = "#b0b0b0" if c == -1 else palette[c % len(palette)]
        folium.CircleMarker(
            location=[row["위도"], row["경도"]],
            radius=3 if c == -1 else 5,
            popup=f"{row['매장명']} (군집 {c})",
            color=color, fill=True, fill_color=color,
            fill_opacity=.3 if c == -1 else .9,
        ).add_to(m)

    st_folium(m, height=520, use_container_width=True, returned_objects=[])

    download_button(df, "hollys_cluster.csv", "군집 결과 CSV 내려받기")


# ---------------------------------------------------------------------------
# 6. 지도 시각화
# ---------------------------------------------------------------------------
def page_map():
    st.header("지도 시각화")
    st.caption("노트북 8단계 — Choropleth")

    import folium
    from streamlit_folium import st_folium

    kind = st.radio("지도 종류", ["밀도 지도 (Choropleth)", "매장 위치"], horizontal=True)

    if kind.startswith("밀도"):
        merged = build_merged()
        geo = load_geojson()

        m = folium.Map(location=[35.9, 127.8], zoom_start=7)
        folium.Choropleth(
            geo_data=geo,
            data=merged,
            columns=["시도", "10만명당_매장수"],
            key_on="feature.properties.name",
            fill_color="YlOrRd",
            fill_opacity=.7,
            line_opacity=.3,
            legend_name="10만명당 매장 수",   # 길면 지도 밖으로 잘린다
        ).add_to(m)

        # 마우스를 올리면 시도 이름과 값이 보이도록 투명 레이어를 덧댄다
        value_map = merged.set_index("시도")["10만명당_매장수"].to_dict()
        for feat in geo["features"]:
            name = feat["properties"]["name"]
            feat["properties"]["표시"] = f"{name} · {value_map.get(name, 0):.2f}개"

        folium.GeoJson(
            geo,
            style_function=lambda x: {"fillOpacity": 0, "weight": 0},
            tooltip=folium.GeoJsonTooltip(fields=["표시"], labels=False),
        ).add_to(m)

        st_folium(m, height=560, use_container_width=True, returned_objects=[])
        st.caption("색이 진할수록 인구 대비 매장이 촘촘하다.")

    else:
        df = load_stores().dropna(subset=["위도", "경도"])
        m = folium.Map(location=[36.5, 127.8], zoom_start=7)
        for _, row in df.iterrows():
            folium.CircleMarker(
                location=[row["위도"], row["경도"]],
                radius=3,
                popup=f"{row['매장명']}<br>{row['주소']}",
                color="#c0392b", fill=True, fill_color="#c0392b", fill_opacity=.8,
            ).add_to(m)
        st_folium(m, height=560, use_container_width=True, returned_objects=[])
        st.caption(f"좌표가 있는 매장 {len(df):,}곳")


# ---------------------------------------------------------------------------
# 7. 데이터 수집 (크롤링)
# ---------------------------------------------------------------------------
def crawl_store_page(page):
    """한 페이지의 매장 목록을 긁는다. 표는 td 6칸 구조."""
    import requests
    from bs4 import BeautifulSoup

    res = requests.get(BASE_URL, params={"pageNo": page})
    if res.status_code != 200:
        return []

    tbody = BeautifulSoup(res.text, "html.parser").select_one("table.tb_store tbody")
    if tbody is None:
        return []

    rows = []
    for tr in tbody.select("tr"):
        tds = tr.select("td")
        if len(tds) < 6:
            continue
        services = "/".join(
            img.get("alt", "").strip() for img in tds[4].select("img") if img.get("alt")
        )
        rows.append([
            tds[0].get_text(strip=True),   # 지역
            tds[1].get_text(strip=True),   # 매장명
            tds[2].get_text(strip=True),   # 현황
            tds[3].get_text(strip=True),   # 주소
            services,                      # 매장서비스
            tds[5].get_text(strip=True),   # 전화번호
        ])
    return rows


def clean_address(address):
    """좌표 검색이 잘 되도록 층·호수·괄호 등을 걷어낸다. (노트북 2단계)"""
    if pd.isna(address):
        return ""

    addr = re.sub(r"\(.*?\)", "", str(address)).split(",")[0]
    for pattern in (r"\d+\s*층", r"\d+\s*호", r"지하\s*\d*", r"B\d+",
                    r"\d+F", r"\d+~\d+층", r"\d+~\d+", r"\s*층"):
        addr = re.sub(pattern, "", addr)

    addr = addr.replace("·", " ").replace(".", " ")
    return re.sub(r"\s+", " ", addr).strip()


def page_collect():
    st.header("데이터 수집")
    st.warning(
        "이 메뉴는 할리스 홈페이지를 직접 호출한다. "
        "시간이 걸리고 이미 저장된 CSV를 덮어쓴다. 필요할 때만 실행하자.",
        icon="⚠️",
    )

    st.subheader("1단계 · 매장 목록 크롤링")
    st.caption(f"출처: {BASE_URL}")

    max_page = st.number_input(
        "가져올 페이지 수", 1, 100, 5,
        help="한 페이지에 10개씩 들어 있다. 전체는 약 45페이지."
    )

    if st.button("크롤링 시작", type="primary"):
        try:
            import requests  # noqa: F401
            from bs4 import BeautifulSoup  # noqa: F401
        except ImportError:
            st.error("requests 와 beautifulsoup4 가 필요하다 → `uv add requests beautifulsoup4`")
            return

        bar = st.progress(0.0, "준비 중…")
        rows = []
        for i, page in enumerate(range(1, int(max_page) + 1), start=1):
            bar.progress(i / max_page, f"{page}/{max_page} 페이지 수집 중…")
            rows.extend(crawl_store_page(page))
            time.sleep(0.3)   # 서버 부담을 줄이기 위한 간격
        bar.empty()

        if not rows:
            st.error("가져온 데이터가 없다. 사이트 구조가 바뀌었을 수 있다.")
            return

        df = pd.DataFrame(rows, columns=["지역", "매장명", "현황", "주소", "매장서비스", "전화번호"])
        os.makedirs(SOURCE_DIR, exist_ok=True)
        df.to_csv(STORE_RAW, index=False, encoding="utf-8")

        st.success(f"매장 {len(df)}개 수집 완료 → {STORE_RAW}")
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("2단계 · 주소를 좌표로 (카카오 API)")

    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("KAKAO_API_KEY")

    if api_key:
        st.success("`.env` 에서 KAKAO_API_KEY 를 찾았다.", icon="🔑")
    else:
        st.error(
            "KAKAO_API_KEY 가 없다. 프로젝트 루트의 `.env` 에 "
            "`KAKAO_API_KEY=발급받은키` 형태로 넣어두자.  \n"
            "발급: developers.kakao.com → 앱 생성 → 앱 키 → REST API 키",
            icon="🔑",
        )

    st.markdown(
        """
좌표 변환은 두 단계로 시도한다.

1. **주소검색** — 전처리한 주소로 찾는다.
2. **키워드검색** — 실패하면 매장명으로 찾는다.
   휴게소점은 `○○휴게소 할리스` 형태로 바꿔 검색한다.

건물 내부·캠퍼스 안 매장은 주소만으로 못 찾는 경우가 있어 2단계가 필요하다.
        """
    )

    st.info(
        "좌표 변환은 매장 1곳당 0.2초씩 걸려 전체 실행에 2분 이상 걸린다. "
        "결과가 이미 `source/hollys_store_geo_kakao_final.csv` 에 있으므로, "
        "새로 크롤링한 경우에만 노트북 2단계 셀에서 실행하는 편이 낫다."
    )

    st.subheader("주소 전처리 확인")
    st.caption("좌표 검색 전에 주소를 어떻게 다듬는지 직접 넣어볼 수 있다.")
    sample = st.text_input(
        "주소 입력",
        "서울특별시 동작구 동작대로25길 16 (사당동) 1층~2층",
    )
    if sample:
        st.code(clean_address(sample), language=None)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
PAGES = {
    "데이터 개요": page_overview,
    "시도별 매장 수": page_store_count,
    "인구 대비 밀도": page_density,
    "그래프": page_charts,
    "밀집 상권 (DBSCAN)": page_cluster,
    "지도": page_map,
    "데이터 수집": page_collect,
}


def main():
    st.sidebar.title("☕ 할리스 상권 분석")
    st.sidebar.caption("종합.ipynb → Streamlit")

    choice = st.sidebar.radio("메뉴", list(PAGES), label_visibility="collapsed")
    st.sidebar.divider()

    missing = missing_files()
    if missing:
        st.sidebar.error("없는 파일:\n" + "\n".join(f"- `{p}`" for p in missing))
    else:
        st.sidebar.success("데이터 파일 준비됨")

    st.sidebar.caption(
        "매장 정보: 할리스 홈페이지 매장검색  \n"
        "인구: KOSIS 주민등록인구현황  \n"
        "경계: southkorea-maps (2018)"
    )

    if missing and choice != "데이터 수집":
        st.error(
            "분석에 필요한 파일이 없다. 아래 파일을 `source/` 에 두고 다시 실행하자.\n\n"
            + "\n".join(f"- `{p}`" for p in missing)
        )
        return

    PAGES[choice]()


if __name__ == "__main__":
    main()
