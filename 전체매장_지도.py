"""
할리스 전국 매장 지도 (단독 HTML)

전국 매장을 한 화면에 모두 띄우고, 지도를 움직여도 점이 사라지지 않는다.
매장에 마우스를 올리면 이름이 뜨고, 누르면 주소·전화번호가 나온다.

실행:
    uv run python 전체매장_지도.py

결과:
    output/hollys_all_stores_map.html   ← 브라우저로 바로 열면 된다
"""

import json
import os

import folium
import pandas as pd
from branca.colormap import linear
from folium.plugins import Fullscreen, MiniMap, MousePosition, Search

STORE_GEO = "source/hollys_store_geo_kakao_final.csv"
POPULATION = "source/population_sido.csv"
GEOJSON = "source/korea_sido.geojson"
OUT_HTML = "output/hollys_all_stores_map.html"

BRAND_RED = "#C8102E"

# GeoJSON 이 2018년 기준이라 개편 전 지명을 쓴다. 현재 지명으로 맞춘다.
NAME_FIX = {"강원도": "강원특별자치도", "전라북도": "전북특별자치도"}


def load_stores() -> pd.DataFrame:
    df = pd.read_csv(STORE_GEO)
    return df.dropna(subset=["위도", "경도"]).reset_index(drop=True)


def load_density() -> pd.DataFrame:
    """시도별 10만명당 매장 수. 배경 색칠용."""
    stores = load_stores()["시도"].value_counts().reset_index()
    stores.columns = ["시도", "매장수"]

    merged = stores.merge(pd.read_csv(POPULATION), on="시도", how="inner")
    merged["10만명당_매장수"] = merged["매장수"] / merged["인구"] * 100000
    return merged


def load_geojson(step: int = 3) -> dict:
    """
    시도 경계.

    원본 7.2MB를 그대로 넣으면 HTML이 무거워 브라우저에서 버벅인다.
    좌표를 step 간격으로 솎고 소수점을 4자리로 줄인다.
    """
    def thin(ring):
        pts = ring if len(ring) <= 12 else ring[::step] + [ring[-1]]
        return [[round(float(x), 4), round(float(y), 4)] for x, y in pts]

    with open(GEOJSON, encoding="utf-8") as f:
        geo = json.load(f)

    for feat in geo["features"]:
        feat["properties"]["name"] = NAME_FIX.get(
            feat["properties"]["name"], feat["properties"]["name"]
        )
        geom = feat["geometry"]
        if geom["type"] == "Polygon":
            geom["coordinates"] = [thin(r) for r in geom["coordinates"]]
        elif geom["type"] == "MultiPolygon":
            geom["coordinates"] = [[thin(r) for r in p] for p in geom["coordinates"]]

    return geo


def clean_value(value) -> str:
    """빈 값이나 '.' 같은 자리채움 문자를 '-' 로 통일한다."""
    if pd.isna(value):
        return "-"
    text = str(value).strip()
    return "-" if text in ("", ".", "-") else text


def stores_to_geojson(df: pd.DataFrame) -> dict:
    """매장을 GeoJSON 으로. 검색 기능(Search 플러그인)이 이 형식을 요구한다."""
    features = []
    for _, row in df.iterrows():
        phone = clean_value(row["전화번호"])       # 원본에 '.' 로 들어온 곳이 26군데 있다
        service = clean_value(row["매장서비스"])
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point",
                         "coordinates": [float(row["경도"]), float(row["위도"])]},
            "properties": {
                "매장명": row["매장명"],
                "주소": row["주소"],
                "전화번호": phone,
                "시도": row["시도"],
                "매장서비스": service,
            },
        })
    return {"type": "FeatureCollection", "features": features}


def build_map() -> folium.Map:
    df = load_stores()

    m = folium.Map(location=[36.3, 127.8], zoom_start=7, tiles=None)

    # 밝은 배경 타일을 기본으로. 붉은 점이 잘 보인다.
    folium.TileLayer("CartoDB positron", name="밝은 지도").add_to(m)
    folium.TileLayer("OpenStreetMap", name="일반 지도").add_to(m)

    # ── 배경: 시도별 밀도 (기본은 꺼둠) ────────────────────────────
    # Choropleth 는 Map 에만 붙일 수 있어 껐다 켤 수 없다.
    # 색을 직접 계산해 GeoJson 으로 만들면 LayerControl 로 토글된다.
    density = load_density()
    geo = load_geojson()

    value_map = density.set_index("시도")["10만명당_매장수"].to_dict()
    colormap = linear.YlOrRd_09.scale(
        min(value_map.values()), max(value_map.values())
    )
    colormap.caption = "10만명당 매장 수"

    for feat in geo["features"]:
        name = feat["properties"]["name"]
        feat["properties"]["밀도"] = f"{value_map.get(name, 0):.2f}개"

    folium.GeoJson(
        geo,
        name="시도별 밀도 (10만명당)",
        show=False,
        style_function=lambda f: {
            "fillColor": colormap(value_map.get(f["properties"]["name"], 0)),
            "color": "#888",
            "weight": 1,
            "fillOpacity": .55,
        },
        highlight_function=lambda f: {"weight": 3, "color": "#333"},
        tooltip=folium.GeoJsonTooltip(fields=["name", "밀도"],
                                      aliases=["시도", "10만명당"]),
    ).add_to(m)

    colormap.add_to(m)

    # ── 매장 (444곳) ──────────────────────────────────────────────
    # 표시와 검색을 레이어 하나로 처리한다.
    # 표시용 CircleMarker 와 검색용 GeoJson 을 따로 두면 같은 자리에
    # 점이 두 겹으로 깔려 클릭이 엉키고 렌더링도 두 배가 된다.
    store_layer = folium.GeoJson(
        stores_to_geojson(df),
        name=f"할리스 매장 ({len(df)})",
        marker=folium.CircleMarker(
            radius=5, color="white", weight=1.5,
            fill=True, fill_color=BRAND_RED, fill_opacity=.9,
        ),
        tooltip=folium.GeoJsonTooltip(fields=["매장명"], labels=False,
                                      style="font-weight:700"),
        popup=folium.GeoJsonPopup(
            fields=["매장명", "주소", "전화번호", "매장서비스"],
            aliases=["매장", "주소", "전화", "서비스"],
            max_width=280,
        ),
    )
    store_layer.add_to(m)

    # ── 검색 (매장명) ─────────────────────────────────────────────
    Search(
        layer=store_layer,
        search_label="매장명",
        placeholder="매장명 검색 (예: 이수점)",
        collapsed=False,
        position="topright",
    ).add_to(m)

    # ── 편의 기능 ─────────────────────────────────────────────────
    Fullscreen(title="전체화면", title_cancel="닫기", position="topleft").add_to(m)
    MiniMap(toggle_display=True, position="bottomleft").add_to(m)
    MousePosition(position="bottomright", prefix="좌표",
                  num_digits=4, separator=" , ").add_to(m)

    folium.LayerControl(collapsed=True, position="topright").add_to(m)

    # ── 제목 ──────────────────────────────────────────────────────
    # 왼쪽 아래 확대·전체화면 버튼과 오른쪽 위 범례·검색창을 피해 왼쪽 위에 둔다.
    title = f"""
    <div style="position:fixed;top:106px;left:10px;z-index:9999;
                background:rgba(255,255,255,.95);padding:10px 14px;
                border-left:4px solid {BRAND_RED};border-radius:4px;
                box-shadow:0 2px 10px rgba(0,0,0,.18);max-width:230px;
                font-family:-apple-system,'Malgun Gothic',sans-serif">
      <div style="font-size:15px;font-weight:800;color:{BRAND_RED};margin-bottom:5px">
        할리스 전국 매장 {len(df)}곳
      </div>
      <div style="font-size:11px;color:#555;line-height:1.6">
        점에 마우스를 올리면 <b>매장명</b><br>
        누르면 <b>주소 · 전화번호</b><br>
        오른쪽 위에서 <b>검색 · 밀도 지도</b>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title))

    return m


def main():
    os.makedirs("output", exist_ok=True)

    m = build_map()
    m.save(OUT_HTML)

    size_mb = os.path.getsize(OUT_HTML) / 1024 / 1024
    print(f"저장 완료: {OUT_HTML} ({size_mb:.1f} MB)")
    print(f"매장 {len(load_stores())}곳 표시")


if __name__ == "__main__":
    main()
