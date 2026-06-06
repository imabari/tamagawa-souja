# -*- coding: utf-8 -*-
"""
河川水位モニタリング Streamlit アプリ
"""

import io
import logging

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ─────────────────────────────────────────
# 設定
# ─────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_URL = (
    "http://183.176.244.72/kawabou-mng/stageKeikaTime.do"
    "?GID=05-1002&KKB=100011&SGN=0110&YKE=01101"
    "&datatype=10000&userId=U1001&grpId=U1001_MMAP002&PG=1&KTM=3"
)

REQUEST_TIMEOUT = 15       # 秒
MAX_RETRIES     = 3
BACKOFF_FACTOR  = 0.5

STATIONS = {
    "片山": {"ymax": 4},
    "高野": {"ymax": 4},
    "中通": {"ymax": 3},
}

# ─────────────────────────────────────────
# HTTP セッション（リトライ付き）
# ─────────────────────────────────────────
def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# ─────────────────────────────────────────
# データ取得・パース
# ─────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)   # 10 分キャッシュ
def fetch_river_data(url: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """水位時系列 df1 と基準水位 df2 を返す。"""
    dt_now = pd.Timestamp.now(tz="Asia/Tokyo").tz_localize(None)

    session = _build_session()
    resp = session.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.content, "html.parser")
    table = soup.select_one("table")
    if table is None:
        raise ValueError("ページ内にテーブルが見つかりませんでした。")

    # colspan 属性を除去しないと pd.read_html が列数をずらす
    for tag in table.find_all(attrs={"colspan": True}):
        del tag["colspan"]

    html_str = io.StringIO(table.prettify())

    # ── 時系列データ ──────────────────────
    df1 = (
        pd.read_html(
            html_str,
            header=0,
            skiprows=range(1, 17),
            na_values=["-", "閉局", "欠測"],
        )[0]
        .rename(columns={"観測所名": "日時"})
    )
    df1 = _parse_datetime_index(df1, dt_now)

    # ── 基準水位（4 行分）────────────────
    html_str.seek(0)
    df2 = (
        pd.read_html(
            html_str,
            header=0,
            na_values=["-", "閉局", "欠測"],
        )[0]
        .rename(columns={"観測所名": "水位"})
        .iloc[11:15]
        .set_index("水位")
        .astype(float)
    )
    df2.index = df2.index.str.replace("(m)", "", regex=False)

    return df1, df2, dt_now


def _parse_datetime_index(df: pd.DataFrame, dt_now: pd.Timestamp) -> pd.DataFrame:
    """「日時」列を datetime に変換してインデックスにセットする。"""
    extracted = df["日時"].str.extract(r"(?:(\d{2}/\d{2})\s+)?(\d{2}:\d{2})").ffill()

    date_parts = extracted.copy()
    date_parts["year"] = dt_now.year
    date_parts[["month", "day"]] = (
        extracted[0].str.strip().str.split("/", expand=True).astype(int)
    )
    date_parts[["hour", "minute"]] = (
        extracted[1].str.strip().str.split(":", expand=True).astype(int)
    )

    datetimes = pd.to_datetime(
        date_parts[["year", "month", "day", "hour", "minute"]]
    )
    # 未来日付になっている場合は前年に補正
    past_mask = dt_now < datetimes
    date_parts.loc[past_mask, "year"] -= 1
    datetimes = pd.to_datetime(
        date_parts[["year", "month", "day", "hour", "minute"]]
    )

    df = df.copy()
    df["日時"] = datetimes
    df.set_index("日時", inplace=True)
    return df


# ─────────────────────────────────────────
# グラフ生成
# ─────────────────────────────────────────
def make_station_figure(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    station: str,
    ymax: float | None = None,
) -> go.Figure:
    """観測所ごとの水位グラフを返す。"""
    series = df1[station].dropna()
    levels = df2[station].dropna()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines+markers",
            name=station,
            line=dict(width=2),
        )
    )

    # 基準水位の水平線
    colors = ["red", "orange", "yellow", "blue"]
    for i, (level_name, level_value) in enumerate(levels.items()):
        fig.add_hline(
            y=level_value,
            line_dash="dot",
            line_color=colors[i % len(colors)],
            annotation_text=level_name,
            annotation_position="right",
        )

    auto_ymax = max(
        series.max() if not series.empty else 0,
        levels.max() if not levels.empty else 0,
    ) * 1.1
    fig.update_layout(
        title=f"{station} 水位",
        xaxis_title="日時",
        yaxis_title="水位 (m)",
        yaxis=dict(range=[0, ymax if ymax else auto_ymax]),
        hovermode="x unified",
        height=400,
    )
    return fig


# ─────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="蒼社川水位モニター",
        page_icon="🌊",
        layout="wide",
    )
    st.title("🌊 蒼社川水位モニター")

    # ── サイドバー ────────────────────────
    with st.sidebar:
        st.header("設定")
        selected_stations = st.multiselect(
            "表示する観測所",
            options=list(STATIONS.keys()),
            default=list(STATIONS.keys()),
        )
        auto_refresh = st.checkbox("10分ごとに自動更新", value=False)
        if st.button("🔄 今すぐ更新"):
            st.cache_data.clear()
            st.rerun()

    if auto_refresh:
        st.markdown(
            '<meta http-equiv="refresh" content="600">',
            unsafe_allow_html=True,
        )

    # ── データ取得 ────────────────────────
    with st.spinner("データ取得中…"):
        try:
            df1, df2, dt_now = fetch_river_data(DATA_URL)
        except requests.exceptions.Timeout:
            st.error("⚠️ タイムアウト：サーバーへの接続がタイムアウトしました。しばらく後に再試行してください。")
            return
        except requests.exceptions.ConnectionError:
            st.error("⚠️ 接続エラー：サーバーに接続できませんでした。ネットワーク環境をご確認ください。")
            return
        except requests.exceptions.HTTPError as e:
            st.error(f"⚠️ HTTPエラー：{e}")
            return
        except (ValueError, KeyError) as e:
            st.error(f"⚠️ データ解析エラー：{e}")
            return

    st.caption(f"取得日時：{dt_now.strftime('%Y-%m-%d %H:%M')} JST")

    # ── サマリーカード ────────────────────
    if selected_stations:
        cols = st.columns(len(selected_stations))
        for col, station in zip(cols, selected_stations):
            latest = df1[station].dropna()
            if not latest.empty:
                val = latest.iloc[-1]
                ymax = STATIONS[station]["ymax"]
                pct = val / ymax * 100
                col.metric(
                    label=station,
                    value=f"{val:.2f} m",
                    delta=f"/{ymax} m ({pct:.0f}%)",
                )

        st.divider()

        # ── グラフ ─────────────────────────
        for station in selected_stations:
            if station not in df1.columns:
                st.warning(f"「{station}」のデータが見つかりません。")
                continue
            fig = make_station_figure(df1, df2, station, STATIONS[station]["ymax"])
            st.plotly_chart(fig, use_container_width=True)

        # ── 生データ表示（折りたたみ）──────
        with st.expander("📋 生データを表示"):
            st.dataframe(df1[selected_stations], use_container_width=True)
            st.subheader("基準水位")
            st.dataframe(df2[selected_stations], use_container_width=True)
    else:
        st.info("サイドバーから観測所を選択してください。")


if __name__ == "__main__":
    main()
