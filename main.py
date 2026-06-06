# -*- coding: utf-8 -*-
"""
玉川ダム位 & 蒼社川水モニタリング Streamlit アプリ
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

RIVER_URL = (
    "http://183.176.244.72/kawabou-mng/stageKeikaTime.do"
    "?GID=05-1002&KKB=100011&SGN=0110&YKE=01101"
    "&datatype=10000&userId=U1001&grpId=U1001_MMAP002&PG=1&KTM=3"
)
DAM_URL = (
    "http://183.176.244.72/kawabou-mng/customizeMyMenuKeika.do"
    "?GID=05-5101&userId=U1001&myMenuId=U1001_MMENU003&PG=1&KTM=2"
)

REQUEST_TIMEOUT = 15
MAX_RETRIES     = 3
BACKOFF_FACTOR  = 0.5

STATIONS = {
    "片山": {"ymax": 4},
    "高野": {"ymax": 4},
    "中通": {"ymax": 3},
}

DAM_COLUMNS = {
    0: "日時",
    1: "貯水位",
    2: "流入量",
    3: "放流量",
    4: "貯水量",
    5: "貯水率",
}


# ─────────────────────────────────────────
# HTTP セッション（リトライ付き）
# ─────────────────────────────────────────
def _build_session() -> requests.Session:
    session = requests.Session()
    retry_policy = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_policy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# ─────────────────────────────────────────
# 共通ユーティリティ
# ─────────────────────────────────────────
def _parse_datetime_column(raw_col: pd.Series, now: pd.Timestamp) -> pd.Series:
    """「MM/DD HH:MM」形式の文字列列を datetime に変換して返す。"""
    date_time_raw = raw_col.str.extract(r"(?:(\d{2}/\d{2})\s+)?(\d{2}:\d{2})").ffill()

    date_parts = date_time_raw.copy()
    date_parts["year"] = now.year
    date_parts[["month", "day"]] = (
        date_time_raw[0].str.strip().str.split("/", expand=True).astype(int)
    )
    date_parts[["hour", "minute"]] = (
        date_time_raw[1].str.strip().str.split(":", expand=True).astype(int)
    )

    parsed_datetimes = pd.to_datetime(
        date_parts[["year", "month", "day", "hour", "minute"]]
    )
    # 未来日付になっている場合は前年に補正
    is_future = now < parsed_datetimes
    date_parts.loc[is_future, "year"] -= 1
    parsed_datetimes = pd.to_datetime(
        date_parts[["year", "month", "day", "hour", "minute"]]
    )
    return parsed_datetimes


# ─────────────────────────────────────────
# 蒼社川データ取得・パース
# ─────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def fetch_river_data(url: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """水位時系列 water_levels と基準水位 alert_thresholds を返す。"""
    now = pd.Timestamp.now(tz="Asia/Tokyo").tz_localize(None)

    session = _build_session()
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")
    raw_table = soup.select_one("table")
    if raw_table is None:
        raise ValueError("ページ内にテーブルが見つかりませんでした。")

    for tag in raw_table.find_all(attrs={"colspan": True}):
        del tag["colspan"]

    table_html = io.StringIO(raw_table.prettify())

    # ── 時系列データ ──────────────────────
    water_levels = (
        pd.read_html(
            table_html,
            header=0,
            skiprows=range(1, 17),
            na_values=["-", "閉局", "欠測"],
            flavor="bs4",
        )[0]
        .rename(columns={"観測所名": "日時"})
    )
    water_levels["日時"] = _parse_datetime_column(water_levels["日時"], now)
    water_levels.set_index("日時", inplace=True)

    # ── 基準水位（4 行分）────────────────
    table_html.seek(0)
    alert_thresholds = (
        pd.read_html(
            table_html,
            header=0,
            na_values=["-", "閉局", "欠測"],
            flavor="bs4",
        )[0]
        .rename(columns={"観測所名": "水位"})
        .iloc[11:15]
        .set_index("水位")
        .astype(float)
    )
    alert_thresholds.index = alert_thresholds.index.str.replace("(m)", "", regex=False)

    return water_levels, alert_thresholds, now


# ─────────────────────────────────────────
# 玉川ダムデータ取得・パース
# ─────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def fetch_dam_data(url: str) -> tuple[pd.DataFrame, pd.Timestamp]:
    """ダム諸量時系列 dam_records と取得日時を返す。"""
    now = pd.Timestamp.now(tz="Asia/Tokyo").tz_localize(None)

    session = _build_session()
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    dam_records = (
        pd.read_html(
            io.StringIO(response.text),
            na_values=["-", "閉局", "欠測"],
            flavor="bs4",
        )[1]
        .rename(columns=DAM_COLUMNS)
        .dropna(how="all", axis=1)
    )

    dam_records["日時"] = _parse_datetime_column(dam_records["日時"], now)
    dam_records.dropna(subset=["貯水率"], inplace=True)
    dam_records.set_index("日時", inplace=True)

    return dam_records, now


# ─────────────────────────────────────────
# グラフ生成：蒼社川
# ─────────────────────────────────────────
def make_station_figure(
    water_levels: pd.DataFrame,
    alert_thresholds: pd.DataFrame,
    station_name: str,
    y_max: float | None = None,
) -> go.Figure:
    """観測所ごとの水位グラフを返す。"""
    water_series     = water_levels[station_name].dropna()
    threshold_values = alert_thresholds[station_name].dropna()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=water_series.index,
            y=water_series.values,
            mode="lines+markers",
            name=station_name,
            line=dict(width=2),
        )
    )

    threshold_colors = ["red", "orange", "yellow", "blue"]
    for i, (threshold_name, threshold_value) in enumerate(threshold_values.items()):
        fig.add_hline(
            y=threshold_value,
            line_dash="dot",
            line_color=threshold_colors[i % len(threshold_colors)],
            annotation_text=threshold_name,
            annotation_position="right",
        )

    auto_y_max = max(
        water_series.max()      if not water_series.empty      else 0,
        threshold_values.max()  if not threshold_values.empty  else 0,
    ) * 1.1
    fig.update_layout(
        title=f"{station_name} 水位",
        xaxis_title="日時",
        yaxis_title="水位 (m)",
        yaxis=dict(range=[0, y_max if y_max else auto_y_max]),
        hovermode="x unified",
        height=400,
    )
    return fig


# ─────────────────────────────────────────
# グラフ生成：玉川ダム
# ─────────────────────────────────────────
def make_dam_figure(dam_records: pd.DataFrame, metric: str, y_label: str) -> go.Figure:
    """ダム諸量（単一指標）の時系列グラフを返す。"""
    metric_series = dam_records[metric].dropna()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=metric_series.index,
            y=metric_series.values,
            mode="lines+markers",
            name=metric,
            line=dict(width=2),
        )
    )
    fig.update_layout(
        title=f"玉川ダム {metric}",
        xaxis_title="日時",
        yaxis_title=y_label,
        hovermode="x unified",
        height=350,
    )
    return fig


def make_dam_flow_figure(dam_records: pd.DataFrame) -> go.Figure:
    """流入量・放流量を重ねた比較グラフを返す。"""
    fig = go.Figure()
    for flow_name, color in [("流入量", "royalblue"), ("放流量", "tomato")]:
        if flow_name not in dam_records.columns:
            continue
        flow_series = dam_records[flow_name].dropna()
        fig.add_trace(
            go.Scatter(
                x=flow_series.index,
                y=flow_series.values,
                mode="lines+markers",
                name=flow_name,
                line=dict(width=2, color=color),
            )
        )
    fig.update_layout(
        title="玉川ダム 流入量・放流量",
        xaxis_title="日時",
        yaxis_title="流量 (m³/s)",
        hovermode="x unified",
        height=350,
    )
    return fig


# ─────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="蒼社川・玉川ダム モニター",
        page_icon="🌊",
        layout="wide",
    )
    st.title("🌊 蒼社川・玉川ダム モニター")

    # ── サイドバー ────────────────────────
    with st.sidebar:
        st.header("設定")
        selected_stations = st.multiselect(
            "表示する観測所（蒼社川）",
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
        river_error = dam_error = None
        try:
            water_levels, alert_thresholds, river_now = fetch_river_data(RIVER_URL)
        except requests.exceptions.Timeout:
            river_error = "⚠️ 蒼社川：タイムアウトしました。しばらく後に再試行してください。"
        except requests.exceptions.ConnectionError:
            river_error = "⚠️ 蒼社川：サーバーに接続できませんでした。"
        except requests.exceptions.HTTPError as e:
            river_error = f"⚠️ 蒼社川 HTTPエラー：{e}"
        except (ValueError, KeyError) as e:
            river_error = f"⚠️ 蒼社川 データ解析エラー：{e}"

        try:
            dam_records, dam_now = fetch_dam_data(DAM_URL)
        except requests.exceptions.Timeout:
            dam_error = "⚠️ 玉川ダム：タイムアウトしました。しばらく後に再試行してください。"
        except requests.exceptions.ConnectionError:
            dam_error = "⚠️ 玉川ダム：サーバーに接続できませんでした。"
        except requests.exceptions.HTTPError as e:
            dam_error = f"⚠️ 玉川ダム HTTPエラー：{e}"
        except (ValueError, KeyError) as e:
            dam_error = f"⚠️ 玉川ダム データ解析エラー：{e}"

    # ══════════════════════════════════════
    # 玉川ダムセクション
    # ══════════════════════════════════════
    st.header("🏔️ 玉川ダム")

    if dam_error:
        st.error(dam_error)
    else:
        st.caption(f"取得日時：{dam_now.strftime('%Y-%m-%d %H:%M')} JST")

        # サマリーカード
        dam_metric_cols = st.columns(4)
        dam_summary_metrics = [
            ("貯水率",  "%"),
            ("貯水位",  "m"),
            ("流入量",  "m³/s"),
            ("放流量",  "m³/s"),
        ]
        for col, (metric_name, unit) in zip(dam_metric_cols, dam_summary_metrics):
            latest_series = dam_records[metric_name].dropna()
            if not latest_series.empty:
                latest_value = latest_series.iloc[-1]
                one_hour_ago = latest_series.index[-1] - pd.Timedelta(hours=1)
                past_series  = latest_series[latest_series.index <= one_hour_ago]
                if not past_series.empty:
                    hourly_diff = latest_value - past_series.iloc[-1]
                    delta_text  = f"{hourly_diff:+.2f} {unit}（1時間前比）"
                    delta_color = "normal"
                else:
                    delta_text  = "（1時間前データなし）"
                    delta_color = "off"
                col.metric(
                    label=metric_name,
                    value=f"{latest_value:.1f} {unit}",
                    delta=delta_text,
                    delta_color=delta_color,
                )

        st.divider()

        # グラフ：貯水率・貯水位は単独、流入量・放流量は統合
        for metric_name, y_label in [("貯水率", "貯水率 (%)"), ("貯水位", "貯水位 (m)")]:
            if metric_name in dam_records.columns:
                st.plotly_chart(
                    make_dam_figure(dam_records, metric_name, y_label),
                    use_container_width=True,
                )
        st.plotly_chart(make_dam_flow_figure(dam_records), use_container_width=True)

        # 生データ
        with st.expander("📋 生データを表示（玉川ダム）"):
            st.dataframe(dam_records, use_container_width=True)

    # ══════════════════════════════════════
    # 蒼社川セクション
    # ══════════════════════════════════════
    st.header("🏞️ 蒼社川 水位")

    if river_error:
        st.error(river_error)
    else:
        st.caption(f"取得日時：{river_now.strftime('%Y-%m-%d %H:%M')} JST")

        if selected_stations:
            # サマリーカード
            metric_cols = st.columns(len(selected_stations))
            for col, station_name in zip(metric_cols, selected_stations):
                latest_series = water_levels[station_name].dropna()
                if not latest_series.empty:
                    latest_value = latest_series.iloc[-1]
                    one_hour_ago = latest_series.index[-1] - pd.Timedelta(hours=1)
                    past_series  = latest_series[latest_series.index <= one_hour_ago]
                    if not past_series.empty:
                        hourly_diff = latest_value - past_series.iloc[-1]
                        delta_text  = f"{hourly_diff:+.2f} m（1時間前比）"
                        delta_color = "normal"
                    else:
                        delta_text  = "（1時間前データなし）"
                        delta_color = "off"
                    col.metric(
                        label=station_name,
                        value=f"{latest_value:.2f} m",
                        delta=delta_text,
                        delta_color=delta_color,
                    )

            st.divider()

            # グラフ
            for station_name in selected_stations:
                if station_name not in water_levels.columns:
                    st.warning(f"「{station_name}」のデータが見つかりません。")
                    continue
                fig = make_station_figure(
                    water_levels, alert_thresholds,
                    station_name, STATIONS[station_name]["ymax"],
                )
                st.plotly_chart(fig, use_container_width=True)

            # 生データ
            with st.expander("📋 生データを表示（蒼社川）"):
                st.dataframe(water_levels[selected_stations], use_container_width=True)
                st.subheader("基準水位")
                st.dataframe(alert_thresholds[selected_stations], use_container_width=True)
        else:
            st.info("サイドバーから観測所を選択してください。")


if __name__ == "__main__":
    main()
