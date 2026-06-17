"""
改善版 Football Analytics Dashboard
=====================================
改善点:
- ⑤ Luck（偶然性）スコアの追加・可視化
- クリティカル指標を攻撃/守備に分解表示
- Radarチャートで選手の多軸比較
- チーム対比バーチャートの追加
- 選手詳細ポップアップ
"""

import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.stats import zscore
from statsbombpy import sb

warnings.filterwarnings("ignore")

# =========================================================
# xT グリッド（12×8）
# =========================================================
XT_GRID = np.array([
    [0.00, 0.00, 0.00, 0.00, 0.01, 0.02, 0.03, 0.04],
    [0.00, 0.00, 0.00, 0.01, 0.01, 0.02, 0.03, 0.04],
    [0.00, 0.00, 0.01, 0.01, 0.02, 0.03, 0.04, 0.06],
    [0.00, 0.01, 0.01, 0.02, 0.03, 0.04, 0.06, 0.09],
    [0.01, 0.01, 0.02, 0.03, 0.04, 0.06, 0.09, 0.14],
    [0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12, 0.18],
    [0.02, 0.03, 0.04, 0.06, 0.08, 0.12, 0.18, 0.25],
    [0.03, 0.04, 0.06, 0.08, 0.10, 0.15, 0.22, 0.35],
    [0.04, 0.05, 0.07, 0.10, 0.13, 0.20, 0.30, 0.45],
    [0.05, 0.07, 0.09, 0.13, 0.18, 0.28, 0.40, 0.60],
    [0.07, 0.09, 0.12, 0.18, 0.25, 0.38, 0.55, 0.75],
    [0.10, 0.13, 0.18, 0.25, 0.35, 0.50, 0.70, 0.90],
])


def get_xt(x: float, y: float) -> float:
    xi = min(int(x / 10), 11)
    yi = min(int(y / 10), 7)
    return float(XT_GRID[xi, yi])


# =========================================================
# ページ設定
# =========================================================
st.set_page_config(
    page_title="Football Analytics v2",
    layout="wide",
    page_icon="⚽"
)

st.title("⚽ 新5大指標 サッカー分析ダッシュボード v2")
st.markdown(
    "**①攻撃プロセス ②守備プロセス ③得点近接 ④失点近接 ⑤Luck（偶然性）**  \n"
    "2022 FIFA ワールドカップ 全64試合 分析アプリ"
)

# =========================================================
# データロード・分析
# =========================================================
@st.cache_data
def load_matches():
    return sb.matches(competition_id=43, season_id=106)


@st.cache_data(show_spinner="試合データを分析中...（初回のみ時間がかかります）")
def analyze_match(match_id, home_team, away_team):
    events = sb.events(match_id=match_id)
    events = events.dropna(subset=["player"]).copy()

    events["time_seconds"] = (
        events["minute"] * 60 + events["second"]
        + (events["period"] - 1) * 45 * 60
    )
    events = events.sort_values("time_seconds").reset_index(drop=True)

    players = (
        events[["player", "team"]]
        .drop_duplicates()
        .set_index("player")
    )
    scores = pd.DataFrame(index=players.index)
    scores["team"] = players["team"]

    pa  = dict.fromkeys(scores.index, 0.0)
    pd_ = dict.fromkeys(scores.index, 0.0)
    ca  = dict.fromkeys(scores.index, 0.0)
    cd  = dict.fromkeys(scores.index, 0.0)

    passes = events[events["type"] == "Pass"].copy()

    # ① 攻撃プロセス (xTベース)
    for _, ev in passes.iterrows():
        try:
            sx, sy = ev["location"][0], ev["location"][1]
            ex, ey = ev["pass_end_location"][0], ev["pass_end_location"][1]
        except (TypeError, IndexError, KeyError):
            continue

        passer = ev["player"]
        receiver = ev.get("pass_recipient")
        xt_gain = get_xt(ex, ey) - get_xt(sx, sy)

        if pd.isna(ev.get("pass_outcome")):
            if xt_gain > 0:
                pa[passer] += xt_gain * 10
                if receiver in pa:
                    pa[receiver] += xt_gain * 5
            if sx < 80 <= ex:
                pa[passer] += 1.5
            if ex >= 102 and 18 <= ey <= 62:
                pa[passer] += 2.5
            if ev.get("under_pressure") is True and xt_gain >= 0:
                pa[passer] += 0.3
        else:
            pa[passer] -= 2.0 if sx < 40 else 0.5

    carries = events[events["type"] == "Carry"].copy()
    for _, ev in carries.iterrows():
        try:
            sx, sy = ev["location"][0], ev["location"][1]
            ex, ey = ev["carry_end_location"][0], ev["carry_end_location"][1]
        except (TypeError, IndexError, KeyError):
            continue
        xt_gain = get_xt(ex, ey) - get_xt(sx, sy)
        if xt_gain > 0:
            pa[ev["player"]] += xt_gain * 12

    for _, ev in events[events["type"] == "Foul Won"].iterrows():
        if ev["player"] in pa:
            try:
                x = ev["location"][0]
                pa[ev["player"]] += 0.3 if x >= 80 else 0.1
            except (TypeError, IndexError):
                pa[ev["player"]] += 0.1

    # ② 守備プロセス
    def zone_weight(location, base=1.0):
        try:
            x = location[0]
            return base * (2.0 if x >= 80 else 1.2 if x >= 40 else 0.8)
        except (TypeError, IndexError):
            return base

    for _, ev in events[events["type"] == "Interception"].iterrows():
        if ev["player"] in pd_:
            pd_[ev["player"]] += zone_weight(ev.get("location"), 2.0)

    for _, ev in events[events["type"] == "Ball Recovery"].iterrows():
        if ev["player"] in pd_:
            pd_[ev["player"]] += zone_weight(ev.get("location"), 1.0)

    duels = events[events["type"] == "Duel"]
    if "duel_outcome" in duels.columns:
        won = duels[duels["duel_outcome"].isin(["Won", "Success", "Tackle"])]
        for _, ev in won.iterrows():
            if ev["player"] in pd_:
                pd_[ev["player"]] += zone_weight(ev.get("location"), 1.5)

    for _, ev in events[events["type"] == "Clearance"].iterrows():
        if ev["player"] in pd_:
            try:
                x = ev["location"][0]
                pd_[ev["player"]] += 1.5 if x < 30 else 0.8
            except (TypeError, IndexError):
                pd_[ev["player"]] += 0.8

    # ③ クリティカルアタック
    shots = events[events["type"] == "Shot"].copy()
    for i, (idx, shot) in enumerate(shots.iterrows()):
        xg = shot.get("shot_statsbomb_xg", 0)
        if pd.isna(xg):
            xg = 0.0
        body_part = shot.get("shot_body_part", "")
        xg_adj = xg * (1.2 if body_part in ["Head", "No Touch"] else 1.0)

        shooter = shot["player"]
        if shooter in ca:
            ca[shooter] += xg_adj * 0.50

        contributors = []
        pos = events.index.get_loc(idx)
        for step in range(1, 20):
            prev_pos = pos - step
            if prev_pos < 0:
                break
            prev = events.iloc[prev_pos]
            if prev["team"] != shot["team"]:
                break
            pl = prev["player"]
            if pl != shooter and pl not in contributors:
                contributors.append(pl)
            if len(contributors) >= 2:
                break

        for j, contrib in enumerate(contributors[:2]):
            if contrib in ca:
                ca[contrib] += xg_adj * ([0.30, 0.20][j])

    # ④ クリティカルディフェンス
    loss_events = events[events["type"].isin(["Miscontrol", "Dispossessed"])]
    for _, loss in loss_events.iterrows():
        t = loss["time_seconds"]
        future_shots = shots[
            (shots["team"] != loss["team"])
            & (shots["time_seconds"] >= t)
            & (shots["time_seconds"] <= t + 6)
        ]
        if not future_shots.empty:
            penalty = future_shots["shot_statsbomb_xg"].fillna(0).sum()
            if loss["player"] in cd:
                cd[loss["player"]] -= penalty * 1.2

    gk_events = events[events["type"] == "Goal Keeper"]
    for _, ev in gk_events.iterrows():
        if ev.get("goalkeeper_type") in ["Save", "Shot Saved"]:
            if ev["player"] in cd:
                shot_near = shots[abs(shots["time_seconds"] - ev["time_seconds"]) < 2]
                xg_saved = shot_near["shot_statsbomb_xg"].fillna(0.1).sum()
                cd[ev["player"]] += max(xg_saved, 0.1)

    blocks = events[events["type"] == "Block"]
    for _, ev in blocks.iterrows():
        if ev["player"] in cd:
            shot_near = shots[abs(shots["time_seconds"] - ev["time_seconds"]) < 2]
            xg_blocked = shot_near["shot_statsbomb_xg"].fillna(0.05).sum()
            cd[ev["player"]] += max(xg_blocked * 0.8, 0.05)

    # スコア集計
    scores["①攻撃プロセス"] = pd.Series(pa)
    scores["②守備プロセス"] = pd.Series(pd_)
    scores["③得点近接"] = pd.Series(ca)
    scores["④失点近接"] = pd.Series(cd)

    for col in ["①攻撃プロセス", "②守備プロセス"]:
        if scores[col].std() > 0:
            scores[col] = zscore(scores[col])

    scores["総合プロセス(①+②)"] = scores["①攻撃プロセス"] + scores["②守備プロセス"]
    scores["総合クリティカル(③+④)"] = scores["③得点近接"] + scores["④失点近接"]

    # チーム集計
    xg_team = shots.groupby("team")["shot_statsbomb_xg"].sum()
    team_goals = {home_team: 0, away_team: 0}
    for _, s in shots.iterrows():
        if s.get("shot_outcome") == "Goal":
            team_goals[s["team"]] = team_goals.get(s["team"], 0) + 1

    team_summary = {}
    for t, opp in [(home_team, away_team), (away_team, home_team)]:
        tm = scores[scores["team"] == t]
        xg_t = float(xg_team.get(t, 0))
        xg_o = float(xg_team.get(opp, 0))
        goals_t = team_goals.get(t, 0)
        goals_o = team_goals.get(opp, 0)

        team_summary[t] = {
            "得点": goals_t,
            "失点": goals_o,
            "xG": xg_t,
            "被xG": xg_o,
            "プロセス合計": float(tm["総合プロセス(①+②)"].sum()),
            "クリティカル合計": float(tm["総合クリティカル(③+④)"].sum()),
            "攻撃クリティカル": float(tm["③得点近接"].sum()),
            "守備クリティカル": float(tm["④失点近接"].sum()),
            "⑤得点Luck(決定力)": goals_t - xg_t,
            "⑤守備Luck(死守度)": xg_o - goals_o,
        }

    return scores.reset_index(), team_summary


# =========================================================
# サイドバー: 試合選択
# =========================================================
df_matches = load_matches()
st.sidebar.header("🔍 試合フィルター")
search_q = st.sidebar.text_input("国名で絞り込み (例: Japan, Brazil)", "").lower()

if search_q:
    fdf = df_matches[
        df_matches["home_team"].str.lower().str.contains(search_q)
        | df_matches["away_team"].str.lower().str.contains(search_q)
    ]
else:
    fdf = df_matches

if fdf.empty:
    st.sidebar.warning("該当試合なし")
    st.stop()

fdf = fdf.copy()
fdf["label"] = (
    fdf["match_date"] + " | "
    + fdf["home_team"] + " " + fdf["home_score"].astype(str)
    + " - "
    + fdf["away_score"].astype(str) + " " + fdf["away_team"]
)

selected_label = st.sidebar.selectbox("試合を選択", fdf["label"])
row = fdf[fdf["label"] == selected_label].iloc[0]
match_id = int(row["match_id"])
home_t, away_t = row["home_team"], row["away_team"]

# =========================================================
# 分析実行
# =========================================================
df_players, team_sum = analyze_match(match_id, home_t, away_t)

# =========================================================
# UI: チームサマリー
# =========================================================
st.subheader(f"📊 {home_t}  vs  {away_t}")
st.caption(f"スコア: {int(row['home_score'])} - {int(row['away_score'])}  |  {row['match_date']}")

col_h, col_sep, col_a = st.columns([5, 1, 5])

def render_team_card(col, team, data, is_home):
    with col:
        result_emoji = "🏆" if (
            (is_home and data["得点"] > data["失点"])
            or (not is_home and data["失点"] > data["得点"])
        ) else ("🤝" if data["得点"] == data["失点"] else "❌")

        st.markdown(f"### {result_emoji} {team}")
        c1, c2, c3 = st.columns(3)
        c1.metric("得点", data["得点"])
        c2.metric("xG", f"{data['xG']:.2f}",
                  delta=f"{data['⑤得点Luck(決定力)']:+.2f}",
                  delta_color="normal")
        c3.metric("被xG", f"{data['被xG']:.2f}",
                  delta=f"{data['⑤守備Luck(死守度)']:+.2f}",
                  delta_color="normal")

        luck_atk = data["⑤得点Luck(決定力)"]
        luck_def = data["⑤守備Luck(死守度)"]

        st.markdown(
            f"🎯 **得点Luck(決定力):** `{luck_atk:+.2f}` "
            + ("✨確率超えゴール" if luck_atk > 0.3 else
               "⚡標準的" if luck_atk > -0.3 else "😬決定機逸し")
        )
        st.markdown(
            f"🛡️ **守備Luck(死守度):** `{luck_def:+.2f}` "
            + ("🧱神守備" if luck_def > 0.3 else
               "⚡標準的" if luck_def > -0.3 else "💥守備崩壊")
        )

        st.progress(
            min(max((data["プロセス合計"] + 15) / 30, 0.0), 1.0),
            text=f"ゲーム支配力(プロセス): {data['プロセス合計']:.2f}"
        )
        st.progress(
            min(max((data["クリティカル合計"] + 5) / 10, 0.0), 1.0),
            text=f"決定局面支配(クリティカル): {data['クリティカル合計']:.2f}"
        )


render_team_card(col_h, home_t, team_sum[home_t], True)
with col_sep:
    st.markdown("<div style='text-align:center; padding-top:80px; font-size:24px'>VS</div>", unsafe_allow_html=True)
render_team_card(col_a, away_t, team_sum[away_t], False)

# =========================================================
# UI: チーム比較チャート
# =========================================================
st.markdown("---")
st.subheader("📈 チーム指標 対比")

fig_comp, ax_comp = plt.subplots(figsize=(10, 4))
categories = ["プロセス合計", "攻撃クリティカル", "守備クリティカル",
              "⑤得点Luck(決定力)", "⑤守備Luck(死守度)", "xG"]
home_vals = [team_sum[home_t][c] for c in categories]
away_vals = [team_sum[away_t][c] for c in categories]

x = np.arange(len(categories))
w = 0.35
bars_h = ax_comp.bar(x - w/2, home_vals, w, label=home_t, color="#e74c3c", alpha=0.8)
bars_a = ax_comp.bar(x + w/2, away_vals, w, label=away_t, color="#3498db", alpha=0.8)
ax_comp.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax_comp.set_xticks(x)
ax_comp.set_xticklabels(categories, rotation=15, ha="right", fontsize=9)
ax_comp.legend()
ax_comp.set_title("チーム指標 比較（棒グラフ）", fontsize=12)
plt.tight_layout()
st.pyplot(fig_comp)

# =========================================================
# UI: 選手個人スタッツ
# =========================================================
st.markdown("---")
st.subheader("🏃 選手個人スタッツ")

target_team = st.radio("チームを選択", [home_t, away_t], horizontal=True)
df_team = df_players[df_players["team"] == target_team].copy()

tab1, tab2, tab3, tab4 = st.tabs([
    "⭐ 総合", "🟢 攻撃プロセス", "🟣 守備プロセス", "🔴 クリティカル"
])

display_cols_all = [
    "player", "総合プロセス(①+②)", "総合クリティカル(③+④)",
    "①攻撃プロセス", "②守備プロセス", "③得点近接", "④失点近接"
]

with tab1:
    df_s = df_team[display_cols_all].sort_values("総合プロセス(①+②)", ascending=False)
    st.dataframe(
        df_s.style
        .background_gradient(subset=["総合プロセス(①+②)"], cmap="Blues")
        .background_gradient(subset=["総合クリティカル(③+④)"], cmap="Purples")
        .format({c: "{:.3f}" for c in df_s.columns if c != "player"}),
        use_container_width=True
    )

with tab2:
    df_s2 = df_team[["player", "①攻撃プロセス", "総合プロセス(①+②)"]].sort_values("①攻撃プロセス", ascending=False)
    st.dataframe(df_s2.style.background_gradient(cmap="YlGn").format(
        {c: "{:.3f}" for c in df_s2.columns if c != "player"}
    ), use_container_width=True)

with tab3:
    df_s3 = df_team[["player", "②守備プロセス", "総合プロセス(①+②)"]].sort_values("②守備プロセス", ascending=False)
    st.dataframe(df_s3.style.background_gradient(cmap="Blues").format(
        {c: "{:.3f}" for c in df_s3.columns if c != "player"}
    ), use_container_width=True)

with tab4:
    df_s4 = df_team[["player", "③得点近接", "④失点近接", "総合クリティカル(③+④)"]].sort_values("総合クリティカル(③+④)", ascending=False)
    st.dataframe(df_s4.style.background_gradient(subset=["③得点近接"], cmap="Oranges")
                 .background_gradient(subset=["④失点近接"], cmap="Greens")
                 .format({c: "{:.3f}" for c in df_s4.columns if c != "player"}),
                 use_container_width=True)

# =========================================================
# UI: 選手レーダーチャート
# =========================================================
st.markdown("---")
st.subheader("🕸️ 選手レーダーチャート（比較）")

all_players = sorted(df_team["player"].tolist())
selected_players = st.multiselect(
    "比較する選手を選択（2〜5名推奨）",
    all_players,
    default=all_players[:2] if len(all_players) >= 2 else all_players
)

radar_dims = ["①攻撃プロセス", "②守備プロセス", "③得点近接", "④失点近接", "総合クリティカル(③+④)"]

if selected_players:
    df_radar = df_team[df_team["player"].isin(selected_players)].set_index("player")

    # 0-1正規化
    df_norm = df_radar[radar_dims].copy()
    for col in radar_dims:
        mn, mx = df_norm[col].min(), df_norm[col].max()
        df_norm[col] = (df_norm[col] - mn) / (mx - mn + 1e-9)

    angles = np.linspace(0, 2 * np.pi, len(radar_dims), endpoint=False).tolist()
    angles += angles[:1]

    fig_radar, ax_r = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(selected_players)))

    for player, color in zip(selected_players, colors):
        if player not in df_norm.index:
            continue
        vals = df_norm.loc[player, radar_dims].tolist()
        vals += vals[:1]
        ax_r.plot(angles, vals, "o-", linewidth=2, label=player, color=color)
        ax_r.fill(angles, vals, alpha=0.15, color=color)

    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(radar_dims, size=9)
    ax_r.set_ylim(0, 1)
    ax_r.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax_r.set_title(f"{target_team} 選手レーダー", size=13, pad=15)
    plt.tight_layout()
    st.pyplot(fig_radar)

# =========================================================
# UI: 指標説明
# =========================================================
with st.expander("📖 指標の説明"):
    st.markdown("""
    | 指標 | 内容 | 算出方法 |
    |------|------|---------|
    | ① 攻撃プロセス | ゲームを有利に進める攻撃貢献 | xT利得（パス・キャリー）＋ファイナルサード・ボックス侵入ボーナス |
    | ② 守備プロセス | ゲームを有利に進める守備貢献 | インターセプト・デュエル勝利・クリアランス（エリア重み付き） |
    | ③ 得点近接 | 得点に直結した貢献 | xGチェーン（シューター50%・アシスト30%・前々パス20%） |
    | ④ 失点近接 | 失点阻止に直結した貢献 | セーブ(xG加重)・ブロック(xG加重) − ターンオーバー被xGペナルティ |
    | ⑤ Luck（偶然性） | 確率から乖離した得失点 | 実際の得点 − xG（決定力）/ 被xG − 失点（守備の幸運度） |
    
    **Z標準化**: ①②は試合内で全選手を標準化（相対評価）  
    **xT**: 各座標のゴール確率変化量（Transition Threat Model）
    """)
