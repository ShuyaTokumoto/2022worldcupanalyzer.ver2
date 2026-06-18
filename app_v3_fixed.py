"""
Football Analytics Dashboard v3.1
- Radar chart font fix: axis labels switched to English
- Explanations rewritten in plain Japanese for non-analysts
"""

import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from statsbombpy import sb

warnings.filterwarnings("ignore")

# =========================================================
# レーダー軸ラベル（英語で文字化け完全回避）
# 表示用の日本語はStreamlit側（st.markdown等）で行う
# =========================================================
RADAR_COLS = [
    "①攻撃プロセス",
    "②守備プロセス",
    "③得点近接",
    "④失点近接",
    "総合クリティカル(③+④)",
]
RADAR_LABELS_EN = [
    "① Attack\nProcess",
    "② Defense\nProcess",
    "③ Goal\nThreat",
    "④ Save\nContrib",
    "Critical\nTotal",
]

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
# コア分析関数
# =========================================================
def compute_raw_scores(match_id: int):
    events = sb.events(match_id=match_id)
    events = events.dropna(subset=["player"]).copy()

    teams = events["team"].dropna().unique()
    if len(teams) < 2:
        return None

    events["time_seconds"] = (
        events["minute"] * 60 + events["second"]
        + (events["period"] - 1) * 45 * 60
    )
    events = events.sort_values("time_seconds").reset_index(drop=True)

    players = events[["player", "team"]].drop_duplicates().set_index("player")
    scores = pd.DataFrame(index=players.index)
    scores["team"] = players["team"]
    scores["match_id"] = match_id

    pa  = dict.fromkeys(scores.index, 0.0)
    pd_ = dict.fromkeys(scores.index, 0.0)
    ca  = dict.fromkeys(scores.index, 0.0)
    cd  = dict.fromkeys(scores.index, 0.0)

    passes = events[events["type"] == "Pass"].copy()
    shots  = events[events["type"] == "Shot"].copy()

    # ① 攻撃プロセス
    for _, ev in passes.iterrows():
        try:
            sx, sy = ev["location"][0], ev["location"][1]
            ex, ey = ev["pass_end_location"][0], ev["pass_end_location"][1]
        except (TypeError, IndexError, KeyError):
            continue
        passer   = ev["player"]
        receiver = ev.get("pass_recipient")
        xt_gain  = get_xt(ex, ey) - get_xt(sx, sy)
        if pd.isna(ev.get("pass_outcome")):
            if xt_gain > 0:
                pa[passer] += xt_gain * 2.0
                if receiver in pa:
                    pa[receiver] += xt_gain * 8.0
            if sx < 80 <= ex:
                pa[passer] += 1.0
                if receiver in pa:
                    pa[receiver] += 0.5
            if ex >= 102 and 18 <= ey <= 62:
                pa[passer] += 1.5
                if receiver in pa:
                    pa[receiver] += 1.5
            if ev.get("under_pressure") is True and xt_gain > 0:
                pa[passer] += 0.5
        else:
            pa[passer] -= 3.0 if sx < 40 else (1.0 if sx < 60 else 0.3)

    carries = events[events["type"] == "Carry"].copy()
    for _, ev in carries.iterrows():
        try:
            sx, sy = ev["location"][0], ev["location"][1]
            ex, ey = ev["carry_end_location"][0], ev["carry_end_location"][1]
        except (TypeError, IndexError, KeyError):
            continue
        xt_gain = get_xt(ex, ey) - get_xt(sx, sy)
        if xt_gain > 0:
            pa[ev["player"]] += xt_gain * 15.0

    dribbles = events[events["type"] == "Dribble"]
    if "dribble_outcome" in dribbles.columns:
        for _, ev in dribbles[dribbles["dribble_outcome"] == "Complete"].iterrows():
            if ev["player"] in pa:
                try:
                    x = ev["location"][0]
                    pa[ev["player"]] += 1.5 if x >= 80 else 0.5
                except (TypeError, IndexError):
                    pa[ev["player"]] += 0.5

    for _, ev in events[events["type"] == "Foul Won"].iterrows():
        if ev["player"] in pa:
            try:
                x = ev["location"][0]
                pa[ev["player"]] += 0.5 if x >= 80 else 0.2
            except (TypeError, IndexError):
                pa[ev["player"]] += 0.2

    # ② 守備プロセス
    def def_zone_weight(location, action_type="default"):
        try:
            x = float(location[0])
        except (TypeError, IndexError, ValueError):
            x = 60.0
        if action_type == "interception":
            if x >= 80:   return 3.0
            elif x >= 60: return 2.0
            elif x >= 40: return 1.5
            else:         return 1.2
        elif action_type == "duel":
            if x >= 80:   return 2.5
            elif x >= 60: return 1.5
            elif x >= 40: return 0.8
            else:         return 0.3
        elif action_type == "clearance":
            if x < 20:    return 2.0
            elif x < 40:  return 0.5
            else:         return 0.8
        else:
            if x >= 80:   return 2.0
            elif x >= 40: return 1.0
            else:         return 0.6

    for _, ev in events[events["type"] == "Interception"].iterrows():
        if ev["player"] in pd_:
            pd_[ev["player"]] += def_zone_weight(ev.get("location"), "interception")
    for _, ev in events[events["type"] == "Ball Recovery"].iterrows():
        if ev["player"] in pd_:
            pd_[ev["player"]] += def_zone_weight(ev.get("location"), "recovery")
    duels = events[events["type"] == "Duel"]
    if "duel_outcome" in duels.columns:
        won_duels = duels[duels["duel_outcome"].isin(["Won", "Success", "Tackle"])]
        for _, ev in won_duels.iterrows():
            if ev["player"] in pd_:
                pd_[ev["player"]] += def_zone_weight(ev.get("location"), "duel")
    for _, ev in events[events["type"] == "Clearance"].iterrows():
        if ev["player"] in pd_:
            pd_[ev["player"]] += def_zone_weight(ev.get("location"), "clearance")
    pressures = events[events["type"] == "Pressure"]
    for _, ev in pressures.iterrows():
        if ev["player"] in pd_:
            try:
                if float(ev["location"][0]) >= 80:
                    pd_[ev["player"]] += 0.5
            except (TypeError, IndexError):
                pass

    # ③ 得点近接
    for idx in shots.index:
        shot = events.loc[idx]
        xg = shot.get("shot_statsbomb_xg", 0)
        if pd.isna(xg):
            xg = 0.0
        xg_adj = xg * (1.2 if shot.get("shot_body_part") in ["Head", "No Touch"] else 1.0)
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

    # ④ 失点近接
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
        if ev.get("goalkeeper_type") in ["Save", "Shot Saved"] and ev["player"] in cd:
            shot_near = shots[abs(shots["time_seconds"] - ev["time_seconds"]) < 2]
            xg_saved = shot_near["shot_statsbomb_xg"].fillna(0.1).sum()
            cd[ev["player"]] += max(xg_saved, 0.1)
    blocks = events[events["type"] == "Block"]
    for _, ev in blocks.iterrows():
        if ev["player"] in cd:
            shot_near = shots[abs(shots["time_seconds"] - ev["time_seconds"]) < 2]
            xg_blocked = shot_near["shot_statsbomb_xg"].fillna(0.05).sum()
            cd[ev["player"]] += max(xg_blocked * 0.8, 0.05)

    # チームサマリー
    xg_team = shots.groupby("team")["shot_statsbomb_xg"].sum()
    team_goals = {}
    for _, s in shots.iterrows():
        if s.get("shot_outcome") == "Goal":
            team_goals[s["team"]] = team_goals.get(s["team"], 0) + 1

    scores["①攻撃プロセス_raw"]     = pd.Series(pa)
    scores["②守備プロセス_raw"]     = pd.Series(pd_)
    scores["③得点近接"]             = pd.Series(ca)
    scores["④失点近接"]             = pd.Series(cd)
    scores["総合クリティカル(③+④)"] = scores["③得点近接"] + scores["④失点近接"]

    team_summaries = {}
    for t in events["team"].dropna().unique():
        opp_list = [x for x in events["team"].dropna().unique() if x != t]
        opp = opp_list[0] if opp_list else t
        xg_t = float(xg_team.get(t, 0))
        xg_o = float(xg_team.get(opp, 0))
        g_t  = team_goals.get(t, 0)
        g_o  = team_goals.get(opp, 0)
        tm   = scores[scores["team"] == t]
        team_summaries[t] = {
            "得点": g_t, "失点": g_o,
            "xG": xg_t, "被xG": xg_o,
            "攻撃クリティカル": float(tm["③得点近接"].sum()),
            "守備クリティカル": float(tm["④失点近接"].sum()),
            "クリティカル合計": float(tm["総合クリティカル(③+④)"].sum()),
            "⑤得点Luck(決定力)": g_t - xg_t,
            "⑤守備Luck(死守度)": xg_o - g_o,
        }

    return scores.reset_index(), team_summaries


# =========================================================
# 全試合プール標準化
# =========================================================
@st.cache_data(show_spinner="全64試合を事前分析中...（初回のみ約2分かかります）")
def build_pool():
    matches = sb.matches(competition_id=43, season_id=106)
    all_scores = []
    match_meta = {}

    for _, row in matches.iterrows():
        mid = int(row["match_id"])
        try:
            result = compute_raw_scores(mid)
            if result is None:
                continue
            df_s, team_sum = result
            all_scores.append(df_s)
            match_meta[mid] = {
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_score": int(row["home_score"]),
                "away_score": int(row["away_score"]),
                "match_date": row["match_date"],
                "team_summaries": team_sum,
            }
        except Exception:
            pass

    if not all_scores:
        return None, None

    pool = pd.concat(all_scores, ignore_index=True)

    for col_raw, col_norm in [
        ("①攻撃プロセス_raw", "①攻撃プロセス"),
        ("②守備プロセス_raw", "②守備プロセス"),
    ]:
        mu  = pool[col_raw].mean()
        std = pool[col_raw].std()
        pool[col_norm] = (pool[col_raw] - mu) / (std if std > 0 else 1.0)

    pool["総合プロセス(①+②)"] = pool["①攻撃プロセス"] + pool["②守備プロセス"]
    return pool, match_meta


# =========================================================
# ページ設定
# =========================================================
st.set_page_config(page_title="Football Analytics v3", layout="wide", page_icon="⚽")
st.title("⚽ 新5大指標 サッカー分析ダッシュボード v3")
st.markdown(
    "**① 攻め方の上手さ　② 守り方の上手さ　③ 得点への関与　④ 失点阻止への関与　⑤ 運の要素**  \n"
    "2022 FIFA ワールドカップ 全64試合｜全選手を同一基準で評価"
)

# =========================================================
# 素人向け指標説明（アプリ内固定表示）
# =========================================================
with st.expander("📖 5つの指標をわかりやすく説明（はじめての方はここを読んでください）", expanded=False):
    st.markdown("""
    ---
    ### ① 攻め方の上手さ（Attack Process）
    > **「ボールをゴールに近づける動きに、どれだけ貢献したか」**

    サッカーは「ボールをゴールに近い危険な場所へ運ぶ」ほど有利になります。  
    この指標では、**ドリブルで前に運んだ選手**や**危険なパスを受けた選手**を高く評価します。  
    ディフェンダーが安全な横パスを100本つないでも、この点は上がりません。

    ---
    ### ② 守り方の上手さ（Defense Process）
    > **「相手の攻撃をどれだけ積極的に潰したか」**

    ただ自陣で跳ね返すだけでなく、**相手陣地で積極的にボールを奪う**（ハイプレス）行動を高く評価します。  
    自陣でのクリアランスは低めの評価で、アグレッシブな守備かどうかで点数が変わります。

    ---
    ### ③ 得点への関与（Goal Threat）
    > **「実際にゴールが生まれそうな場面に、どれだけ絡んだか」**

    シュートを打った選手・その直前にパスを出した選手・その前にパスを出した選手、それぞれに点数が入ります。  
    点数の重さは「このシュートがゴールになる確率（xG）」に比例するため、決定機に絡んだ選手が高くなります。

    ---
    ### ④ 失点阻止への関与（Save Contribution）
    > **「ゴールを防ぐ直接的な行動をどれだけしたか」**

    キーパーのセーブ、ディフェンダーのシュートブロック、これらを「防いだシュートの危険度」に応じて評価します。  
    逆に、**自分のミスから相手のシュートを生んだ場合はマイナス**になります。  
    この指標で、「試合を決定づけたのは誰か」が一目でわかります（日本×ドイツ戦の権田選手など）。

    ---
    ### ⑤ 運の要素（Luck Score）
    > **「確率どおりなら何点のところを、何点取ったか・守ったか」**

    「xG（ゴール期待値）」という統計上の予測点数と、実際の得点の差です。  
    - **得点Luck がプラス** → 確率より多く決めた（決定力あり、または運が良かった）  
    - **守備Luck がプラス** → 確率より少ない失点で済んだ（好セーブ、または運が良かった）  

    これにより「実力で勝ったのか、運で勝ったのか」が分かるようになります。

    ---
    ### 数字の読み方（①②のみ）
    | スコア | 意味 |
    |--------|------|
    | **+2.0以上** | ワールドクラス（WC全体で上位2〜3%） |
    | **+1.0〜+2.0** | 非常に優秀（上位16%） |
    | **0.0付近** | WC平均的な選手 |
    | **-1.0以下** | 平均を大きく下回る |
    """)

# =========================================================
# データ読み込み
# =========================================================
with st.spinner("データを準備中..."):
    pool_df, match_meta = build_pool()

if pool_df is None:
    st.error("データの読み込みに失敗しました。")
    st.stop()

# =========================================================
# サイドバー
# =========================================================
st.sidebar.header("📋 メニュー")
page = st.sidebar.radio("ページを選択", ["🔍 試合分析", "🏆 全WC選手ランキング"])
st.sidebar.markdown("---")
st.sidebar.header("🔍 試合フィルター")
search_q = st.sidebar.text_input("国名で絞り込み (例: Japan)", "").lower()

labels = []
label_to_mid = {}
for mid, m in match_meta.items():
    label = (
        f"{m['match_date']} | "
        f"{m['home_team']} {m['home_score']} - "
        f"{m['away_score']} {m['away_team']}"
    )
    if not search_q or search_q in label.lower():
        labels.append(label)
        label_to_mid[label] = mid

if not labels:
    st.sidebar.warning("該当試合なし")
    st.stop()

selected_label = st.sidebar.selectbox("試合を選択", sorted(labels))
match_id = label_to_mid[selected_label]
meta     = match_meta[match_id]
home_t   = meta["home_team"]
away_t   = meta["away_team"]

df_match = pool_df[pool_df["match_id"] == match_id].copy()
team_sum = meta["team_summaries"]

for t in [home_t, away_t]:
    tm = df_match[df_match["team"] == t]
    team_sum[t]["プロセス合計(絶対)"] = float(tm["総合プロセス(①+②)"].sum())
    team_sum[t]["攻撃プロセス合計"]   = float(tm["①攻撃プロセス"].sum())
    team_sum[t]["守備プロセス合計"]   = float(tm["②守備プロセス"].sum())


# =========================================================
# ページ A: 試合分析
# =========================================================
if page == "🔍 試合分析":

    st.subheader(f"📊 {home_t}  vs  {away_t}")
    st.caption(f"スコア: {meta['home_score']} - {meta['away_score']}  |  {meta['match_date']}")

    col_h, col_sep, col_a = st.columns([5, 1, 5])

    def render_team_card(col, team, data):
        with col:
            g_for = data["得点"]
            g_ag  = data["失点"]
            result = "🏆" if g_for > g_ag else ("🤝" if g_for == g_ag else "❌")
            st.markdown(f"### {result} {team}")
            c1, c2, c3 = st.columns(3)
            c1.metric("得点", g_for)
            c2.metric("xG（期待値）", f"{data['xG']:.2f}",
                      delta=f"Luck {data['⑤得点Luck(決定力)']:+.2f}")
            c3.metric("被xG（被期待値）", f"{data['被xG']:.2f}",
                      delta=f"Luck {data['⑤守備Luck(死守度)']:+.2f}")

            lk_a = data["⑤得点Luck(決定力)"]
            lk_d = data["⑤守備Luck(死守度)"]
            st.markdown(
                f"🎯 **得点の運要素:** `{lk_a:+.2f}` "
                + ("✨確率超えゴール" if lk_a > 0.3 else "⚡確率どおり" if lk_a > -0.3 else "😬決定機を逃した")
            )
            st.markdown(
                f"🛡️ **守備の運要素:** `{lk_d:+.2f}` "
                + ("🧱確率より守れた" if lk_d > 0.3 else "⚡確率どおり" if lk_d > -0.3 else "💥確率より崩された")
            )
            proc = data.get("プロセス合計(絶対)", 0)
            crit = data["クリティカル合計"]
            st.info(
                f"試合の主導権（プロセス合計）: **{proc:+.2f}**  \n"
                f"決定的場面への関与（クリティカル）: **{crit:+.2f}**"
            )

    render_team_card(col_h, home_t, team_sum[home_t])
    with col_sep:
        st.markdown(
            "<div style='text-align:center;padding-top:80px;font-size:24px'>VS</div>",
            unsafe_allow_html=True
        )
    render_team_card(col_a, away_t, team_sum[away_t])

    # --- チーム比較チャート ---
    st.markdown("---")
    st.subheader("📈 チーム指標 対比")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    proc_cats_en = ["Attack\nProcess", "Defense\nProcess", "Process\nTotal"]
    proc_keys    = ["攻撃プロセス合計", "守備プロセス合計", "プロセス合計(絶対)"]
    x = np.arange(len(proc_cats_en))
    w = 0.35
    h_vals = [team_sum[home_t].get(k, 0) for k in proc_keys]
    a_vals = [team_sum[away_t].get(k, 0) for k in proc_keys]
    axes[0].bar(x - w/2, h_vals, w, label=home_t, color="#e74c3c", alpha=0.8)
    axes[0].bar(x + w/2, a_vals, w, label=away_t, color="#3498db", alpha=0.8)
    axes[0].axhline(0, color="gray", lw=0.8, ls="--")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(proc_cats_en, fontsize=9)
    axes[0].legend()
    axes[0].set_title("Process Metrics (Absolute Score)")

    crit_cats_en = ["Goal\nThreat", "Save\nContrib", "Critical\nTotal", "Goal\nLuck", "Defense\nLuck"]
    crit_keys    = ["攻撃クリティカル", "守備クリティカル", "クリティカル合計",
                    "⑤得点Luck(決定力)", "⑤守備Luck(死守度)"]
    x2 = np.arange(len(crit_cats_en))
    h_vals2 = [team_sum[home_t].get(k, 0) for k in crit_keys]
    a_vals2 = [team_sum[away_t].get(k, 0) for k in crit_keys]
    axes[1].bar(x2 - w/2, h_vals2, w, label=home_t, color="#e74c3c", alpha=0.8)
    axes[1].bar(x2 + w/2, a_vals2, w, label=away_t, color="#3498db", alpha=0.8)
    axes[1].axhline(0, color="gray", lw=0.8, ls="--")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(crit_cats_en, fontsize=9)
    axes[1].legend()
    axes[1].set_title("Critical & Luck Metrics")

    plt.tight_layout()
    st.pyplot(fig)

    # --- 選手個人スタッツ ---
    st.markdown("---")
    st.subheader("🏃 選手個人スタッツ")

    target_team = st.radio("チームを選択", [home_t, away_t], horizontal=True)
    df_team = df_match[df_match["team"] == target_team].copy()

    tab1, tab2, tab3, tab4 = st.tabs([
        "⭐ 全指標まとめ", "🟢 ① 攻め方の上手さ", "🔵 ② 守り方の上手さ", "🔴 ③④ 決定的場面"
    ])

    def fmt_df(df, sort_col):
        num_cols = [c for c in df.columns if c != "player"]
        return df.sort_values(sort_col, ascending=False).style \
            .background_gradient(subset=[sort_col], cmap="RdYlGn") \
            .format({c: "{:+.3f}" for c in num_cols})

    all_cols = [
        "player", "総合プロセス(①+②)", "①攻撃プロセス", "②守備プロセス",
        "総合クリティカル(③+④)", "③得点近接", "④失点近接"
    ]

    with tab1:
        st.caption(
            "スコアはWC全選手の平均を0とした絶対評価です。+1.0以上は全選手の上位16%に相当します。"
        )
        st.dataframe(fmt_df(df_team[all_cols], "総合プロセス(①+②)"), use_container_width=True)

    with tab2:
        st.caption(
            "ドリブルでの前進・危険なパスの受け取りを高評価。"
            "ディフェンダーの横パスは評価されません。"
        )
        st.dataframe(fmt_df(df_team[["player", "①攻撃プロセス"]], "①攻撃プロセス"), use_container_width=True)

    with tab3:
        st.caption(
            "相手陣内でのボール奪取（ハイプレス）を最高評価。"
            "自陣でのクリアランスだけでは高い点数になりません。"
        )
        st.dataframe(fmt_df(df_team[["player", "②守備プロセス"]], "②守備プロセス"), use_container_width=True)

    with tab4:
        st.caption(
            "③ 得点近接: シュートに絡んだ選手が高得点（正の値が高いほど良い）。  \n"
            "④ 失点近接: セーブ・ブロックした選手はプラス。自分のミスから失点につながった選手はマイナス。"
        )
        df_c = df_team[
            ["player", "③得点近接", "④失点近接", "総合クリティカル(③+④)"]
        ].sort_values("総合クリティカル(③+④)", ascending=False)
        st.dataframe(
            df_c.style
            .background_gradient(subset=["③得点近接"], cmap="Oranges")
            .background_gradient(subset=["④失点近接"], cmap="RdYlGn")
            .format({c: "{:+.3f}" for c in df_c.columns if c != "player"}),
            use_container_width=True
        )

    # --- レーダーチャート（文字化け修正版）---
    st.markdown("---")
    st.subheader("🕸️ 選手レーダーチャート")
    st.caption("外側ほど全WC選手の中で上位であることを示します（パーセンタイル表示）")

    all_p = sorted(df_team["player"].tolist())
    sel_p = st.multiselect(
        "比較する選手を選んでください（2〜5名推奨）", all_p,
        default=all_p[:3] if len(all_p) >= 3 else all_p
    )

    if sel_p:
        df_r = df_team[df_team["player"].isin(sel_p)].set_index("player")
        df_norm = df_r[RADAR_COLS].copy()
        for col in RADAR_COLS:
            pool_vals = pool_df[col].dropna()
            df_norm[col] = df_norm[col].apply(
                lambda v: float((pool_vals <= v).mean())
            )

        n = len(RADAR_LABELS_EN)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]

        fig_r, ax_r = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        colors = plt.cm.Set1(np.linspace(0, 0.8, len(sel_p)))

        for player, color in zip(sel_p, colors):
            if player not in df_norm.index:
                continue
            vals = df_norm.loc[player, RADAR_COLS].tolist()
            vals += vals[:1]
            ax_r.plot(angles, vals, "o-", lw=2, label=player, color=color)
            ax_r.fill(angles, vals, alpha=0.15, color=color)

        ax_r.set_xticks(angles[:-1])
        # 文字化け修正: matplotlib軸ラベルは英語のみ
        ax_r.set_xticklabels(RADAR_LABELS_EN, size=8)
        ax_r.set_ylim(0, 1)
        ax_r.set_yticks([0.25, 0.5, 0.75])
        ax_r.set_yticklabels(["25%ile", "50%ile", "75%ile"], size=7)
        ax_r.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1))
        # タイトルも英語（文字化け防止）＋ StreamlitのcaptionでフォローOK
        ax_r.set_title(f"{target_team} — WC Percentile Radar", size=11, pad=20)
        plt.tight_layout()
        st.pyplot(fig_r)

        # 軸の意味を日本語でフォロー（Streamlit側で表示するので文字化けなし）
        st.markdown("""
        **レーダー軸の意味:**  
        `① Attack Process` = 攻め方の上手さ　`② Defense Process` = 守り方の上手さ  
        `③ Goal Threat` = 得点への関与　`④ Save Contrib` = 失点阻止への関与　`Critical Total` = ③+④合計
        """)


# =========================================================
# ページ B: 全WC選手ランキング
# =========================================================
elif page == "🏆 全WC選手ランキング":

    st.subheader("🏆 2022 FIFA ワールドカップ 全選手ランキング（絶対評価）")
    st.info(
        "全64試合・全選手のスコアを同じ基準で比較しています。  \n"
        "複数試合に出場した選手は合計スコアが高くなります。  \n"
        "サイドバーでチーム名を入力すると絞り込めます。"
    )

    rank_tab1, rank_tab2, rank_tab3, rank_tab4 = st.tabs([
        "🥇 総合プロセス", "⚡ ① 攻め方", "🛡️ ② 守り方", "🎯 ③④ 決定場面"
    ])

    team_filter = st.sidebar.text_input("チーム名フィルター（全ランキング）", "")

    def get_ranked(col, top_n=50):
        df = pool_df[["player", "team", col]].copy()
        if team_filter:
            df = df[df["team"].str.lower().str.contains(team_filter.lower())]
        df_agg = df.groupby(["player", "team"])[col].sum().reset_index()
        df_agg = df_agg.sort_values(col, ascending=False).head(top_n)
        df_agg["rank"] = range(1, len(df_agg) + 1)
        return df_agg[["rank", "player", "team", col]]

    with rank_tab1:
        st.caption("攻め方＋守り方を合算したトータル貢献ランキング")
        df_rank = get_ranked("総合プロセス(①+②)")
        st.dataframe(
            df_rank.style
            .background_gradient(subset=["総合プロセス(①+②)"], cmap="RdYlGn")
            .format({"総合プロセス(①+②)": "{:+.3f}"}),
            use_container_width=True
        )

    with rank_tab2:
        st.caption("ドリブル前進・パスの受け手として危険地帯に絡んだ選手が上位")
        df_rank2 = get_ranked("①攻撃プロセス")
        st.dataframe(
            df_rank2.style
            .background_gradient(subset=["①攻撃プロセス"], cmap="YlGn")
            .format({"①攻撃プロセス": "{:+.3f}"}),
            use_container_width=True
        )

    with rank_tab3:
        st.caption("高い位置でのボール奪取・インターセプトを評価するランキング")
        df_rank3 = get_ranked("②守備プロセス")
        st.dataframe(
            df_rank3.style
            .background_gradient(subset=["②守備プロセス"], cmap="Blues")
            .format({"②守備プロセス": "{:+.3f}"}),
            use_container_width=True
        )

    with rank_tab4:
        st.caption("得点・失点の直接的な場面に関わった度合い（③＋④合計）")
        df_rank4 = get_ranked("総合クリティカル(③+④)")
        st.dataframe(
            df_rank4.style
            .background_gradient(subset=["総合クリティカル(③+④)"], cmap="Purples")
            .format({"総合クリティカル(③+④)": "{:+.3f}"}),
            use_container_width=True
        )
