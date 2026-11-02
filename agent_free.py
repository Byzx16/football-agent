"""
Football Analysis Agent v2 (100% gratis)
- Jadwal & form tim : football-data.org (gratis, 10 request/menit)
- Analisa AI + berita: Google Gemini API (free tier)
- Notifikasi        : Telegram Bot
- Pelacak akurasi   : predictions.csv + laporan

Pemakaian:
    python agent_free.py --test     # analisa 1 laga terdekat, kirim ke Telegram (tidak dicatat)
    python agent_free.py            # mode normal: kirim laga yang kick-off 45-90 menit lagi + catat prediksi
    python agent_free.py --report   # hitung akurasi prediksi yang sudah selesai, kirim ke Telegram
"""
import os, sys, json, re, time, csv, html, math, datetime as dt
import requests
from dotenv import load_dotenv

load_dotenv()

FOOTBALL_DATA_KEY = os.environ["FOOTBALL_DATA_KEY"]
GEMINI_KEY        = os.environ["GEMINI_API_KEY"]
TELEGRAM_TOKEN    = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID  = os.environ["TELEGRAM_CHAT_ID"]

GEMINI_MODEL = "gemini-2.5-flash"
LEAGUES = "PL,PD,SA,BL1,FL1,CL"
NOTIFY_WINDOW_MIN = (int(os.getenv("NOTIFY_MIN_MINUTES", 45)),
                     int(os.getenv("NOTIFY_MAX_MINUTES", 90)))   # kirim saat kick-off tinggal sekian menit
MAX_MATCHES_PER_RUN = 8          # maksimal laga yang dianalisa dalam satu run
SENT_FILE = "sent.json"
PRED_FILE = "predictions.csv"
PARLAY_HOUR_WIB = int(os.getenv("PARLAY_HOUR_WIB", 6))   # jam kombinasi harian dikirim (WIB)
PARLAY_FILE = "parlays.json"
PARLAY_LAST_FILE = "last_parlay.txt"
PARLAY_SHORTLIST = 5         # jumlah laga yang dianalisa AI untuk kombinasi (hemat kuota Gemini)
PARLAY_MIN_PROB = 55         # peluang minimal sebuah pilihan supaya boleh masuk kombinasi (%)
BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
CREST_FILE = "crests.json"
ANALYSIS_FILE = "analyses.json"
REPORT_FILE = "last_report.txt"
REPORT_HOUR_WIB = 9                           # laporan harian dikirim di run pertama setelah jam ini (WIB)
GOALS_FILE = "goals_cache.json"
GOAL_PRIOR = 8.0                 # kekuatan prior model gol (setara kira-kira 6 laga)
PARLAY_MIN_GOAL_PROB = 62        # peluang minimal pasar gol (over/under) supaya boleh masuk kombinasi (%)
ELO_FILE = "elo_cache.json"
ELO_CACHE_HOURS = 12
ELO_K, ELO_HFA, ELO_CARRYOVER = 20, 65, 0.7   # K-factor, keunggulan kandang, sisa rating antar musim
W_GEMINI = 0.5                                # bobot AI di probabilitas gabungan (sisanya Elo)
PRED_FIELDS = ["match_id", "kickoff_utc", "competition", "home", "away",
               "p_home", "p_draw", "p_away", "pick", "confidence",
               "likely_score", "outcome", "final_score", "correct",
               "g_home", "g_draw", "g_away", "e_home", "e_draw", "e_away",
               "tg_msg_id", "home_s", "away_s",
               "lam_h", "lam_a", "o15", "o25", "o35", "btts"]

FD = "https://api.football-data.org/v4"
FD_HEADERS = {"X-Auth-Token": FOOTBALL_DATA_KEY}
_last_fd_call = 0.0


# =====================================================================
# football-data.org (dibatasi 10 request/menit)
# =====================================================================
def fd_get(path, params=None):
    global _last_fd_call
    wait = 6.5 - (time.time() - _last_fd_call)
    if wait > 0:
        time.sleep(wait)
    _last_fd_call = time.time()
    r = requests.get(f"{FD}{path}", headers=FD_HEADERS, params=params, timeout=30)
    if r.status_code != 200:
        print(f"[football-data] {path} -> {r.status_code}")
        return None
    return r.json()

def load_sent():
    try:
        return set(json.load(open(SENT_FILE)))
    except Exception:
        return set()

def save_sent(s):
    json.dump(sorted(s), open(SENT_FILE, "w"))

def upcoming_matches(test_mode=False):
    now = dt.datetime.now(dt.timezone.utc)
    data = fd_get("/matches", {
        "dateFrom": now.date().isoformat(),
        "dateTo": (now + dt.timedelta(days=2)).date().isoformat(),
        "competitions": LEAGUES,
    })
    if not data:
        return []
    out = []
    for m in data.get("matches", []):
        if m["status"] not in ("TIMED", "SCHEDULED"):
            continue
        ko = dt.datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
        mins = (ko - now).total_seconds() / 60
        if mins <= 0:
            continue
        if test_mode or NOTIFY_WINDOW_MIN[0] <= mins <= NOTIFY_WINDOW_MIN[1]:
            out.append((mins, m))
    out.sort(key=lambda x: x[0])
    return [m for _, m in out]

_standings_cache = {}
def standings(comp_code):
    if comp_code in _standings_cache:
        return _standings_cache[comp_code]
    data = fd_get(f"/competitions/{comp_code}/standings")
    result = {}
    if data and data.get("standings"):
        for t in data["standings"][0]["table"]:
            result[t["team"]["id"]] = (
                f'#{t["position"]} ({t["points"]} poin, selisih gol {t["goalDifference"]})'
            )
    _standings_cache[comp_code] = result
    return result

def team_form(team_id, n=5):
    data = fd_get(f"/teams/{team_id}/matches", {"status": "FINISHED", "limit": n})
    if not data:
        return []
    rows = []
    for m in data.get("matches", [])[-n:]:
        s = m["score"]["fullTime"]
        rows.append(f'{m["homeTeam"]["shortName"]} {s["home"]}-{s["away"]} {m["awayTeam"]["shortName"]}')
    return rows


# =====================================================================
# Model Elo: "pendapat kedua" yang stabil, dihitung dari hasil pertandingan
# =====================================================================
def _elo_update(R, m):
    s = (m.get("score") or {}).get("fullTime") or {}
    gh, ga = s.get("home"), s.get("away")
    if gh is None or ga is None:
        return
    h, a = m["homeTeam"]["id"], m["awayTeam"]["id"]
    rh, ra = R.get(h, 1500.0), R.get(a, 1500.0)
    exp_h = 1 / (1 + 10 ** (-((rh + ELO_HFA) - ra) / 400))
    act = 1.0 if gh > ga else 0.5 if gh == ga else 0.0
    gd = abs(gh - ga)
    mult = 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
    delta = ELO_K * mult * (act - exp_h)
    R[h], R[a] = rh + delta, ra - delta

def build_ratings():
    """Rating Elo dari cache (<12 jam) atau dihitung ulang dari hasil musim lalu + musim ini."""
    now = dt.datetime.now(dt.timezone.utc)
    try:
        c = json.load(open(ELO_FILE))
        built = dt.datetime.fromisoformat(c["built_utc"])
        if now - built < dt.timedelta(hours=ELO_CACHE_HOURS) and c.get("ratings") and _load_goals():
            return {int(k): v for k, v in c["ratings"].items()}
    except Exception:
        pass
    print("[elo] menghitung rating dari hasil pertandingan (sekitar 1-2 menit)...")
    cur = now.year if now.month >= 7 else now.year - 1
    R, allm = {}, []
    for season in (cur - 1, cur):
        ms = []
        for code in LEAGUES.split(","):
            data = fd_get(f"/competitions/{code}/matches", {"season": season, "status": "FINISHED"})
            if data:
                ms += data.get("matches", [])
        ms.sort(key=lambda m: m["utcDate"])
        allm += [((1.0 if season == cur else 0.5), m) for m in ms]
        for m in ms:
            _elo_update(R, m)
        if season == cur - 1:   # antar musim, rating ditarik sebagian ke rata-rata
            R = {k: 1500 + (v - 1500) * ELO_CARRYOVER for k, v in R.items()}
    try:
        _build_goals(allm)
    except Exception as e:
        print("[gol] gagal membangun model gol ->", e)
    if not R:
        return {}
    with open(ELO_FILE, "w") as f:
        json.dump({"built_utc": now.isoformat(), "ratings": {str(k): v for k, v in R.items()}}, f)
    return R

GOALS = None

def _load_goals():
    global GOALS
    try:
        with open(GOALS_FILE, encoding="utf-8") as f:
            c = json.load(f)
        if c.get("teams") and c.get("avg"):
            GOALS = {"avg": c["avg"], "teams": {int(i): tuple(v) for i, v in c["teams"].items()}}
            return GOALS
    except Exception:
        pass
    return None

def _build_goals(allm):
    """Model gol (Poisson): kekuatan serang dan bertahan tiap tim relatif terhadap rata-rata kompetisinya."""
    global GOALS
    comp = {}
    for w, m in allm:
        ft = (m.get("score") or {}).get("fullTime") or {}
        if ft.get("home") is None or ft.get("away") is None:
            continue
        t = comp.setdefault(m["competition"]["code"], [0.0, 0.0, 0.0])
        t[0] += w * ft["home"]
        t[1] += w * ft["away"]
        t[2] += w
    avg = {c: {"home": t[0] / t[2], "away": t[1] / t[2]} for c, t in comp.items() if t[2] > 0}
    team = {}
    for w, m in allm:
        ft = (m.get("score") or {}).get("fullTime") or {}
        a = avg.get(m["competition"]["code"])
        if not a or ft.get("home") is None or ft.get("away") is None:
            continue
        th = team.setdefault(m["homeTeam"]["id"], [0.0] * 4)
        ta = team.setdefault(m["awayTeam"]["id"], [0.0] * 4)
        th[0] += w * ft["home"]; th[1] += w * ft["away"]; th[2] += w * a["home"]; th[3] += w * a["away"]
        ta[0] += w * ft["away"]; ta[1] += w * ft["home"]; ta[2] += w * a["away"]; ta[3] += w * a["home"]
    k = GOAL_PRIOR
    teams = {str(i): [(v[0] + k) / (v[2] + k), (v[1] + k) / (v[3] + k)] for i, v in team.items()}
    if not teams:
        return
    with open(GOALS_FILE, "w", encoding="utf-8") as f:
        json.dump({"built_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "avg": avg, "teams": teams}, f)
    GOALS = {"avg": avg, "teams": {int(i): tuple(v) for i, v in teams.items()}}

def goal_probs(hid, aid, comp_code):
    """Perkiraan gol dan peluang over/under, kedua tim cetak gol, serta skor paling mungkin (Poisson)."""
    g = GOALS or _load_goals()
    if not g or hid not in g["teams"] or aid not in g["teams"]:
        return None
    avg = g["avg"].get(comp_code) or {"home": 1.5, "away": 1.2}
    att_h, def_h = g["teams"][hid]
    att_a, def_a = g["teams"][aid]
    lh = max(0.2, min(4.5, avg["home"] * att_h * def_a))
    la = max(0.2, min(4.5, avg["away"] * att_a * def_h))
    n = 11
    ph = [math.exp(-lh) * lh ** i / math.factorial(i) for i in range(n)]
    pa = [math.exp(-la) * la ** i / math.factorial(i) for i in range(n)]
    grid = {(i, j): ph[i] * pa[j] for i in range(n) for j in range(n)}
    norm = sum(grid.values())
    pct = lambda f: sum(p for (i, j), p in grid.items() if f(i, j)) / norm * 100
    top = sorted(grid.items(), key=lambda kv: -kv[1])[:3]
    return {
        "lam_h": lh, "lam_a": la,
        "o15": pct(lambda i, j: i + j >= 2), "o25": pct(lambda i, j: i + j >= 3),
        "o35": pct(lambda i, j: i + j >= 4), "btts": pct(lambda i, j: i >= 1 and j >= 1),
        "top": [(f"{i}-{j}", p / norm * 100) for (i, j), p in top],
    }

def _total_goals(r):
    try:
        h, a = r["final_score"].split("-")
        return int(h) + int(a)
    except Exception:
        return None

def _both_scored(r):
    try:
        h, a = r["final_score"].split("-")
        return int(h) > 0 and int(a) > 0
    except Exception:
        return False

def goal_lean(gp):
    """Sinyal gol terkuat dari model: pasar over/under dengan peluang tertinggi."""
    opts = [("Over 1.5", gp["o15"]), ("Under 1.5", 100 - gp["o15"]), ("Over 2.5", gp["o25"]),
            ("Under 2.5", 100 - gp["o25"]), ("Over 3.5", gp["o35"]), ("Under 3.5", 100 - gp["o35"])]
    return max(opts, key=lambda x: x[1])

def elo_predict(R, hid, aid):
    """Probabilitas (home, seri, away) dalam persen, atau None kalau salah satu tim belum punya rating."""
    if hid not in R or aid not in R:
        return None
    e = 1 / (1 + 10 ** (-((R[hid] + ELO_HFA) - R[aid]) / 400))
    d = 0.28 * (1 - abs(2 * e - 1)) ** 0.6          # peluang seri mengecil kalau selisih kekuatan besar
    ph, pa = max(0.01, e - d / 2), max(0.01, 1 - e - d / 2)
    tot = ph + d + pa
    return (ph / tot * 100, d / tot * 100, pa / tot * 100)

def elo_only(elo):
    """Sinyal cadangan kalau analisa AI gagal: hanya dari model Elo."""
    res = normalize({
        "home_win": elo[0], "draw": elo[1], "away_win": elo[2],
        "key_factors": ["Analisa AI tidak tersedia (kuota atau gangguan), sinyal ini hanya dari model Elo, yaitu kekuatan tim berdasarkan hasil pertandingan."],
        "risks": ["Tanpa info berita, cedera, dan lineup. Perlakukan dengan lebih hati-hati."],
        "likely_score": "-", "summary": "Hanya berdasarkan rating Elo, tanpa berita terbaru.",
    })
    res["e"] = elo
    res["ai_failed"] = True
    return res

def combine(res, elo):
    """Gabungkan probabilitas AI dan Elo. Kalau Elo tidak tersedia, pakai AI saja."""
    res["g"] = (res["home_win"], res["draw"], res["away_win"])
    if elo:
        res["e"] = elo
        mix = [W_GEMINI * g + (1 - W_GEMINI) * e for g, e in zip(res["g"], elo)]
        res["home_win"], res["draw"], res["away_win"] = mix
        res = normalize(res)
    return res


# =====================================================================
# Gemini
# =====================================================================
SYSTEM = """Kamu analis sepak bola kuantitatif. Tugasmu menganalisa SATU pertandingan.
Jika tersedia pencarian web, gunakan untuk: berita terbaru, cedera/skorsing, kemungkinan lineup,
jadwal padat, motivasi (zona gelar/degradasi), head-to-head, kondisi pelatih.
Aturan:
- Beri probabilitas jujur (home_win + draw + away_win = 100). Jangan terlalu yakin.
- Pastikan klaim konsisten satu sama lain (mis. posisi klasemen harus cocok dengan data yang diberikan).
- Data klasemen dan form dari pengguna adalah acuan utama; kalau hasil pencarian web bertentangan, sebutkan keraguan.
- Sebut apa yang BELUM diketahui (mis. lineup belum rilis).
- Jangan mengarang fakta. Kalau tidak menemukan info, katakan tidak ada info.
- key_factors: maksimal 4 poin singkat (1 kalimat). risks: maksimal 3 poin singkat.
Balas HANYA JSON valid (tanpa markdown) dengan format:
{"home_win": int, "draw": int, "away_win": int,
 "key_factors": ["...", "..."], "risks": ["...", "..."],
 "likely_score": "x-y", "summary": "2-3 kalimat Bahasa Indonesia"}"""

def gemini_call(prompt, use_search=True):
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 8192,
            "thinkingConfig": {"thinkingBudget": 1024},
        },
    }
    if use_search:
        body["tools"] = [{"google_search": {}}]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    return requests.post(url, headers={"x-goog-api-key": GEMINI_KEY,
                                       "Content-Type": "application/json"},
                         json=body, timeout=120)

def analyze(match, ctx):
    h, a = match["homeTeam"]["name"], match["awayTeam"]["name"]
    prompt = f"""Pertandingan: {h} (kandang) vs {a} (tandang)
Kompetisi: {match['competition']['name']}
Kick-off (UTC): {match['utcDate']}

Posisi klasemen (data resmi): {h}: {ctx['home_rank']} | {a}: {ctx['away_rank']}
Form 5 laga terakhir {h}: {ctx['home_form']}
Form 5 laga terakhir {a}: {ctx['away_form']}

Berikan analisa dalam format JSON."""

    r = gemini_call(prompt, use_search=True)
    if r.status_code != 200:
        print(f"[gemini] dengan search gagal ({r.status_code}): {r.text[:200]}")
        r = gemini_call(prompt, use_search=False)
    if r.status_code == 429:
        print("[gemini] kena batas kuota, tunggu 45 detik lalu coba sekali lagi...")
        time.sleep(45)
        r = gemini_call(prompt, use_search=False)
    if r.status_code != 200:
        raise RuntimeError(f"Gemini error {r.status_code}: {r.text[:300]}")
    data = r.json()
    cands = data.get("candidates") or []
    if not cands:
        raise RuntimeError(f"Gemini tidak memberi kandidat jawaban: {json.dumps(data)[:500]}")
    cand = cands[0]
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise RuntimeError(
            f"Tidak ada JSON di balasan Gemini. finishReason={cand.get('finishReason')} "
            f"balasan={text[:500]!r}"
        )
    return normalize(json.loads(m.group(0)))

def normalize(res):
    """Rapikan probabilitas jadi total 100, lalu tentukan pick & keyakinan dari angkanya
    (bukan dari klaim model) supaya konsisten."""
    keys = ("home_win", "draw", "away_win")
    vals = []
    for k in keys:
        try:
            vals.append(max(0.0, float(res.get(k, 0))))
        except (TypeError, ValueError):
            vals.append(0.0)
    total = sum(vals)
    vals = [100 / 3] * 3 if total == 0 else [v * 100 / total for v in vals]
    vals = [round(v) for v in vals]
    vals[vals.index(max(vals))] += 100 - sum(vals)

    labels = ["home", "draw", "away"]
    order = sorted(range(3), key=lambda i: -vals[i])
    gap = vals[order[0]] - vals[order[1]]
    res["home_win"], res["draw"], res["away_win"] = vals
    res["pick"] = labels[order[0]]
    res["confidence"] = "rendah" if gap < 8 else "sedang" if gap < 20 else "tinggi"
    return res


# =====================================================================
# Telegram: tampilan pesan
# =====================================================================
def _tg_error(r):
    try:
        return r.json().get("description", r.text)
    except Exception:
        return r.text

def send_telegram(text, reply_to=None):
    """Kirim pesan ke Telegram dan kembalikan message_id-nya.
    reply_to = message_id pesan yang mau dibalas (tampil sebagai reply)."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    base = {"chat_id": TELEGRAM_CHAT_ID, "disable_web_page_preview": True}
    if reply_to:
        base["reply_parameters"] = {"message_id": int(reply_to), "allow_sending_without_reply": True}
    r = requests.post(url, json={**base, "text": text, "parse_mode": "HTML"}, timeout=30)
    if r.status_code == 200:
        return r.json()["result"]["message_id"]
    print(f"[telegram] gagal kirim (HTML): {r.status_code} - {_tg_error(r)}")
    plain = html.unescape(re.sub(r"<[^>]+>", "", text))
    r2 = requests.post(url, json={**base, "text": plain}, timeout=30)
    if r2.status_code != 200:
        raise RuntimeError(f"Telegram menolak pesan: {r2.status_code} - {_tg_error(r2)}")
    return r2.json()["result"]["message_id"]

def _tgl(d):
    return f"{d.day} {BULAN[d.month - 1]} {d.year}"

def bar(p, width=10):
    f = max(0, min(width, round(p / 100 * width)))
    return "█" * f + "░" * (width - f)

CONF_ICON = {"rendah": "🟡", "sedang": "🟠", "tinggi": "🟢"}
LINE = "━━━━━━━━━━━━━━━━━━"

def format_msg(match, res):
    esc = lambda x: html.escape(str(x))
    h_full, a_full = match["homeTeam"]["name"], match["awayTeam"]["name"]
    h = (match["homeTeam"].get("shortName") or h_full)[:14]
    a = (match["awayTeam"].get("shortName") or a_full)[:14]
    ko = dt.datetime.fromisoformat(match["utcDate"].replace("Z", "+00:00"))
    ko_wib = ko.astimezone(dt.timezone(dt.timedelta(hours=7)))

    rows = [("🟢", h, res["home_win"]), ("⚪", "Seri", res["draw"]), ("🔴", a, res["away_win"])]
    w = max(len(n) for _, n, _ in rows)
    prob_lines = "\n".join(
        f"{ic} <code>{esc(n.ljust(w))} {bar(p)} {p:>3}%</code>" for ic, n, p in rows
    )

    title, models = "PROBABILITAS", ""
    if res.get("ai_failed"):
        title = "PROBABILITAS (Elo saja, AI sedang tidak tersedia)"
    elif res.get("e") and res.get("g"):
        g, e = res["g"], res["e"]
        title = "PROBABILITAS (gabungan AI + Elo)"
        models = (f"<code>🤖 AI {g[0]:.0f}/{g[1]:.0f}/{g[2]:.0f}   "
                  f"📐 Elo {e[0]:.0f}/{e[1]:.0f}/{e[2]:.0f}</code>\n")

    pick_name = {"home": h_full, "draw": "Seri", "away": a_full}[res["pick"]]
    conf = res["confidence"]
    if conf == "rendah":
        signal = f"⚖️ <b>Laga ketat, tanpa sinyal kuat</b>\n    Sedikit condong: {esc(pick_name)}"
    else:
        signal = f"🎯 <b>Sinyal: {esc(pick_name)}{' menang' if res['pick'] != 'draw' else ''}</b>"

    goals_block, gp = "", res.get("goals")
    if gp:
        lean_name, lean_p = goal_lean(gp)
        goals_block = (
            f"⚽ <b>GOL</b>  <i>(model statistik)</i>\n"
            f"Perkiraan gol: {esc(h)} <b>{gp['lam_h']:.1f}</b> – <b>{gp['lam_a']:.1f}</b> {esc(a)}\n"
            f"📈 Over 1.5: <b>{gp['o15']:.0f}%</b>   Over 2.5: <b>{gp['o25']:.0f}%</b>   Over 3.5: <b>{gp['o35']:.0f}%</b>\n"
            f"🤝 Kedua tim cetak gol: <b>{gp['btts']:.0f}%</b>\n"
            f"🔢 Skor paling mungkin: " + ", ".join(f"{sc} ({pp:.0f}%)" for sc, pp in gp["top"]) + "\n"
            f"🎯 Sinyal gol terkuat: <b>{lean_name}</b> ({lean_p:.0f}%)\n{LINE}\n"
        )
    factors = "\n".join(f"✅ {esc(x)}" for x in res.get("key_factors", [])[:4])
    risks = "\n".join(f"⚠️ {esc(x)}" for x in res.get("risks", [])[:3])

    return (
        f"⚽ <b>{esc(h_full)} vs {esc(a_full)}</b>\n"
        f"🏆 {esc(match['competition']['name'])}  •  🕐 {ko_wib.day} {BULAN[ko_wib.month - 1]}, {ko_wib:%H:%M} WIB\n"
        f"{LINE}\n"
        f"📊 <b>{title}</b>\n{prob_lines}\n{models}"
        f"{LINE}\n"
        f"{signal}\n"
        f"{CONF_ICON[conf]} Keyakinan: <b>{conf}</b>   🔢 Skor mungkin: <b>{esc(res.get('likely_score', '-'))}</b>\n"
        f"{LINE}\n"
        f"{goals_block}"
        f"💡 <b>FAKTOR KUNCI</b>\n{factors}\n\n"
        f"🚧 <b>RISIKO</b>\n{risks}\n"
        f"{LINE}\n"
        f"📝 <b>RINGKASAN</b>\n{esc(res.get('summary', ''))}\n"
        f"{LINE}\n"
        f"{track_record_line()}\n\n"
        f"<i>Analisa probabilistik, bukan jaminan hasil.</i>"
    )


# =====================================================================
# Pelacak akurasi
# =====================================================================
def read_preds():
    if not os.path.exists(PRED_FILE):
        return []
    with open(PRED_FILE, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write_preds(rows):
    with open(PRED_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PRED_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in PRED_FIELDS})

def track_record_line():
    """Satu baris rekor agent untuk ditempel di setiap pesan."""
    done = [r for r in read_preds() if r["outcome"]]
    if not done:
        return "📈 <b>Rekor agent:</b> belum ada hasil (baru mulai dicatat)"
    n = len(done)
    ok = sum(r["correct"] == "1" for r in done)
    line = f"📈 <b>Rekor agent:</b> {ok}/{n} benar ({ok / n * 100:.0f}%)"
    strong = [r for r in done if r["confidence"] in ("sedang", "tinggi")]
    if strong:
        sok = sum(r["correct"] == "1" for r in strong)
        line += f"\n🎯 Sinyal kuat: {sok}/{len(strong)} ({sok / len(strong) * 100:.0f}%)"
    if n < 30:
        line += "\n<i>(sampel masih kecil, belum bisa disimpulkan)</i>"
    return line

def _save_analysis(match_id, res):
    """Simpan teks analisa (ringkasan, faktor kunci, risiko) agar bisa dibaca di dashboard."""
    try:
        with open(ANALYSIS_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    data[str(match_id)] = {
        "summary": res.get("summary", ""),
        "factors": list(res.get("key_factors") or [])[:4],
        "risks": list(res.get("risks") or [])[:3],
    }
    for k in list(data)[:-400]:      # simpan 400 analisa terakhir saja
        del data[k]
    with open(ANALYSIS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def log_prediction(match, res, msg_id=None):
    rows = read_preds()
    if any(r["match_id"] == str(match["id"]) for r in rows):
        return
    try:
        _save_analysis(match["id"], res)
    except Exception as e:
        print("[analisa] gagal menyimpan ->", e)
    rows.append({
        "match_id": match["id"],
        "kickoff_utc": match["utcDate"],
        "competition": match["competition"]["name"],
        "home": match["homeTeam"]["name"],
        "away": match["awayTeam"]["name"],
        "home_s": match["homeTeam"].get("shortName") or "",
        "away_s": match["awayTeam"].get("shortName") or "",
        "tg_msg_id": msg_id or "",
        "p_home": res["home_win"], "p_draw": res["draw"], "p_away": res["away_win"],
        "pick": res["pick"], "confidence": res["confidence"],
        "likely_score": res.get("likely_score", ""),
        "outcome": "", "final_score": "", "correct": "",
        **{k: round(v, 1) for k, v in zip(("g_home", "g_draw", "g_away"), res.get("g") or ())},
        **{k: round(v, 1) for k, v in zip(("e_home", "e_draw", "e_away"), res.get("e") or ())},
        **({k: round(res["goals"][k], 1) for k in ("lam_h", "lam_a", "o15", "o25", "o35", "btts")} if res.get("goals") else {}),
    })
    write_preds(rows)

def _nm(r, side):
    """Nama tim singkat untuk tampilan (pakai nama pendek kalau ada)."""
    name = r.get(f"{side}_s") or r.get(side) or "?"
    return name if len(name) <= 18 else name[:17] + "…"

def _pick_label(r, key):
    return {"home": _nm(r, "home"), "away": _nm(r, "away"), "draw": "Seri"}[key]

def format_result_msg(r):
    """Pesan FULL TIME yang dikirim sebagai reply ke pesan analisa."""
    esc = lambda x: html.escape(str(x))
    h, a = esc(_nm(r, "home")), esc(_nm(r, "away"))
    try:
        hs, as_ = r["final_score"].split("-")
    except Exception:
        hs, as_ = "?", "?"
    ok = r["correct"] == "1"
    names = {"home": f"{h} menang", "away": f"{a} menang", "draw": "Seri"}
    lean = "condong ke " if r["confidence"] == "rendah" and r["pick"] != "draw" else ""
    prob = {"home": r["p_home"], "draw": r["p_draw"], "away": r["p_away"]}[r["outcome"]]
    verdict = "✅ <b>TEBAKAN BENAR</b>" if ok else "❌ <b>TEBAKAN MELESET</b>"
    if not ok and r["confidence"] == "rendah":
        verdict += "  <i>(sinyal lemah)</i>"
    lines = [
        f"🏁 <b>FULL TIME</b>  •  {esc(r['competition'])}",
        LINE,
        f"⚽ <b>{h}</b>  <code>{esc(hs)} – {esc(as_)}</code>  <b>{a}</b>",
        LINE,
        verdict,
        f"🎯 Prediksi: {lean}{names[r['pick']]} (keyakinan {esc(r['confidence'])})",
        f"🏆 Hasil: {names[r['outcome']]}",
        f"📊 Peluang yang kami beri untuk hasil ini: {float(prob):.0f}%",
    ]
    ls = (r.get("likely_score") or "").strip()
    if ls and ls != "-":
        lines.append(f"🔢 Skor tebakan {esc(ls)} → aktual {esc(r['final_score'])}"
                     + ("  🎯 <b>tepat!</b>" if ls == r["final_score"] else ""))
    tg = _total_goals(r)
    if tg is not None and r.get("o25"):
        both = _both_scored(r)

        def g_line(label, p, hit):
            ok = (float(p) >= 50) == hit
            return f"{'✅' if ok else '❌'} {label}: model {float(p):.0f}%, {'terjadi' if hit else 'tidak terjadi'}"
        lines += [LINE, f"⚽ <b>Total gol: {tg}</b>",
                  g_line("Over 1.5", r["o15"], tg >= 2), g_line("Over 2.5", r["o25"], tg >= 3),
                  g_line("Over 3.5", r["o35"], tg >= 4), g_line("Kedua tim cetak gol", r["btts"], both)]
    lines += [LINE, track_record_line()]
    return "\n".join(lines)

RESULT_DELAY_MIN = 105   # laga paling cepat selesai sekitar 105 menit setelah kick-off

def settle_results(max_calls=3, notify=True):
    """Cek hasil akhir laga yang sudah dicatat. Begitu ada yang selesai, isi hasilnya
    dan (kalau notify) kirim pesan FULL TIME sebagai reply ke pesan analisanya."""
    rows = read_preds()
    now = dt.datetime.now(dt.timezone.utc)
    winner_map = {"HOME_TEAM": "home", "AWAY_TEAM": "away", "DRAW": "draw"}

    pending = []
    for r in rows:
        if r["outcome"]:
            continue
        ko = dt.datetime.fromisoformat(r["kickoff_utc"].replace("Z", "+00:00"))
        if now - ko >= dt.timedelta(minutes=RESULT_DELAY_MIN):
            pending.append((r, ko))
    if not pending:
        return 0

    # 1 panggilan API untuk semua laga dalam 8 hari terakhir
    recent = [(r, ko) for r, ko in pending if now - ko <= dt.timedelta(days=8)]
    found, bulk_ok = {}, False
    if recent:
        d0 = min(ko for _, ko in recent).date()
        d1 = max(ko for _, ko in recent).date() + dt.timedelta(days=1)
        data = fd_get("/matches", {"dateFrom": d0.isoformat(), "dateTo": d1.isoformat(),
                                   "competitions": LEAGUES})
        if data:
            bulk_ok = True
            found = {m["id"]: m for m in data.get("matches", [])}

    calls, newly = 0, []
    for r, ko in pending:
        mid = int(r["match_id"])
        m = found.get(mid)
        if m is None and calls < max_calls and (not bulk_ok or now - ko > dt.timedelta(days=8)):
            data = fd_get(f"/matches/{mid}")
            calls += 1
            m = (data.get("match", data) if data else None)
        if not m or m.get("status") != "FINISHED":
            continue
        score = m.get("score") or {}
        winner, ft = score.get("winner"), score.get("fullTime") or {}
        if winner not in winner_map or ft.get("home") is None:
            continue
        r["outcome"] = winner_map[winner]
        r["final_score"] = f'{ft["home"]}-{ft["away"]}'
        r["correct"] = "1" if r["pick"] == r["outcome"] else "0"
        newly.append(r)

    if newly:
        write_preds(rows)
        if notify:
            for r in newly:
                try:
                    send_telegram(format_result_msg(r), reply_to=r.get("tg_msg_id") or None)
                except Exception as e:
                    print("Gagal kirim hasil akhir ->", e)
    return len(newly)

def _brier(rows, pref):
    tot = 0.0
    for r in rows:
        p = [float(r[f"{pref}_{k}"]) / 100 for k in ("home", "draw", "away")]
        o = [1.0 if r["outcome"] == k else 0.0 for k in ("home", "draw", "away")]
        tot += sum((x - y) ** 2 for x, y in zip(p, o))
    return tot / len(rows)

def build_report():
    rows = read_preds()
    done = sorted([r for r in rows if r["outcome"]], key=lambda r: r["kickoff_utc"])
    pending = len(rows) - len(done)
    wib = dt.datetime.now(dt.timezone(dt.timedelta(hours=7)))
    head = f"📈 <b>LAPORAN HARIAN</b>  •  {_tgl(wib)}"
    if not done:
        return f"{head}\n{LINE}\nBelum ada prediksi yang selesai ({pending} menunggu hasil)."

    esc = lambda x: html.escape(str(x))
    n = len(done)
    acc = sum(r["correct"] == "1" for r in done) / n * 100
    home_base = sum(r["outcome"] == "home" for r in done) / n * 100

    recent = []
    for r in reversed(done[-5:]):
        icon = "✅" if r["correct"] == "1" else "❌"
        lean = "condong " if r["confidence"] == "rendah" and r["pick"] != "draw" else ""
        recent.append(
            f"{icon} {esc(_nm(r, 'home'))} {esc(r['final_score'])} {esc(_nm(r, 'away'))}\n"
            f"      <i>tebakan: {lean}{esc(_pick_label(r, r['pick']))} ({esc(r['confidence'])})</i>"
        )
    last = done[-10:]
    streak = "".join("🟢" if r["correct"] == "1" else "🔴" for r in last)

    lines = [
        head, LINE,
        "🕐 <b>HASIL TERAKHIR</b>", *recent, LINE,
        f"🎯 <b>AKURASI</b>  ({n} laga selesai, {pending} menunggu)",
        f"<code>Agent      {bar(acc)} {acc:>3.0f}%</code>",
        f"<code>Tuan rumah {bar(home_base)} {home_base:>3.0f}%</code>  <i>(pembanding)</i>",
        f"Tren {len(last)} terakhir: {streak}",
        LINE, "🔒 <b>PER KEYAKINAN</b>",
    ]
    for c in ("tinggi", "sedang", "rendah"):
        sub = [r for r in done if r["confidence"] == c]
        if sub:
            a = sum(r["correct"] == "1" for r in sub) / len(sub) * 100
            lines.append(f"{CONF_ICON[c]} <code>{c:<7} {bar(a)} {a:>3.0f}%</code>  ({len(sub)} laga)")
        else:
            lines.append(f"{CONF_ICON[c]} <code>{c:<7}</code> <i>belum ada laga</i>")

    lines += [LINE, "📐 <b>KUALITAS PROBABILITAS</b>  <i>(Brier, makin kecil makin baik)</i>"]
    cmp_rows = [r for r in done if r.get("g_home") and r.get("e_home")]
    if cmp_rows:
        sc = {"🤖 AI": _brier(cmp_rows, "g"), "📐 Elo": _brier(cmp_rows, "e"),
              "🔀 Gabungan": _brier(cmp_rows, "p")}
        best = min(sc, key=sc.get)
        for k, v in sc.items():
            lines.append(f"{k}: <b>{v:.3f}</b>" + ("  🏅" if k == best else ""))
        lines.append(f"<i>({len(cmp_rows)} laga; tebak acak = 0.667)</i>")
    else:
        lines.append(f"Gabungan: <b>{_brier(done, 'p'):.3f}</b>  <i>(tebak acak = 0.667)</i>")

    gl = [r for r in done if r.get("o25") and _total_goals(r) is not None]
    if gl:
        lines += [LINE, f"⚽ <b>GOL</b>  <i>(model Poisson, {len(gl)} laga)</i>"]
        for label, key, f in (("Over 1.5", "o15", lambda r: _total_goals(r) >= 2),
                              ("Over 2.5", "o25", lambda r: _total_goals(r) >= 3),
                              ("Kedua tim cetak gol", "btts", _both_scored)):
            okn = sum((float(r[key]) >= 50) == f(r) for r in gl)
            a = okn / len(gl) * 100
            lines.append(f"<code>{label:<19} {bar(a)} {a:>3.0f}%</code>")
    ps = [x for x in _load_parlays() if x["outcome"] in ("win", "lose")]
    if ps:
        lines += [LINE, "🎰 <b>KOMBINASI HARIAN</b>  <i>(hasil nyata vs perkiraan)</i>"]
        for kind, label in (("aman", "🛡️ Aman"), ("mix", "🎲 Mix")):
            sub = [x for x in ps if x["kind"] == kind]
            if sub:
                wins = sum(x["outcome"] == "win" for x in sub)
                expected = sum(x["p_comb"] for x in sub)
                hit = sum(int(x["legs_hit"].split("/")[0]) for x in sub)
                tot = sum(int(x["legs_hit"].split("/")[1]) for x in sub)
                lines.append(f"{label}: menang {wins}/{len(sub)} (perkiraan {expected:.1f}) • pilihan benar {hit}/{tot}")
    lines.append(LINE)
    if n < 30:
        lines.append(f"<i>⚠️ Sampel baru {n} laga, terlalu sedikit untuk disimpulkan. Tunggu minimal 30-50 laga.</i>")
    else:
        edge = acc - home_base
        verdict = "mengungguli" if edge > 3 else "setara dengan" if edge >= -3 else "di bawah"
        lines.append(f"💬 Agent {verdict} pembanding tuan rumah ({edge:+.0f} poin).")
    return "\n".join(lines)


# =====================================================================
# Kombinasi harian (parlay): gabungan pilihan terkuat dari semua liga
# =====================================================================
WINNER_MAP = {"HOME_TEAM": "home", "AWAY_TEAM": "away", "DRAW": "draw"}

def _load_parlays():
    try:
        with open(PARLAY_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def _save_parlays(items):
    with open(PARLAY_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=1)

def matches_next_24h():
    now = dt.datetime.now(dt.timezone.utc)
    data = fd_get("/matches", {
        "dateFrom": now.date().isoformat(),
        "dateTo": (now + dt.timedelta(days=2)).date().isoformat(),
        "competitions": LEAGUES,
    })
    out = []
    for m in (data or {}).get("matches", []):
        if m["status"] not in ("TIMED", "SCHEDULED"):
            continue
        ko = dt.datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
        mins = (ko - now).total_seconds() / 60
        if 20 <= mins <= 24 * 60:
            out.append(m)
    out.sort(key=lambda m: m["utcDate"])
    return out

GOAL_LABEL = {"over15": "Over 1.5 gol", "over25": "Over 2.5 gol", "under25": "Under 2.5 gol",
              "under35": "Under 3.5 gol", "btts": "Kedua tim cetak gol"}
GOAL_MARKETS = [("over15", "o15", False), ("over25", "o25", False), ("under25", "o25", True),
                ("under35", "o35", True), ("btts", "btts", False)]

def _base_leg(m):
    return {
        "match_id": m["id"], "kickoff_utc": m["utcDate"],
        "competition": m["competition"]["name"],
        "home": m["homeTeam"]["name"], "away": m["awayTeam"]["name"],
        "home_s": m["homeTeam"].get("shortName") or m["homeTeam"]["name"],
        "away_s": m["awayTeam"].get("shortName") or m["awayTeam"]["name"],
    }

def _leg_from(m, res):
    key = {"home": "home_win", "draw": "draw", "away": "away_win"}[res["pick"]]
    leg = _base_leg(m)
    leg.update(market="1x2", pick=res["pick"], p=float(res[key]), ai=bool(res.get("g")))
    return leg

def _goal_legs(m, gp):
    out = []
    for mk, key, inv in GOAL_MARKETS:
        p = 100 - gp[key] if inv else gp[key]
        if p >= PARLAY_MIN_GOAL_PROB:
            leg = _base_leg(m)
            leg.update(market=mk, pick=mk, p=float(p), ai=True)
            out.append(leg)
    return out

def _leg_pick_text(leg):
    if leg.get("market", "1x2") != "1x2":
        return GOAL_LABEL[leg["market"]]
    return {"home": f'{leg["home_s"]} menang', "away": f'{leg["away_s"]} menang', "draw": "Seri"}[leg["pick"]]

def _leg_hit(leg, gh, ga):
    mk, t = leg.get("market", "1x2"), gh + ga
    if mk == "over15":
        return t >= 2
    if mk == "over25":
        return t >= 3
    if mk == "under25":
        return t <= 2
    if mk == "under35":
        return t <= 3
    if mk == "btts":
        return gh > 0 and ga > 0
    return leg["pick"] == ("home" if gh > ga else "away" if ga > gh else "draw")

def _combo(legs):
    prod = 1.0
    for l in legs:
        prod *= l["p"] / 100
    return prod

def _ctx_for(m):
    comp = m["competition"]["code"]
    table = standings(comp) if comp != "CL" else {}
    hid, aid = m["homeTeam"]["id"], m["awayTeam"]["id"]
    return {"home_rank": table.get(hid, "n/a"), "away_rank": table.get(aid, "n/a"),
            "home_form": team_form(hid), "away_form": team_form(aid)}

def build_parlay_legs(ratings):
    """Kumpulkan pilihan per laga: hasil laga (untuk shortlist yang dianalisa AI) dan pasar gol (model Poisson).
    Mengembalikan juga bundel data tiap laga (klasemen, form, analisa) untuk pesan detail."""
    info = []
    for m in matches_next_24h():
        hid, aid = m["homeTeam"]["id"], m["awayTeam"]["id"]
        elo = elo_predict(ratings, hid, aid)
        gp = goal_probs(hid, aid, m["competition"]["code"])
        if elo or gp:
            info.append((m, elo, gp))
    n_scanned = len(info)
    ranked = sorted([x for x in info if x[1]], key=lambda x: -max(x[1]))
    short_ids = {x[0]["id"] for x in ranked[:PARLAY_SHORTLIST]}

    per_match, bundles = [], {}
    for m, elo, gp in info:
        b = {"m": m, "elo": elo, "gp": gp, "res": None, "ctx": None}
        opts = []
        if m["id"] in short_ids:
            ctx = _ctx_for(m)
            try:
                res = combine(analyze(m, ctx), elo)
            except Exception as e:
                print("[parlay] AI gagal, pakai Elo saja ->", e)
                res = elo_only(elo)
            b["ctx"], b["res"] = ctx, res
            leg = _leg_from(m, res)
            if leg["p"] >= PARLAY_MIN_PROB:
                opts.append(leg)
        if gp:
            opts += _goal_legs(m, gp)
        bundles[m["id"]] = b
        if opts:
            per_match.append(opts)
    return per_match, n_scanned, bundles

def _leg_notes(b):
    """Catatan detail satu laga untuk disimpan (dan ditampilkan di dashboard)."""
    res, ctx, gp, elo = b["res"], b["ctx"] or {}, b["gp"], b["elo"]
    note = {"home_rank": ctx.get("home_rank", ""), "away_rank": ctx.get("away_rank", ""),
            "home_form": ctx.get("home_form") or [], "away_form": ctx.get("away_form") or []}
    if res:
        note["p"] = [res["home_win"], res["draw"], res["away_win"]]
        if res.get("g"):
            note["g"] = list(res["g"])
        if not res.get("ai_failed"):
            note.update(summary=res.get("summary", ""), factors=list(res.get("key_factors") or [])[:4],
                        risks=list(res.get("risks") or [])[:3])
    if elo:
        note["e"] = [round(v, 1) for v in elo]
    if gp:
        note["goals"] = {k: round(gp[k], 1) for k in ("lam_h", "lam_a", "o15", "o25", "o35", "btts")}
        note["top"] = [[sc, round(pp, 1)] for sc, pp in gp["top"]]
    return note

def format_leg_detail(i, n, b, uses):
    """Pesan detail satu laga yang dipakai di kombinasi (dikirim sebagai balasan pesan kombinasi)."""
    esc = lambda x: html.escape(str(x))
    m, res, ctx, gp, elo = b["m"], b["res"], b["ctx"] or {}, b["gp"], b["elo"]
    h_full, a_full = m["homeTeam"]["name"], m["awayTeam"]["name"]
    h = (m["homeTeam"].get("shortName") or h_full)[:14]
    a = (m["awayTeam"].get("shortName") or a_full)[:14]
    ko = dt.datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00")).astimezone(dt.timezone(dt.timedelta(hours=7)))
    lines = [f"🔎 <b>DETAIL PILIHAN {i}/{n}</b>",
             f"⚽ <b>{esc(h_full)} vs {esc(a_full)}</b>",
             f"🏆 {esc(m['competition']['name'])}  •  🕐 {ko:%H:%M} WIB", LINE,
             "📌 <b>Dipakai di kombinasi</b>"]
    for kind, leg in uses:
        lines.append(f"{'🛡️ Aman' if kind == 'aman' else '🎲 Mix'}: <b>{esc(_leg_pick_text(leg))}</b> ({leg['p']:.0f}%)")
    lines += [LINE, "📋 <b>KONDISI TIM</b>"]
    for name, rank, form in ((h, ctx.get("home_rank"), ctx.get("home_form")),
                             (a, ctx.get("away_rank"), ctx.get("away_form"))):
        lines.append(f"<b>{esc(name)}</b>: {esc(rank or 'n/a')}")
        if form:
            lines.append("   Form 5 laga: " + esc("; ".join(form)))

    probs, title = None, "PROBABILITAS"
    if res:
        probs = (res["home_win"], res["draw"], res["away_win"])
        title += " (Elo saja, AI gagal)" if res.get("ai_failed") else " (gabungan AI + Elo)"
    elif elo:
        probs, title = elo, "PROBABILITAS (Elo saja)"
    if probs:
        rows = [("🟢", h, probs[0]), ("⚪", "Seri", probs[1]), ("🔴", a, probs[2])]
        w = max(len(r[1]) for r in rows)
        lines += [LINE, f"📊 <b>{title}</b>"]
        lines += [f"{ic} <code>{esc(nm_.ljust(w))} {bar(p)} {p:>3.0f}%</code>" for ic, nm_, p in rows]
        if res and res.get("g") and res.get("e"):
            g, e = res["g"], res["e"]
            lines.append(f"<code>🤖 AI {g[0]:.0f}/{g[1]:.0f}/{g[2]:.0f}   📐 Elo {e[0]:.0f}/{e[1]:.0f}/{e[2]:.0f}</code>")
    if gp:
        lean_name, lean_p = goal_lean(gp)
        lines += [LINE, "⚽ <b>GOL</b>  <i>(model statistik)</i>",
                  f"Perkiraan gol: {esc(h)} <b>{gp['lam_h']:.1f}</b> – <b>{gp['lam_a']:.1f}</b> {esc(a)}",
                  f"📈 Over 1.5: <b>{gp['o15']:.0f}%</b>   Over 2.5: <b>{gp['o25']:.0f}%</b>   Over 3.5: <b>{gp['o35']:.0f}%</b>",
                  f"🤝 Kedua tim cetak gol: <b>{gp['btts']:.0f}%</b>",
                  "🔢 Skor paling mungkin: " + ", ".join(f"{sc} ({pp:.0f}%)" for sc, pp in gp["top"]),
                  f"🎯 Sinyal gol terkuat: <b>{lean_name}</b> ({lean_p:.0f}%)"]
    if res and not res.get("ai_failed"):
        factors = "\n".join(f"✅ {esc(x)}" for x in (res.get("key_factors") or [])[:4])
        risks = "\n".join(f"⚠️ {esc(x)}" for x in (res.get("risks") or [])[:3])
        lines += [LINE, f"💡 <b>FAKTOR KUNCI</b>\n{factors}", f"🚧 <b>RISIKO</b>\n{risks}",
                  LINE, f"📝 <b>RINGKASAN</b>\n{esc(res.get('summary', ''))}"]
    elif res:
        lines += [LINE, "<i>Analisa AI gagal untuk laga ini (kuota atau gangguan), jadi pilihan memakai model statistik.</i>"]
    else:
        lines += [LINE, "<i>Analisa AI tidak dijalankan untuk laga ini (kuota dihemat untuk 5 laga dengan Elo terkuat), "
                        "jadi pilihan ini berdasarkan model statistik.</i>"]
    return "\n".join(lines)

def _fmt_leg(i, leg):
    ko = dt.datetime.fromisoformat(leg["kickoff_utc"].replace("Z", "+00:00"))
    ko_wib = ko.astimezone(dt.timezone(dt.timedelta(hours=7)))
    esc = lambda x: html.escape(str(x))
    if leg.get("market", "1x2") != "1x2":
        tag = " <i>(model gol)</i>"
    else:
        tag = "" if leg["ai"] else " <i>(Elo saja)</i>"
    return (f"{i}. {ko_wib:%H:%M} WIB • {esc(leg['competition'])}\n"
            f"    {esc(leg['home_s'])} vs {esc(leg['away_s'])} → <b>{esc(_leg_pick_text(leg))}</b> ({leg['p']:.0f}%){tag}")

def _fmt_combo(title, legs):
    prod = _combo(legs)
    odds = 1 / prod if prod > 0 else 0
    rows = "\n".join(_fmt_leg(i, l) for i, l in enumerate(sorted(legs, key=lambda l: l["kickoff_utc"]), 1))
    return (f"{title} — {len(legs)} pilihan\n{rows}\n"
            f"📊 Peluang SEMUA benar: <b>{prod * 100:.0f}%</b>  <i>(odds adil ≈ {odds:.1f})</i>")

def format_parlay_msg(safe, mix, n_scanned):
    wib = dt.datetime.now(dt.timezone(dt.timedelta(hours=7)))
    lines = [
        f"🎰 <b>KOMBINASI HARIAN</b>  •  {_tgl(wib)}",
        f"<i>Draft: lineup belum keluar, bisa berubah. {n_scanned} laga dipindai dari semua liga. "
        f"Pilihan mencakup hasil laga dan over/under gol. "
        f"Detail tiap laga ada di balasan pesan ini. Analisa per laga tetap datang sekitar 1 jam sebelum kick-off.</i>",
        LINE,
    ]
    if safe:
        lines.append(_fmt_combo("🛡️ <b>AMAN</b>", safe))
    else:
        lines.append(f"Hari ini tidak ada kombinasi yang layak: tidak ada cukup pilihan dengan peluang ≥ {PARLAY_MIN_PROB}%.")
    if mix:
        lines += [LINE, _fmt_combo("🎲 <b>MIX</b>", mix)]
    lines += [
        LINE,
        "⚠️ <b>Ingat:</b> peluang gabungan jauh lebih kecil dari peluang tiap pilihan "
        "(3 pilihan @70% hanya sekitar 34% benar semuanya). Satu laga meleset, seluruh kombinasi gagal.",
        track_record_line(),
        "",
        "<i>Ini analisa probabilitas, bukan jaminan hasil maupun saran taruhan.</i>",
    ]
    return "\n".join(lines)

def run_daily_parlay(manual=False):
    try:
        ratings = build_ratings()
    except Exception as e:
        print("[parlay] rating Elo gagal ->", e)
        return False
    per_match, n_scanned, bundles = build_parlay_legs(ratings)
    # AMAN: satu pilihan terkuat per laga (pasar apa pun), ambil 3 teratas
    best = sorted((max(o, key=lambda l: l["p"]) for o in per_match), key=lambda l: -l["p"])
    safe = best[:3] if len(best) >= 2 else []
    # MIX: utamakan hasil laga (menang/seri/kalah); kalau tidak layak, pakai pasar gol terbaik laga itu
    mixp = []
    for o in per_match:
        r1 = [l for l in o if l.get("market", "1x2") == "1x2"]
        mixp.append(max(r1, key=lambda l: l["p"]) if r1 else max(o, key=lambda l: l["p"]))
    mixp.sort(key=lambda l: -l["p"])
    mix = mixp[:5] if len(mixp) > len(safe) and len(mixp) >= 4 else []

    # laga yang dipakai di kombinasi, beserta pilihannya
    used = {}
    for kind, combo in (("aman", safe), ("mix", mix)):
        for leg in combo:
            used.setdefault(leg["match_id"], []).append((kind, leg))
    for mid in used:
        if bundles[mid]["ctx"] is None:
            try:
                bundles[mid]["ctx"] = _ctx_for(bundles[mid]["m"])
            except Exception as e:
                print("[parlay] kondisi tim gagal diambil ->", e)

    msg_id = send_telegram(format_parlay_msg(safe, mix, n_scanned))
    ordered = sorted(used.items(), key=lambda kv: kv[1][0][1]["kickoff_utc"])
    for i, (mid, uses) in enumerate(ordered, 1):
        try:
            send_telegram(format_leg_detail(i, len(ordered), bundles[mid], uses), reply_to=msg_id)
        except Exception as e:
            print("[parlay] detail laga gagal dikirim ->", e)

    today = dt.datetime.now(dt.timezone(dt.timedelta(hours=7))).date().isoformat()
    items = [x for x in _load_parlays() if not (x["date"] == today and not x["outcome"])]
    for kind, combo in (("aman", safe), ("mix", mix)):
        if combo:
            items.append({"id": f"{today}-{kind}", "date": today, "kind": kind, "legs": combo,
                          "notes": {str(l["match_id"]): _leg_notes(bundles[l["match_id"]]) for l in combo},
                          "p_comb": round(_combo(combo), 4), "tg_msg_id": msg_id,
                          "outcome": "", "legs_hit": ""})
    _save_parlays(items)
    return True

def maybe_send_daily_parlay():
    """Kirim kombinasi harian sekali sehari, di run pertama setelah PARLAY_HOUR_WIB (WIB)."""
    wib = dt.datetime.now(dt.timezone(dt.timedelta(hours=7)))
    if wib.hour < PARLAY_HOUR_WIB:
        return
    today = wib.date().isoformat()
    try:
        last = open(PARLAY_LAST_FILE).read().strip()
    except Exception:
        last = ""
    if last == today:
        return
    try:
        if run_daily_parlay():
            with open(PARLAY_LAST_FILE, "w") as f:
                f.write(today)
            print("Kombinasi harian terkirim.")
    except Exception as e:
        print("Kombinasi harian gagal ->", e)

def format_parlay_result(x, details):
    esc = lambda ee: html.escape(str(ee))
    names = {"aman": "🛡️ AMAN", "mix": "🎲 MIX"}
    rows = []
    for leg, w, score in details:
        ok = _leg_hit(leg, *map(int, score.split("-")))
        rows.append(f"{'✅' if ok else '❌'} {esc(leg['home_s'])} {esc(score)} {esc(leg['away_s'])}\n"
                    f"      <i>pilihan: {esc(_leg_pick_text(leg))}</i>")
    win = x["outcome"] == "win"
    return "\n".join([
        f"🏁 <b>HASIL KOMBINASI {names.get(x['kind'], x['kind'].upper())}</b>",
        LINE, *rows, LINE,
        f"{'✅ <b>KOMBINASI MENANG</b>' if win else '❌ <b>KOMBINASI GAGAL</b>'}  ({x['legs_hit']} pilihan benar)",
        f"📊 Peluang awal semua benar: {x['p_comb'] * 100:.0f}%",
    ])

def settle_parlays(notify=True):
    items = _load_parlays()
    if not any(not x["outcome"] for x in items):
        return 0
    now = dt.datetime.now(dt.timezone.utc)
    parse = lambda z: dt.datetime.fromisoformat(z.replace("Z", "+00:00"))
    changed = False
    ready = []
    for x in items:
        if x["outcome"]:
            continue
        last_ko = max(parse(l["kickoff_utc"]) for l in x["legs"])
        if now - last_ko > dt.timedelta(days=3):      # laga tertunda terlalu lama: abaikan
            x["outcome"], changed = "void", True
        elif now - last_ko >= dt.timedelta(minutes=RESULT_DELAY_MIN):
            ready.append(x)
    newly = []
    if ready:
        kos = [parse(l["kickoff_utc"]) for x in ready for l in x["legs"]]
        data = fd_get("/matches", {"dateFrom": min(kos).date().isoformat(),
                                   "dateTo": (max(kos).date() + dt.timedelta(days=1)).isoformat(),
                                   "competitions": LEAGUES})
        found = {m["id"]: m for m in (data or {}).get("matches", [])}
        for x in ready:
            details, complete = [], True
            for leg in x["legs"]:
                m = found.get(int(leg["match_id"]))
                sc = (m or {}).get("score") or {}
                ft = sc.get("fullTime") or {}
                if not m or m.get("status") != "FINISHED" or sc.get("winner") not in WINNER_MAP or ft.get("home") is None:
                    complete = False
                    break
                details.append((leg, WINNER_MAP[sc["winner"]], f'{ft["home"]}-{ft["away"]}'))
            if not complete:
                continue
            hits = sum(1 for leg, w, sc in details if _leg_hit(leg, *map(int, sc.split("-"))))
            x["legs_hit"] = f"{hits}/{len(details)}"
            x["leg_results"] = {str(leg["match_id"]): {"score": sc, "hit": _leg_hit(leg, *map(int, sc.split("-")))}
                                for leg, w, sc in details}
            x["outcome"] = "win" if hits == len(details) else "lose"
            newly.append((x, details))
            changed = True
    if changed:
        _save_parlays(items)
    if notify:
        for x, details in newly:
            try:
                send_telegram(format_parlay_result(x, details), reply_to=x.get("tg_msg_id") or None)
            except Exception as e:
                print("Gagal kirim hasil kombinasi ->", e)
    return len(newly)


def update_crests(max_age_days=14):
    """Simpan peta nama tim -> URL logo (dari football-data.org) untuk dashboard.
    Dihitung ulang tiap 14 hari saja (6 panggilan API)."""
    now = dt.datetime.now(dt.timezone.utc)
    try:
        with open(CREST_FILE, encoding="utf-8") as f:
            c = json.load(f)
        limit = max_age_days if c.get("crests") else 1
        if now - dt.datetime.fromisoformat(c["built_utc"]) < dt.timedelta(days=limit):
            return
    except Exception:
        pass
    crests = {}
    for code in LEAGUES.split(","):
        data = fd_get(f"/competitions/{code}/teams")
        for t in (data or {}).get("teams", []):
            if t.get("crest"):
                crests[t["name"]] = t["crest"]
                if t.get("shortName"):
                    crests.setdefault(t["shortName"], t["crest"])
    with open(CREST_FILE, "w", encoding="utf-8") as f:
        json.dump({"built_utc": now.isoformat(), "crests": crests}, f, ensure_ascii=False)
    print(f"[logo] {len(crests)} logo tim tersimpan")

def maybe_send_daily_report():
    """Kirim laporan akurasi sekali sehari, di run pertama setelah jam 09:00 WIB.
    Tidak bergantung pada jadwal cron khusus (yang sering telat/terlewat di GitHub)."""
    wib = dt.datetime.now(dt.timezone(dt.timedelta(hours=7)))
    if wib.hour < REPORT_HOUR_WIB:
        return
    today = wib.date().isoformat()
    try:
        last = open(REPORT_FILE).read().strip()
    except Exception:
        last = ""
    if last == today:
        return
    try:
        settle_results(max_calls=8)
        send_telegram(build_report())
        with open(REPORT_FILE, "w") as f:
            f.write(today)
        print("Laporan harian terkirim.")
    except Exception as e:
        print("Laporan harian gagal ->", e)


# =====================================================================
# Main
# =====================================================================
def main():
    args = sys.argv[1:]

    if "--parlay" in args:
        run_daily_parlay(manual=True)
        return

    if "--report" in args:
        settle_results(max_calls=8)
        settle_parlays()
        text = build_report()
        print(html.unescape(re.sub(r"<[^>]+>", "", text)))
        send_telegram(text)
        return

    test_mode = "--test" in args
    sent = set() if test_mode else load_sent()
    if not test_mode:
        settle_results()
        settle_parlays()
        maybe_send_daily_report()
        maybe_send_daily_parlay()
        try:
            update_crests()
        except Exception as e:
            print("[logo] gagal ->", e)

    matches = upcoming_matches(test_mode)
    if test_mode:
        matches = matches[:1]
    if not matches:
        print("Tidak ada laga yang cocok saat ini.")
        return

    print(f"{len(matches)} laga di jendela waktu:",
          "; ".join(f'{m["homeTeam"]["shortName"]} vs {m["awayTeam"]["shortName"]}' for m in matches))
    try:
        ratings = build_ratings()
    except Exception as e:
        print("[elo] gagal menghitung rating, pakai AI saja ->", e)
        ratings = {}

    done = 0
    for m in matches:
        if str(m["id"]) in sent:
            continue
        if done >= MAX_MATCHES_PER_RUN:
            break
        comp = m["competition"]["code"]
        table = standings(comp) if comp != "CL" else {}
        hid, aid = m["homeTeam"]["id"], m["awayTeam"]["id"]
        ctx = {
            "home_rank": table.get(hid, "n/a"),
            "away_rank": table.get(aid, "n/a"),
            "home_form": team_form(hid),
            "away_form": team_form(aid),
        }
        try:
            elo = elo_predict(ratings, hid, aid)
            try:
                res = combine(analyze(m, ctx), elo)
            except Exception as ai_err:
                print("[ai] analisa gagal ->", ai_err)
                if not elo:
                    raise
                print("[ai] pakai sinyal Elo saja untuk laga ini")
                res = elo_only(elo)
            gp = goal_probs(hid, aid, comp)
            if gp:
                res["goals"] = gp
                res["likely_score"] = gp["top"][0][0]
            msg_id = send_telegram(format_msg(m, res))
            if not test_mode:
                log_prediction(m, res, msg_id)
                sent.add(str(m["id"]))
                save_sent(sent)
            done += 1
            print("Terkirim:", m["homeTeam"]["name"], "vs", m["awayTeam"]["name"])
        except Exception as e:
            print("Gagal untuk match", m["id"], "->", e)

if __name__ == "__main__":
    main()
