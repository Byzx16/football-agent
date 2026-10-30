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
import os, sys, json, re, time, csv, html, datetime as dt
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
MAX_MATCHES_PER_RUN = 3          # jaga kuota harian Gemini gratis
SENT_FILE = "sent.json"
PRED_FILE = "predictions.csv"
ELO_FILE = "elo_cache.json"
ELO_CACHE_HOURS = 12
ELO_K, ELO_HFA, ELO_CARRYOVER = 20, 65, 0.7   # K-factor, keunggulan kandang, sisa rating antar musim
W_GEMINI = 0.5                                # bobot AI di probabilitas gabungan (sisanya Elo)
PRED_FIELDS = ["match_id", "kickoff_utc", "competition", "home", "away",
               "p_home", "p_draw", "p_away", "pick", "confidence",
               "likely_score", "outcome", "final_score", "correct",
               "g_home", "g_draw", "g_away", "e_home", "e_draw", "e_away"]

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
        if now - built < dt.timedelta(hours=ELO_CACHE_HOURS) and c.get("ratings"):
            return {int(k): v for k, v in c["ratings"].items()}
    except Exception:
        pass
    print("[elo] menghitung rating dari hasil pertandingan (sekitar 1-2 menit)...")
    cur = now.year if now.month >= 7 else now.year - 1
    R = {}
    for season in (cur - 1, cur):
        ms = []
        for code in LEAGUES.split(","):
            data = fd_get(f"/competitions/{code}/matches", {"season": season, "status": "FINISHED"})
            if data:
                ms += data.get("matches", [])
        ms.sort(key=lambda m: m["utcDate"])
        for m in ms:
            _elo_update(R, m)
        if season == cur - 1:   # antar musim, rating ditarik sebagian ke rata-rata
            R = {k: 1500 + (v - 1500) * ELO_CARRYOVER for k, v in R.items()}
    if not R:
        return {}
    with open(ELO_FILE, "w") as f:
        json.dump({"built_utc": now.isoformat(), "ratings": {str(k): v for k, v in R.items()}}, f)
    return R

def elo_predict(R, hid, aid):
    """Probabilitas (home, seri, away) dalam persen, atau None kalau salah satu tim belum punya rating."""
    if hid not in R or aid not in R:
        return None
    e = 1 / (1 + 10 ** (-((R[hid] + ELO_HFA) - R[aid]) / 400))
    d = 0.28 * (1 - abs(2 * e - 1)) ** 0.6          # peluang seri mengecil kalau selisih kekuatan besar
    ph, pa = max(0.01, e - d / 2), max(0.01, 1 - e - d / 2)
    tot = ph + d + pa
    return (ph / tot * 100, d / tot * 100, pa / tot * 100)

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
        print(f"[gemini] dengan search gagal ({r.status_code}), coba tanpa search...")
        r = gemini_call(prompt, use_search=False)
    r.raise_for_status()
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
def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    r = requests.post(
        url,
        json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML",
              "disable_web_page_preview": True},
        timeout=30,
    )
    if r.status_code == 200:
        return
    reason = r.json().get("description", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
    print(f"[telegram] gagal kirim (HTML): {r.status_code} - {reason}")
    plain = html.unescape(re.sub(r"<[^>]+>", "", text))
    r2 = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": plain,
                                  "disable_web_page_preview": True}, timeout=30)
    if r2.status_code != 200:
        reason2 = r2.json().get("description", r2.text) if r2.headers.get("content-type", "").startswith("application/json") else r2.text
        raise RuntimeError(f"Telegram menolak pesan: {r2.status_code} - {reason2}")

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
    if res.get("e"):
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

    factors = "\n".join(f"✅ {esc(x)}" for x in res.get("key_factors", [])[:4])
    risks = "\n".join(f"⚠️ {esc(x)}" for x in res.get("risks", [])[:3])

    return (
        f"⚽ <b>{esc(h_full)} vs {esc(a_full)}</b>\n"
        f"🏆 {esc(match['competition']['name'])}  •  🕐 {ko_wib:%d %b, %H:%M} WIB\n"
        f"{LINE}\n"
        f"📊 <b>{title}</b>\n{prob_lines}\n{models}"
        f"{LINE}\n"
        f"{signal}\n"
        f"{CONF_ICON[conf]} Keyakinan: <b>{conf}</b>   🔢 Skor mungkin: <b>{esc(res.get('likely_score', '-'))}</b>\n"
        f"{LINE}\n"
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

def log_prediction(match, res):
    rows = read_preds()
    if any(r["match_id"] == str(match["id"]) for r in rows):
        return
    rows.append({
        "match_id": match["id"],
        "kickoff_utc": match["utcDate"],
        "competition": match["competition"]["name"],
        "home": match["homeTeam"]["name"],
        "away": match["awayTeam"]["name"],
        "p_home": res["home_win"], "p_draw": res["draw"], "p_away": res["away_win"],
        "pick": res["pick"], "confidence": res["confidence"],
        "likely_score": res.get("likely_score", ""),
        "outcome": "", "final_score": "", "correct": "",
        **{k: round(v, 1) for k, v in zip(("g_home", "g_draw", "g_away"), res.get("g") or ())},
        **{k: round(v, 1) for k, v in zip(("e_home", "e_draw", "e_away"), res.get("e") or ())},
    })
    write_preds(rows)

def settle_results(max_calls=4):
    """Cek hasil akhir pertandingan yang sudah dicatat dan belum diisi hasilnya."""
    rows = read_preds()
    now = dt.datetime.now(dt.timezone.utc)
    calls, changed = 0, False
    winner_map = {"HOME_TEAM": "home", "AWAY_TEAM": "away", "DRAW": "draw"}
    for r in rows:
        if r["outcome"]:
            continue
        ko = dt.datetime.fromisoformat(r["kickoff_utc"].replace("Z", "+00:00"))
        if now - ko < dt.timedelta(hours=3):
            continue
        if calls >= max_calls:
            break
        data = fd_get(f"/matches/{r['match_id']}")
        calls += 1
        if not data:
            continue
        m = data.get("match", data)
        if m.get("status") != "FINISHED":
            continue
        winner = (m.get("score") or {}).get("winner")
        ft = (m.get("score") or {}).get("fullTime") or {}
        if winner not in winner_map:
            continue
        r["outcome"] = winner_map[winner]
        r["final_score"] = f'{ft.get("home")}-{ft.get("away")}'
        r["correct"] = "1" if r["pick"] == r["outcome"] else "0"
        changed = True
    if changed:
        write_preds(rows)

def build_report():
    rows = read_preds()
    done = [r for r in rows if r["outcome"]]
    pending = len(rows) - len(done)
    if not done:
        return (f"📈 <b>LAPORAN AKURASI</b>\nBelum ada prediksi yang selesai "
                f"({pending} masih menunggu hasil).")
    n = len(done)
    acc = sum(r["correct"] == "1" for r in done) / n * 100
    home_base = sum(r["outcome"] == "home" for r in done) / n * 100

    brier_sum = 0.0
    for r in done:
        p = [float(r["p_home"]) / 100, float(r["p_draw"]) / 100, float(r["p_away"]) / 100]
        o = [1.0 if r["outcome"] == k else 0.0 for k in ("home", "draw", "away")]
        brier_sum += sum((pi - oi) ** 2 for pi, oi in zip(p, o))
    brier = brier_sum / n

    lines = [
        "📈 <b>LAPORAN AKURASI</b>",
        LINE,
        f"Prediksi selesai: <b>{n}</b>   (menunggu hasil: {pending})",
        f"🎯 Akurasi sinyal: <b>{acc:.0f}%</b>",
        f"🏠 Pembanding (selalu pilih tuan rumah): {home_base:.0f}%",
        f"📐 Brier score: <b>{brier:.3f}</b>  (tebak acak 33/33/33 = 0.667, makin kecil makin baik)",
        LINE,
        "<b>Per tingkat keyakinan</b>",
    ]
    cmp_rows = [r for r in done if r.get("g_home") and r.get("e_home")]
    if cmp_rows:
        def _b(pref):
            tot = 0.0
            for r in cmp_rows:
                p = [float(r[f"{pref}_{k}"]) / 100 for k in ("home", "draw", "away")]
                o = [1.0 if r["outcome"] == k else 0.0 for k in ("home", "draw", "away")]
                tot += sum((x - y) ** 2 for x, y in zip(p, o))
            return tot / len(cmp_rows)
        lines.insert(6, f"🤖 vs 📐 Brier ({len(cmp_rows)} laga): AI {_b('g'):.3f} • Elo {_b('e'):.3f} • Gabungan {_b('p'):.3f}")
    for c in ("tinggi", "sedang", "rendah"):
        sub = [r for r in done if r["confidence"] == c]
        if sub:
            a = sum(r["correct"] == "1" for r in sub) / len(sub) * 100
            lines.append(f"{CONF_ICON[c]} {c}: {a:.0f}% benar  ({len(sub)} laga)")
    if n < 30:
        lines.append(f"\n<i>Sampel baru {n} laga, terlalu sedikit untuk disimpulkan. Tunggu minimal 30-50 laga.</i>")
    return "\n".join(lines)


# =====================================================================
# Main
# =====================================================================
def main():
    args = sys.argv[1:]

    if "--report" in args:
        settle_results(max_calls=8)
        text = build_report()
        print(html.unescape(re.sub(r"<[^>]+>", "", text)))
        send_telegram(text)
        return

    test_mode = "--test" in args
    sent = set() if test_mode else load_sent()
    if not test_mode:
        settle_results()

    matches = upcoming_matches(test_mode)
    if test_mode:
        matches = matches[:1]
    if not matches:
        print("Tidak ada laga yang cocok saat ini.")
        return

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
            res = combine(analyze(m, ctx), elo_predict(ratings, hid, aid))
            send_telegram(format_msg(m, res))
            if not test_mode:
                log_prediction(m, res)
                sent.add(str(m["id"]))
                save_sent(sent)
            done += 1
            print("Terkirim:", m["homeTeam"]["name"], "vs", m["awayTeam"]["name"])
        except Exception as e:
            print("Gagal untuk match", m["id"], "->", e)

if __name__ == "__main__":
    main()
