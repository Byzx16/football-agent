"""
Konektor The Odds API (paket gratis: 500 kredit/bulan) untuk jadwal, odds pasar, dan skor
kompetisi yang tidak ada di football-data.org gratis (mis. UEFA Europa League).

Mode diagnostik (jalankan sekali, kunci tidak pernah dicetak):
    python oddsapi.py --check

Isi kunci di file .env:   ODDS_API_KEY=kunci_kamu
"""
import os, re, sys, difflib, unicodedata, datetime as dt
import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("ODDS_API_KEY", "").strip()
BASE = "https://api.the-odds-api.com/v4"
SPORT_KEYS = {"EL": "soccer_uefa_europa_league", "CL": "soccer_uefa_champs_league", "PL": "soccer_epl",
              "PD": "soccer_spain_la_liga", "SA": "soccer_italy_serie_a", "BL1": "soccer_germany_bundesliga",
              "FL1": "soccer_france_ligue_one"}
QUOTA = {"remaining": None, "used": None, "last": None}


def oa_get(path, params=None):
    """Panggil The Odds API. Mengembalikan JSON (None kalau gagal). Sisa kuota dicatat di QUOTA."""
    p = dict(params or {})
    p["apiKey"] = KEY
    try:
        r = requests.get(BASE + path, params=p, timeout=30)
    except Exception as e:
        print(f"  [odds-api] {path}: gagal tersambung ({type(e).__name__})")
        return None
    for k, h in (("remaining", "x-requests-remaining"), ("used", "x-requests-used"), ("last", "x-requests-last")):
        if r.headers.get(h) is not None:
            QUOTA[k] = r.headers.get(h)
    if r.status_code != 200:
        print(f"  [odds-api] {path} -> HTTP {r.status_code}: {r.text[:200]}")
        return None
    try:
        return r.json()
    except Exception:
        return None


# ---------------------------------------------------------------- nama tim
_STOP = {"fc", "cf", "afc", "sc", "ac", "as", "ss", "ssc", "us", "rc", "cd", "sv", "fk", "ud", "de", "di", "da",
         "the", "club", "calcio", "futbol", "football", "1907", "1901", "1899", "1909", "04", "05", "1846", "and"}

def norm_name(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s.replace("&", " and "))
    return [t for t in s.split() if t not in _STOP]

def name_sim(a, b):
    """Kemiripan 0..1 antara dua nama tim (tahan singkatan seperti 'Inter Milan' vs 'Internazionale')."""
    ta, tb = norm_name(a), norm_name(b)
    if not ta or not tb:
        return 0.0
    score = difflib.SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if all(any(l.startswith(t) or t.startswith(l) for l in long_) for t in short if len(t) >= 3):
        score = max(score, 0.85)
    if len(short) == 1 and len(short[0]) >= 5 and any(difflib.SequenceMatcher(None, short[0], l).ratio() >= 0.75 for l in long_):
        score = max(score, 0.82)          # mis. 'Rennes' vs 'Stade Rennais'
    return score

def best_match(name, candidates, min_score=0.8):
    """Cari nama terdekat dari daftar (nama -> data). Mengembalikan (nama, data, skor) atau None."""
    best = None
    for c, data in candidates.items():
        s = name_sim(name, c)
        if best is None or s > best[2]:
            best = (c, data, s)
    return best if best and best[2] >= min_score else None


# ---------------------------------------------------------------- odds
def devig(odds):
    """Odds desimal -> probabilitas (persen) setelah margin bandar dibuang. None kalau ada odds tidak valid."""
    if any((o is None or o <= 1) for o in odds):
        return None
    inv = [1 / o for o in odds]
    t = sum(inv)
    return [x / t * 100 for x in inv]

def parse_event_odds(ev):
    """Dari satu event /odds: probabilitas pasar rata-rata seluruh bandar (margin dibuang), odds desimal rata-rata
    dan terbaik per hasil. Mengembalikan {'p1x2', 'avg', 'best', 'books', 'o25'?} atau None."""
    home, away = ev.get("home_team"), ev.get("away_team")
    h2h, prices, ou25 = [], [], []
    for b in ev.get("bookmakers") or []:
        for m in b.get("markets") or []:
            if m.get("key") == "h2h":
                price = {o["name"]: o["price"] for o in m.get("outcomes") or []}
                trio = [price.get(home), price.get("Draw"), price.get(away)]
                p = devig(trio)
                if p:
                    h2h.append(p)
                    prices.append(trio)
            elif m.get("key") == "totals":
                ov = un = None
                for o in m.get("outcomes") or []:
                    if o.get("point") == 2.5:
                        if o.get("name") == "Over":
                            ov = o["price"]
                        elif o.get("name") == "Under":
                            un = o["price"]
                p = devig([ov, un])
                if p:
                    ou25.append(p[0])
    if not h2h:
        return None
    n = len(h2h)
    out = {"p1x2": [sum(x[i] for x in h2h) / n for i in range(3)], "books": n,
           "avg": [sum(x[i] for x in prices) / n for i in range(3)],
           "best": [max(x[i] for x in prices) for i in range(3)]}
    if ou25:
        out["o25"] = sum(ou25) / len(ou25)
    return out

def parse_scores(ev):
    """Dari satu event /scores yang sudah selesai: (skor kandang, skor tandang) atau None."""
    if not ev.get("completed"):
        return None
    sc = {s["name"]: s.get("score") for s in ev.get("scores") or []}
    try:
        return int(sc[ev["home_team"]]), int(sc[ev["away_team"]])
    except Exception:
        return None


# ---------------------------------------------------------------- diagnostik
def _wib(iso):
    return dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(dt.timezone(dt.timedelta(hours=7))).strftime("%d %b %H:%M")

def check():
    if not KEY:
        print("ODDS_API_KEY belum diisi. Tambahkan baris  ODDS_API_KEY=kunci_kamu  di file .env lalu jalankan lagi.")
        return
    print("== 1. Daftar sport sepak bola ==")
    sports = oa_get("/sports", {"all": "true"})
    if sports is None:
        print("Gagal membaca daftar sport. Periksa kuncinya.")
        return
    soccer = {s["key"]: s for s in sports if s.get("key", "").startswith("soccer")}
    for code, key in SPORT_KEYS.items():
        s = soccer.get(key)
        print(f"{code}: {key} -> " + (f"ada | aktif: {s.get('active')}" if s else "TIDAK ditemukan"))
    print(f"(kuota setelah langkah ini: sisa {QUOTA['remaining']}, terpakai {QUOTA['used']})")

    el = SPORT_KEYS["EL"]
    print(f"\n== 2. Jadwal UEL (events) ==")
    before = QUOTA["remaining"]
    evs = oa_get(f"/sports/{el}/events") or []
    print(f"Jumlah laga terjadwal: {len(evs)} | biaya kredit panggilan ini: {before} -> {QUOTA['remaining']}")
    for e in evs[:6]:
        print(f"  {_wib(e['commence_time'])} WIB  {e['home_team']} vs {e['away_team']}")

    print(f"\n== 3. Odds pasar UEL (h2h + totals, wilayah eu) ==")
    before = QUOTA["remaining"]
    odds = oa_get(f"/sports/{el}/odds", {"regions": "eu", "markets": "h2h,totals", "oddsFormat": "decimal"}) or []
    print(f"Laga dengan odds: {len(odds)} | kredit dipakai panggilan ini: {QUOTA['last']} (sisa {QUOTA['remaining']}, sebelumnya {before})")
    for e in odds[:5]:
        o = parse_event_odds(e)
        if o:
            p = o["p1x2"]
            print(f"  {e['home_team']} vs {e['away_team']}: menang {p[0]:.0f}% / seri {p[1]:.0f}% / tandang {p[2]:.0f}%"
                  + (f" | Over 2.5 {o['o25']:.0f}%" if "o25" in o else "") + f" ({o['books']} bandar)")

    print(f"\n== 4. Skor UEL 3 hari terakhir (untuk menilai tebakan) ==")
    sc = oa_get(f"/sports/{el}/scores", {"daysFrom": 3}) or []
    done = [(e, parse_scores(e)) for e in sc if parse_scores(e)]
    print(f"Laga selesai: {len(done)} | kredit dipakai: {QUOTA['last']} (sisa {QUOTA['remaining']})")
    for e, s in done[:5]:
        print(f"  {e['home_team']} {s[0]}-{s[1]} {e['away_team']}")

    print(f"\n== 5. Pencocokan nama tim dengan football-data.org ==")
    try:
        import agent_free as a
    except Exception as ex:
        print("Lewati (agent_free tidak bisa dimuat):", type(ex).__name__)
        return
    idx = {}
    for code in ("PL", "PD", "SA", "BL1", "FL1", "PPL", "DED", "CL"):
        data = a.fd_get(f"/competitions/{code}/teams")
        for t in (data or {}).get("teams", []):
            idx.setdefault(t["name"], t["id"])
    names = sorted({n for e in (evs or odds) for n in (e["home_team"], e["away_team"])})
    hit = {n: best_match(n, idx) for n in names}
    ok = [n for n, m in hit.items() if m]
    print(f"Tim UEL: {len(names)} | cocok dengan data kita (punya Elo dan logo): {len(ok)} ({len(ok) / max(len(names), 1) * 100:.0f}%)")
    print("Tidak cocok (akan tanpa Elo, hanya odds pasar dan AI): " + (", ".join(n for n in names if not hit[n]) or "-"))
    print("Contoh cocok: " + "; ".join(f"{n} = {hit[n][0]}" for n in ok[:6]))
    print(f"\nSisa kuota bulan ini: {QUOTA['remaining']}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        print(__doc__)
