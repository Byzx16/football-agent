"""
Konektor API-Football (paket gratis: 100 request/hari) untuk odds pasar, cedera, dan lineup resmi.

Mode diagnostik (jalankan sekali, kunci tidak pernah dicetak):
    python apifootball.py --check

Isi kunci di file .env:   APIFOOTBALL_KEY=kunci_kamu
"""
import os, re, sys, difflib, unicodedata, datetime as dt
import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("APIFOOTBALL_KEY", "").strip()
BASE = "https://v3.football.api-sports.io"
LEAGUE_IDS = {"PL": 39, "PD": 140, "SA": 135, "BL1": 78, "FL1": 61, "CL": 2}   # kode football-data.org -> id API-Football
_last_left = None


def af_get(path, params=None):
    """Panggil API-Football. Mengembalikan isi JSON ({} kalau gagal)."""
    global _last_left
    try:
        r = requests.get(BASE + path, headers={"x-apisports-key": KEY}, params=params, timeout=30)
    except Exception as e:
        print(f"  [api-football] {path}: gagal tersambung ({type(e).__name__})")
        return {}
    _last_left = r.headers.get("x-ratelimit-requests-remaining", _last_left)
    try:
        data = r.json()
    except Exception:
        data = {}
    errs = data.get("errors")
    if r.status_code != 200 or errs:
        print(f"  [api-football] {path} {params or ''} -> HTTP {r.status_code}, errors={errs}")
    return data


# ---------------------------------------------------------------- pencocokan nama tim
_STOP = {"fc", "cf", "afc", "sc", "ac", "as", "ss", "ssc", "us", "rc", "cd", "sv", "fk", "ud", "de", "di", "da",
         "the", "club", "calcio", "futbol", "football", "1907", "1901", "1899", "1909", "04", "05", "1846"}

def norm_name(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s.replace("&", " and "))
    return [t for t in s.split() if t not in _STOP]

def name_sim(a, b):
    """Kemiripan 0..1 antara dua nama tim (tahan singkatan seperti 'Inter' vs 'Internazionale Milano')."""
    ta, tb = norm_name(a), norm_name(b)
    if not ta or not tb:
        return 0.0
    sa, sb = " ".join(ta), " ".join(tb)
    score = difflib.SequenceMatcher(None, sa, sb).ratio()
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if all(any(l.startswith(t) or t.startswith(l) for l in long_) for t in short if len(t) >= 3):
        score = max(score, 0.85)
    return score

def map_fixtures(fd_matches, af_fixtures, max_gap_min=90, min_score=0.35):
    """Cocokkan laga football-data.org dengan fixture API-Football (liga sama, jam kick-off berdekatan, nama mirip).
    fd_matches: list laga football-data; af_fixtures: list 'response' dari /fixtures.
    Mengembalikan {id laga football-data: id fixture API-Football}."""
    pairs = []
    for m in fd_matches:
        ko = dt.datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
        lid = LEAGUE_IDS.get(m["competition"]["code"])
        for f in af_fixtures:
            if (f.get("league") or {}).get("id") != lid:
                continue
            fk = dt.datetime.fromisoformat(f["fixture"]["date"])
            gap = abs((fk - ko).total_seconds()) / 60
            if gap > max_gap_min:
                continue
            s = (name_sim(m["homeTeam"]["name"], f["teams"]["home"]["name"])
                 + name_sim(m["awayTeam"]["name"], f["teams"]["away"]["name"])) / 2
            pairs.append((s, -gap, m["id"], f["fixture"]["id"]))
    out, used_f = {}, set()
    for s, _, mid, fid in sorted(pairs, reverse=True):
        if s >= min_score and mid not in out and fid not in used_f:
            out[mid] = fid
            used_f.add(fid)
    return out


# ---------------------------------------------------------------- odds
def devig(odds):
    """Ubah odds desimal jadi probabilitas (persen) setelah margin bandar dibuang."""
    inv = [1 / o for o in odds if o and o > 1]
    if len(inv) != len(odds):
        return None
    t = sum(inv)
    return [x / t * 100 for x in inv]

def parse_odds(resp, prefer=("Pinnacle", "Bet365", "Marathonbet", "1xBet", "Bwin", "Unibet")):
    """Dari jawaban /odds?fixture=ID: ambil probabilitas pasar 1X2, Over/Under 2.5, dan kedua tim cetak gol."""
    books = []
    for item in (resp.get("response") or []):
        books += item.get("bookmakers") or []
    if not books:
        return None
    books.sort(key=lambda b: prefer.index(b["name"]) if b.get("name") in prefer else len(prefer))
    for b in books:
        bets = {bt.get("name", ""): bt.get("values") or [] for bt in b.get("bets") or []}
        mw = bets.get("Match Winner")
        if not mw:
            continue
        val = {v["value"]: float(v["odd"]) for v in mw}
        p1x2 = devig([val.get("Home"), val.get("Draw"), val.get("Away")])
        if not p1x2:
            continue
        out = {"book": b.get("name"), "p1x2": p1x2}
        ou = {v["value"]: float(v["odd"]) for v in bets.get("Goals Over/Under", [])}
        if "Over 2.5" in ou and "Under 2.5" in ou:
            out["o25"] = devig([ou["Over 2.5"], ou["Under 2.5"]])[0]
        bt = {v["value"]: float(v["odd"]) for v in bets.get("Both Teams Score", [])}
        if "Yes" in bt and "No" in bt:
            out["btts"] = devig([bt["Yes"], bt["No"]])[0]
        return out
    return None


# ---------------------------------------------------------------- diagnostik
def check():
    if not KEY:
        print("APIFOOTBALL_KEY belum diisi. Tambahkan baris  APIFOOTBALL_KEY=kunci_kamu  di file .env lalu jalankan lagi.")
        return
    import agent_free as a
    now = dt.datetime.now(dt.timezone.utc)
    season = now.year if now.month >= 7 else now.year - 1

    print("== 1. Akun ==")
    d = af_get("/status")
    resp = d.get("response") or {}
    sub, req = (resp.get("subscription") or {}) if isinstance(resp, dict) else {}, (resp.get("requests") or {}) if isinstance(resp, dict) else {}
    print(f"Paket: {sub.get('plan')} | aktif: {sub.get('active')} | request hari ini: {req.get('current')}/{req.get('limit_day')}")

    print(f"\n== 2. Akses musim {season} per liga ==")
    ok_season = {}
    for code, lid in LEAGUE_IDS.items():
        d = af_get("/leagues", {"id": lid, "season": season})
        lst = d.get("response") or []
        if not lst:
            print(f"{code} ({lid}): musim {season} TIDAK tersedia di akun ini")
            ok_season[code] = False
            continue
        cov = (lst[0].get("seasons") or [{}])[0].get("coverage") or {}
        fx = cov.get("fixtures") or {}
        print(f"{code} ({lid}): musim {season} tersedia | lineup: {fx.get('lineups')} | cedera: {cov.get('injuries')} | odds: {cov.get('odds')}")
        ok_season[code] = True
    if not any(ok_season.values()):
        print("\nAkun gratis ini tidak bisa membaca musim berjalan. Odds bisa memakai The Odds API sebagai gantinya.")
        return

    print("\n== 3. Pencocokan laga football-data.org dengan API-Football (48 jam ke depan) ==")
    fd = a.matches_in_window(0, 48 * 60) or []
    fd = [m for m in fd if ok_season.get(m["competition"]["code"])]
    print(f"Laga di football-data.org: {len(fd)}")
    if not fd:
        print("Tidak ada laga 48 jam ke depan di liga yang tersedia. Jalankan lagi menjelang hari pertandingan.")
        return
    combos = sorted({(m["competition"]["code"], m["utcDate"][:10]) for m in fd})
    af_fx = []
    for code, day in combos:
        d = af_get("/fixtures", {"league": LEAGUE_IDS[code], "season": season, "date": day})
        af_fx += d.get("response") or []
    mp = map_fixtures(fd, af_fx)
    print(f"Fixture dari API-Football: {len(af_fx)} | berhasil dicocokkan: {len(mp)} dari {len(fd)} ({len(mp) / len(fd) * 100:.0f}%)")
    fx_by_id = {f["fixture"]["id"]: f for f in af_fx}
    for m in fd[:12]:
        fid = mp.get(m["id"])
        right = f'{fx_by_id[fid]["teams"]["home"]["name"]} vs {fx_by_id[fid]["teams"]["away"]["name"]}' if fid else "(tidak cocok)"
        print(f'  {m["homeTeam"]["name"]} vs {m["awayTeam"]["name"]}  ->  {right}')

    if mp:
        mid, fid = next(iter(mp.items()))
        print(f"\n== 4. Contoh data untuk satu laga (fixture {fid}) ==")
        o = parse_odds(af_get("/odds", {"fixture": fid}))
        if o:
            p = o["p1x2"]
            print(f"Odds ({o['book']}): home {p[0]:.0f}% / seri {p[1]:.0f}% / away {p[2]:.0f}%"
                  + (f" | Over 2.5 {o['o25']:.0f}%" if "o25" in o else "") + (f" | kedua tim cetak gol {o['btts']:.0f}%" if "btts" in o else ""))
        else:
            print("Odds: belum tersedia untuk laga ini")
        inj = af_get("/injuries", {"fixture": fid}).get("response") or []
        print(f"Cedera/skorsing tercatat: {len(inj)}" + ("  contoh: " + "; ".join(f'{i["player"]["name"]} ({i["team"]["name"]}, {i["player"].get("reason")})' for i in inj[:4]) if inj else ""))
        lu = af_get("/fixtures/lineups", {"fixture": fid}).get("response") or []
        print("Lineup resmi: " + (f"tersedia ({len(lu)} tim)" if lu else "belum keluar (normal kalau masih jauh dari kick-off)"))
    print(f"\nSisa kuota request hari ini: {_last_left}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        print(__doc__)
