"""
Sumber data UEFA Europa League (UEL) lewat The Odds API, karena UEL tidak ada di paket gratis football-data.org.

Menghasilkan 'laga' berbentuk sama dengan football-data.org supaya seluruh alur agent (analisa, kombinasi, pencatatan)
bisa dipakai ulang, ditambah probabilitas pasar (_mk) dari odds. Skor akhir diambil dari endpoint /scores.

Pengaman kredit (paket gratis 500/bulan): jadwal (/events) tidak memakai kredit, odds dan skor di-cache,
dan panggilan berbayar dihentikan kalau sisa kredit di bawah MIN_CREDITS.
"""
import re, json, zlib, datetime as dt
import oddsapi as oa

UEL_SPORT = oa.SPORT_KEYS["EL"]
CACHE = "uel_cache.json"
CREST_FILE = "crests.json"
COMP = {"name": "UEFA Europa League", "code": "EL"}
MIN_CREDITS = 30
EVENTS_TTL_H, ODDS_TTL_H, SCORES_TTL_MIN = 3, 3, 25


def is_uel(name):
    n = name or ""
    return bool(re.search(r"europa league", n, re.I)) and not re.search(r"conference", n, re.I)

def _now():
    return dt.datetime.now(dt.timezone.utc)

def _parse(z):
    return dt.datetime.fromisoformat(z.replace("Z", "+00:00"))

def _load():
    try:
        with open(CACHE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save(c):
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(c, f, ensure_ascii=False)

def match_id(event_id):
    """ID laga stabil (bilangan bulat besar supaya tidak bentrok dengan ID football-data.org)."""
    return 10 ** 9 + zlib.crc32(event_id.encode()) % (10 ** 9)

def _fresh(c, key, ttl_min):
    t = c.get(key + "_at")
    return bool(t) and (_now() - _parse(t)).total_seconds() / 60 < ttl_min

def _credits_ok(c):
    left = c.get("credits")
    return left is None or int(left) >= MIN_CREDITS

def _track(c):
    if oa.QUOTA.get("remaining") is not None:
        try:
            c["credits"] = int(oa.QUOTA["remaining"])
        except ValueError:
            pass


# ---------------------------------------------------------------- tim
def _team_index():
    """Peta nama tim football-data.org -> id dan nama pendek (dari crests.json yang dibuat agent)."""
    try:
        with open(CREST_FILE, encoding="utf-8") as f:
            c = json.load(f)
        return c.get("ids") or {}, c.get("shorts") or {}
    except Exception:
        return {}, {}

def _team(name, ids, shorts):
    m = oa.best_match(name, ids)
    if m:                                    # tim yang kita punya datanya (Elo, logo, form)
        return {"id": m[1], "name": m[0], "shortName": shorts.get(m[0]) or m[0]}
    return {"id": -(zlib.crc32(" ".join(oa.norm_name(name)).encode()) % 10 ** 9) - 1,
            "name": name, "shortName": name}


# ---------------------------------------------------------------- jadwal dan odds
def _events(c):
    if _fresh(c, "events", EVENTS_TTL_H * 60) and c.get("events") is not None:
        return c["events"]
    evs = oa.oa_get(f"/sports/{UEL_SPORT}/events")
    _track(c)
    if evs is None:
        return c.get("events")               # pakai cache lama kalau ada
    c["events"], c["events_at"] = evs, _now().isoformat()
    return evs

def _odds(c):
    if _fresh(c, "odds", ODDS_TTL_H * 60) and c.get("odds") is not None:
        return c["odds"]
    if not _credits_ok(c):
        print(f"[uel] sisa kredit The Odds API rendah ({c.get('credits')}), odds tidak diperbarui")
        return c.get("odds") or {}
    data = oa.oa_get(f"/sports/{UEL_SPORT}/odds", {"regions": "eu", "markets": "h2h,totals", "oddsFormat": "decimal"})
    _track(c)
    if data is None:
        return c.get("odds") or {}
    out = {}
    for ev in data:
        o = oa.parse_event_odds(ev)
        if o:
            out[ev["id"]] = o
    c["odds"], c["odds_at"] = out, _now().isoformat()
    return out

def matches_in_window(min_min=20, max_min=24 * 60):
    """Laga UEL yang kick-off-nya antara min_min dan max_min menit lagi, berbentuk laga football-data.org
    (+ '_mk' = probabilitas pasar kalau odds tersedia). None kalau jadwal gagal diambil."""
    if not oa.KEY:
        return []
    c = _load()
    evs = _events(c)
    if evs is None:
        _save(c)
        return None
    now = _now()
    sel = [e for e in evs if min_min <= (_parse(e["commence_time"]) - now).total_seconds() / 60 <= max_min]
    if not sel:
        _save(c)
        return []
    odds = _odds(c)
    ids, shorts = _team_index()
    out = []
    for e in sorted(sel, key=lambda e: e["commence_time"]):
        m = {"id": match_id(e["id"]), "status": "TIMED", "utcDate": e["commence_time"], "competition": dict(COMP),
             "homeTeam": _team(e["home_team"], ids, shorts), "awayTeam": _team(e["away_team"], ids, shorts)}
        o = odds.get(e["id"])
        if o:
            m["_mk"] = tuple(o["p1x2"])
            m["_od"], m["_ob"] = tuple(o["avg"]), tuple(o["best"])
            if "o25" in o:
                m["_o25"] = o["o25"]
        out.append(m)
    _save(c)
    return out


# ---------------------------------------------------------------- skor akhir
def finished(want_ids):
    """Laga UEL yang sudah selesai dari daftar id, dalam bentuk laga football-data.org ('status' dan 'score')."""
    if not want_ids or not oa.KEY:
        return {}
    c = _load()
    if not _fresh(c, "scores", SCORES_TTL_MIN) or c.get("scores") is None:
        if _credits_ok(c):
            data = oa.oa_get(f"/sports/{UEL_SPORT}/scores", {"daysFrom": 3})
            _track(c)
            if data is not None:
                c["scores"] = [{k: e.get(k) for k in ("id", "completed", "home_team", "away_team", "scores")} for e in data]
                c["scores_at"] = _now().isoformat()
        else:
            print(f"[uel] sisa kredit rendah ({c.get('credits')}), skor tidak diperbarui")
    out = {}
    for ev in c.get("scores") or []:
        mid = match_id(ev["id"])
        r = oa.parse_scores(ev)
        if r and mid in want_ids:
            gh, ga = r
            out[mid] = {"id": mid, "status": "FINISHED",
                        "score": {"winner": "HOME_TEAM" if gh > ga else "AWAY_TEAM" if ga > gh else "DRAW",
                                  "fullTime": {"home": gh, "away": ga}}}
    _save(c)
    return out
