"""
Odds pasar "bayangan" untuk liga domestik dan UCL lewat The Odds API (market h2h saja = 1 kredit per panggilan).

Dipakai hanya untuk mencatat odds (simulasi ROI dan nilai/value). TIDAK mengubah tebakan agent.
Pengaman kredit: cache 5 jam per kompetisi, dan berhenti kalau sisa kredit di bawah MIN_CREDITS
(UEL punya prioritas karena odds-nya dipakai sebagai sinyal).
"""
import json, datetime as dt
import oddsapi as oa

CACHE = "odds_cache.json"
SPORTS = {c: oa.SPORT_KEYS[c] for c in ("PL", "PD", "SA", "BL1", "FL1", "CL")}
TTL_H = 5
MIN_CREDITS = 100


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

def _credits(c):
    """Sisa kredit terendah yang diketahui (dari cache ini dan cache UEL)."""
    vals = []
    if c.get("credits") is not None:
        vals.append(int(c["credits"]))
    try:
        with open("uel_cache.json", encoding="utf-8") as f:
            u = json.load(f).get("credits")
        if u is not None:
            vals.append(int(u))
    except Exception:
        pass
    return min(vals) if vals else None

def _events(c, code):
    """Odds h2h semua laga mendatang untuk satu kompetisi (di-cache)."""
    box = c.setdefault("sports", {}).get(code) or {}
    if box.get("at") and (_now() - _parse(box["at"])).total_seconds() < TTL_H * 3600:
        return box.get("events") or []
    left = _credits(c)
    if left is not None and left < MIN_CREDITS:
        print(f"[odds] sisa kredit rendah ({left}), odds {code} tidak diperbarui")
        return box.get("events") or []
    data = oa.oa_get(f"/sports/{SPORTS[code]}/odds", {"regions": "eu", "markets": "h2h", "oddsFormat": "decimal"})
    if oa.QUOTA.get("remaining") is not None:
        try:
            c["credits"] = int(oa.QUOTA["remaining"])
        except ValueError:
            pass
    if data is None:
        return box.get("events") or []
    evs = []
    for ev in data:
        o = oa.parse_event_odds(ev)
        if o:
            evs.append({"t": ev["commence_time"], "h": ev["home_team"], "a": ev["away_team"],
                        "p": o["p1x2"], "od": o["avg"], "ob": o["best"]})
    c["sports"][code] = {"at": _now().isoformat(), "events": evs}
    return evs

def _find(m, events):
    ko = _parse(m["utcDate"])
    best = None
    for e in events:
        if abs((_parse(e["t"]) - ko).total_seconds()) / 60 > 120:
            continue
        s = (oa.name_sim(m["homeTeam"]["name"], e["h"]) + oa.name_sim(m["awayTeam"]["name"], e["a"])) / 2
        if best is None or s > best[0]:
            best = (s, e)
    return best[1] if best and best[0] >= 0.6 else None

def attach(matches):
    """Pasang '_mkshadow' (probabilitas pasar), '_od' (odds rata-rata), dan '_ob' (odds terbaik) pada laga football-data.org."""
    if not oa.KEY:
        return
    by = {}
    for m in matches:
        if m["competition"]["code"] in SPORTS:
            by.setdefault(m["competition"]["code"], []).append(m)
    if not by:
        return
    c = _load()
    for code, ms in by.items():
        evs = _events(c, code)
        for m in ms:
            e = _find(m, evs)
            if e:
                m["_mkshadow"], m["_od"], m["_ob"] = tuple(e["p"]), tuple(e["od"]), tuple(e["best"] if "best" in e else e["ob"])
    _save(c)
