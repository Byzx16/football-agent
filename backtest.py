"""
Backtest: menguji model statistik agent (Elo dan model gol Poisson) pada hasil musim-musim lalu.

Cara kerja (walk-forward): tiap laga diprediksi hanya dari laga sebelumnya, lalu model diperbarui.
Parameter disetel pada 70% laga evaluasi pertama dan dinilai pada 30% terakhir (hold-out),
supaya hasilnya tidak mengelabui diri sendiri.
Bagian AI (Gemini) tidak bisa di-backtest karena tidak ada berita historis.

Pemakaian:
    python backtest.py              hasil dicetak di terminal
    python backtest.py --telegram   ringkasan juga dikirim ke Telegram
"""
import sys, json, math, itertools, html, datetime as dt
import agent_free as a

LEAGUES_BT = "PL,PD,SA,BL1,FL1"      # liga domestik saja (UCL dilewati: lintas liga tidak sebanding)
CACHE = "backtest_cache.json"
WARMUP = 500                          # laga awal hanya untuk pemanasan, tidak dinilai
FACT = [math.factorial(i) for i in range(12)]
LINE = "━━━━━━━━━━━━━━━━━━"


# ---------------------------------------------------------------- data
def load_matches():
    now = dt.datetime.now(dt.timezone.utc)
    try:
        with open(CACHE, encoding="utf-8") as f:
            c = json.load(f)
        if c.get("matches") and now - dt.datetime.fromisoformat(c["built"]) < dt.timedelta(days=7):
            return c["matches"]
    except Exception:
        pass
    cur = now.year if now.month >= 7 else now.year - 1
    out = []
    for season in (cur - 2, cur - 1, cur):
        for code in LEAGUES_BT.split(","):
            data = a.fd_get(f"/competitions/{code}/matches", {"season": season, "status": "FINISHED"})
            for m in (data or {}).get("matches", []):
                ft = (m.get("score") or {}).get("fullTime") or {}
                if ft.get("home") is None or ft.get("away") is None:
                    continue
                out.append([m["utcDate"], code, m["homeTeam"]["id"], m["awayTeam"]["id"],
                            ft["home"], ft["away"], season])
    out.sort(key=lambda x: x[0])
    if out:
        with open(CACHE, "w", encoding="utf-8") as f:
            json.dump({"built": now.isoformat(), "matches": out}, f)
    return out


# ---------------------------------------------------------------- model
def pois_grid(lh, la, n=11):
    ph = [math.exp(-lh) * lh ** i / FACT[i] for i in range(n)]
    pa = [math.exp(-la) * la ** i / FACT[i] for i in range(n)]
    p1 = px = p2 = o15 = o25 = o35 = bt = tot = 0.0
    for i in range(n):
        for j in range(n):
            p = ph[i] * pa[j]
            tot += p
            if i > j:
                p1 += p
            elif i == j:
                px += p
            else:
                p2 += p
            t = i + j
            if t >= 2:
                o15 += p
            if t >= 3:
                o25 += p
            if t >= 4:
                o35 += p
            if i >= 1 and j >= 1:
                bt += p
    return [p1 / tot, px / tot, p2 / tot], [o15 / tot, o25 / tot, o35 / tot, bt / tot]


def sim_elo(ms, K, HFA, d0, g=0.6, carry=0.7):
    R, out, season = {}, [], ms[0][6]
    for _, _, h, aw, gh, ga, sea in ms:
        if sea != season:
            R = {k: 1500 + (v - 1500) * carry for k, v in R.items()}
            season = sea
        rh, ra = R.get(h, 1500.0), R.get(aw, 1500.0)
        e = 1 / (1 + 10 ** (-((rh + HFA) - ra) / 400))
        d = d0 * (1 - abs(2 * e - 1)) ** g
        ph, pa = max(0.01, e - d / 2), max(0.01, 1 - e - d / 2)
        t = ph + d + pa
        out.append([ph / t, d / t, pa / t])
        act = 1.0 if gh > ga else 0.5 if gh == ga else 0.0
        gd = abs(gh - ga)
        mult = 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
        delta = K * mult * (act - e)
        R[h], R[aw] = rh + delta, ra - delta
    return out


def sim_goals(ms, prior, carry):
    comp, team, out, season = {}, {}, [], ms[0][6]
    for _, code, h, aw, gh, ga, sea in ms:
        if sea != season:
            for v in comp.values():
                v[0] *= carry; v[1] *= carry; v[2] *= carry
            for v in team.values():
                for i in range(4):
                    v[i] *= carry
            season = sea
        c = comp.get(code)
        avh, ava = (c[0] / c[2], c[1] / c[2]) if c and c[2] >= 30 else (1.5, 1.2)
        th, ta = team.get(h, [0.0] * 4), team.get(aw, [0.0] * 4)
        att_h, def_h = (th[0] + prior) / (th[2] + prior), (th[1] + prior) / (th[3] + prior)
        att_a, def_a = (ta[0] + prior) / (ta[2] + prior), (ta[1] + prior) / (ta[3] + prior)
        lh = max(0.2, min(4.5, avh * att_h * def_a))
        la = max(0.2, min(4.5, ava * att_a * def_h))
        out.append(pois_grid(lh, la))
        c = comp.setdefault(code, [0.0, 0.0, 0.0])
        c[0] += gh; c[1] += ga; c[2] += 1
        th, ta = team.setdefault(h, [0.0] * 4), team.setdefault(aw, [0.0] * 4)
        th[0] += gh; th[1] += ga; th[2] += avh; th[3] += ava
        ta[0] += ga; ta[1] += gh; ta[2] += ava; ta[3] += avh
    return out


# ---------------------------------------------------------------- metrik
def outcome(m):
    return 0 if m[4] > m[5] else 1 if m[4] == m[5] else 2

def brier3(ps, ys):
    return sum(sum((p[k] - (1 if k == y else 0)) ** 2 for k in range(3)) for p, y in zip(ps, ys)) / len(ys)

def acc3(ps, ys):
    return sum(1 for p, y in zip(ps, ys) if max(range(3), key=lambda k: p[k]) == y) / len(ys)

def brier1(ps, ys):
    return sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ys)

def base_rates(ys):
    cnt, out = [4.5, 2.7, 2.8], []
    for y in ys:
        t = sum(cnt)
        out.append([c / t for c in cnt])
        cnt[y] += 1
    return out

def goal_actuals(ms):
    out = []
    for m in ms:
        tg = m[4] + m[5]
        out.append([tg >= 2, tg >= 3, tg >= 4, m[4] > 0 and m[5] > 0])
    return out

def goal_base(acts):
    cnt, n, out = [7.0, 5.0, 3.0, 5.0], 10.0, []
    for act in acts:
        out.append([c / n for c in cnt])
        for k in range(4):
            cnt[k] += 1 if act[k] else 0
        n += 1
    return out

def pick(lst, idx):
    return [lst[i] for i in idx]

def calibration(ps, ys):
    bins = [(0, .40, "di bawah 40%"), (.40, .50, "40-50%"), (.50, .60, "50-60%"), (.60, .70, "60-70%"), (.70, 1.01, "70% ke atas")]
    rows = []
    for lo, hi, lab in bins:
        sel = [(max(p), max(range(3), key=lambda k: p[k]) == y) for p, y in zip(ps, ys) if lo <= max(p) < hi]
        if sel:
            rows.append((lab, len(sel), sum(s[0] for s in sel) / len(sel) * 100, sum(s[1] for s in sel) / len(sel) * 100))
    return rows


# ---------------------------------------------------------------- utama
def run(ms):
    n = len(ms)
    ys = [outcome(m) for m in ms]
    split = WARMUP + int((n - WARMUP) * 0.7)
    tune, hold = range(WARMUP, split), range(split, n)

    # --- Elo: bawaan vs disetel
    default = sim_elo(ms, a.ELO_K, a.ELO_HFA, 0.28)
    best_elo, best_b = None, 9.0
    for K, HFA, d0 in itertools.product((12, 20, 30), (40, 65, 90), (0.24, 0.28, 0.32)):
        ps = sim_elo(ms, K, HFA, d0)
        b = brier3(pick(ps, tune), pick(ys, tune))
        if b < best_b:
            best_b, best_elo = b, (K, HFA, d0, ps)
    K, HFA, d0, tuned = best_elo

    # --- Model gol: bawaan vs disetel
    acts = goal_actuals(ms)
    def goal_obj(gs, idx):
        b = brier3([gs[i][0] for i in idx], [ys[i] for i in idx])
        g = sum(brier1([gs[i][1][k] for i in idx], [1 if acts[i][k] else 0 for i in idx]) for k in range(4)) / 4
        return b + g
    gdef = sim_goals(ms, a.GOAL_PRIOR, 0.5)
    best_g, best_gb = None, 9.0
    for prior, carry in itertools.product((4, 8, 16), (0.5, 0.75, 1.0)):
        gs = sim_goals(ms, prior, carry)
        b = goal_obj(gs, tune)
        if b < best_gb:
            best_gb, best_g = b, (prior, carry, gs)
    prior, carry, gtuned = best_g

    # --- campuran Elo (disetel) dan 1X2 dari Poisson (disetel)
    best_w, best_wb = 1.0, 9.0
    for w in (0, .25, .5, .75, 1.0):
        mix = [[w * e + (1 - w) * p for e, p in zip(tuned[i], gtuned[i][0])] for i in range(n)]
        b = brier3(pick(mix, tune), pick(ys, tune))
        if b < best_wb:
            best_wb, best_w = b, w
    mix = [[best_w * e + (1 - best_w) * p for e, p in zip(tuned[i], gtuned[i][0])] for i in range(n)]

    base = base_rates(ys)
    yh = pick(ys, hold)
    res = {
        "n": n, "n_tune": len(tune), "n_hold": len(hold),
        "home_acc": sum(1 for y in yh if y == 0) / len(yh),
        "base": (acc3(pick(base, hold), yh), brier3(pick(base, hold), yh)),
        "elo_default": (acc3(pick(default, hold), yh), brier3(pick(default, hold), yh)),
        "elo_tuned": (acc3(pick(tuned, hold), yh), brier3(pick(tuned, hold), yh)),
        "pois": (acc3([gtuned[i][0] for i in hold], yh), brier3([gtuned[i][0] for i in hold], yh)),
        "mix": (acc3(pick(mix, hold), yh), brier3(pick(mix, hold), yh)),
        "elo_params": (K, HFA, d0), "goal_params": (prior, carry), "w_elo": best_w,
        "goal_default_obj": goal_obj(gdef, hold), "goal_tuned_obj": goal_obj(gtuned, hold),
    }
    gb = goal_base(acts)
    res["goals"] = []
    for k, lab in enumerate(("Over 1.5", "Over 2.5", "Over 3.5", "Kedua tim cetak gol")):
        yk = [1 if acts[i][k] else 0 for i in hold]
        pm = [gtuned[i][1][k] for i in hold]
        pb = [gb[i][k] for i in hold]
        am = sum(1 for p, y in zip(pm, yk) if (p >= .5) == (y == 1)) / len(yk)
        res["goals"].append((lab, brier1(pm, yk), brier1(pb, yk), am))
    res["calib"] = calibration(pick(mix, hold), yh)
    return res


def report(r):
    f = lambda x: f"{x * 100:.1f}%"
    L = [f"🧪 <b>BACKTEST</b>  ({r['n']} laga; setel {r['n_tune']}, uji hold-out {r['n_hold']})", LINE,
         "<b>Hasil 1X2 pada hold-out</b> <i>(laga yang tidak dipakai untuk menyetel)</i>",
         f"Selalu tuan rumah: akurasi {f(r['home_acc'])}",
         f"Base-rate: akurasi {f(r['base'][0])}, Brier {r['base'][1]:.3f}",
         f"Elo bawaan: akurasi {f(r['elo_default'][0])}, Brier {r['elo_default'][1]:.3f}",
         f"Elo disetel: akurasi {f(r['elo_tuned'][0])}, Brier {r['elo_tuned'][1]:.3f}",
         f"Poisson 1X2: akurasi {f(r['pois'][0])}, Brier {r['pois'][1]:.3f}",
         f"Campuran (bobot Elo {r['w_elo']:.2f}): akurasi {f(r['mix'][0])}, Brier {r['mix'][1]:.3f}", LINE,
         "<b>Pasar gol (model Poisson disetel) pada hold-out</b>"]
    for lab, bm, bb, am in r["goals"]:
        L.append(f"{lab}: Brier {bm:.3f} vs base-rate {bb:.3f}, lean benar {f(am)}")
    L += [LINE, "<b>Kalibrasi model campuran</b> <i>(peluang yang diberikan vs kenyataan)</i>"]
    for lab, cnt, pr, ac in r["calib"]:
        L.append(f"{lab}: diberi {pr:.0f}%, benar {ac:.0f}% ({cnt} laga)")
    K, HFA, d0 = r["elo_params"]
    gain_e = r["elo_default"][1] - r["elo_tuned"][1]
    gain_g = r["goal_default_obj"] - r["goal_tuned_obj"]
    gain_m = r["elo_tuned"][1] - r["mix"][1]
    L += [LINE, "<b>Kesimpulan</b>"]
    L.append("✅ Elo mengalahkan base-rate." if r["elo_tuned"][1] < r["base"][1] - 0.002 else "⚠️ Elo belum jelas mengalahkan base-rate.")
    L.append("✅ Model akurasi di atas 'selalu tuan rumah'." if r["mix"][0] > r["home_acc"] else "⚠️ Akurasi belum melampaui 'selalu tuan rumah'.")
    if gain_e > 0.001:
        L.append(f"Setel Elo: ELO_K={K}, ELO_HFA={HFA}, dan angka 0.28 di elo_predict jadi {d0} (Brier hold-out membaik {gain_e:.3f}).")
    else:
        L.append("Parameter Elo bawaan sudah cukup, tidak perlu diubah.")
    if gain_g > 0.001:
        L.append(f"Setel model gol: GOAL_PRIOR={r['goal_params'][0]}, bobot musim lalu {r['goal_params'][1]} (membaik {gain_g:.3f}).")
    else:
        L.append("Parameter model gol bawaan sudah cukup.")
    if gain_m > 0.003 and r["w_elo"] < 1:
        L.append("Menambah probabilitas Poisson ke campuran 1X2 membantu, layak dipasang.")
    return "\n".join(L)


def main():
    ms = load_matches()
    if len(ms) < WARMUP + 300:
        text = f"Data historis terlalu sedikit untuk backtest ({len(ms)} laga). Tier gratis mungkin tidak menyediakan musim lalu."
    else:
        text = report(run(ms))
    print(html.unescape(__import__("re").sub(r"<[^>]+>", "", text)))
    if "--telegram" in sys.argv:
        a.send_telegram(text)


if __name__ == "__main__":
    main()
