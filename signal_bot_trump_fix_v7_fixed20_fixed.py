
def _format_consolidation_block(dbg: dict, lvl: float) -> str:
    """Возвращает читаемый блок о консолидации на русском.
    Ожидаемые keys в dbg: level_type ('support'/'resistance'), nl, nh, ca, cb, tol, atr
    lvl — используемый уровень (float)
    """
    try:
        if not isinstance(dbg, dict):
            return ""
        lt = (dbg.get('level_type') or '').lower()
        role = 'уровень'
        if lt == 'support':
            role = 'уровень поддержки'
        elif lt == 'resistance':
            role = 'уровень сопротивления'
        nl = dbg.get('nl'); nh = dbg.get('nh')
        ca = dbg.get('ca'); cb = dbg.get('cb')
        tol = dbg.get('tol'); atr = dbg.get('atr')
        try:
            tol_abs = float(tol) if tol is not None else None
            lvl_f = float(lvl) if lvl is not None else None
            tol_pct = (abs(tol_abs) / max(1e-9, lvl_f) * 100.0) if (tol_abs is not None and lvl_f) else None
        except Exception:
            tol_pct = None
        lines = [
            "🔎 Консолидация у уровня:",
            f"• Уровень: {lvl:.6f} ({role})" if lvl is not None else f"• Уровень: {role}",
            f"• Касаний рядом: снизу {nl} / сверху {nh}",
            f"• Закрытий свечей: выше {ca} / ниже {cb}",
            (f"• Допуск к уровню: ±{tol_abs:.3f} (≈{tol_pct:.2f}%)" if tol_pct is not None else (f"• Допуск к уровню: ±{tol_abs}" if tol is not None else None)),
            (f"• ATR: {float(atr):.3f}" if atr is not None else None),
        ]
        return "\n".join([ln for ln in lines if ln])
    except Exception:
        return ""


# === Lightweight NN gate (self-contained) ===
# Переключатель NN. Если есть внешний модуль/модель, выставьте True и переопределите nn_score_for_signal()/dynamic_threshold().
_NN_AVAILABLE = True

def _features_quality(arr):
    import numpy as np
    a = np.asarray(arr, dtype=float).ravel()
    nz = int(np.count_nonzero(np.abs(a) > 1e-12))
    uniq = int(len(np.unique(np.round(a, 6))))
    std = float(np.std(a))
    return {"nonzero": nz, "uniq": uniq, "std": std}

def _is_degenerate_features(arr) -> bool:
    m = _features_quality(arr)
    # мало ненулевых, мало уникальных значений, почти нулевая дисперсия
    return (m["nonzero"] <= 3) or (m["uniq"] <= 2) or (m["std"] < 1e-6)

def _sigmoid(x):
    try:
        import math
        return 1 / (1 + math.exp(-float(x)))
    except Exception:
        return 0.5

def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def _clamp(x, lo, hi):
    try:
        x = float(x)
        if x < lo: return lo
        if x > hi: return hi
        return x
    except Exception:
        return lo

# === ПОДМЕНА NN ===
_NN_AVAILABLE = True

def nn_score_for_signal(symbol: str, side: str, zone: dict):
    """
    Возвращает (p_cal, ctx), где p_cal — калиброванная вероятность успеха сделки.
    При любой ошибке — мягкий фолбэк к старой эвристике.
    """
    try:
        if MODEL is None or not FEATURES:
            raise RuntimeError("nn model or features missing")
        X = make_features_for_nn(symbol, side, zone)  # твоя функция
        Xs = SCALE(X)
        try:
            if _is_degenerate_features(X) or _is_degenerate_features(Xs):
                # фолбэк: считаем p_nn примерно как эвристическую вероятность
                p_nn = max(0.0, min(1.0, float(prob) / 100.0))
                zone["nn_p"] = p_nn
                zone["nn_thr"] = zone.get("nn_thr") or 0.01
            else:
                p_nn = model.predict_proba([Xs])[0][POS_IDX]
                zone["nn_p"] = float(p_nn)
        except Exception:
            pass
        p_raw = float(_predict_proba(Xs).ravel()[0])
        if ORIENT_INV:
            p_raw = 1.0 - p_raw
        p_cal = float(CAL([p_raw])[0])
        return p_cal, {"p_raw": p_raw, "p_cal": p_cal}
    except Exception as e:
        # Фолбэк к старой эвристике, чтобы не ронять рассылку
        try:
            prob, ctx = _legacy_nn_proxy(symbol, side, zone)
            return prob, {"fallback": True, **(ctx or {})}
        except Exception:
            return 0.5, {"error": str(e)}

# сохраняем твою старую эвристику как внутренний фолбэк
def _legacy_nn_proxy(symbol, side, zone):
    prob = _safe_float(zone.get('probability_pct'), None)
    if prob is None:
        prob = 50.0
    prob01 = _clamp(prob/100.0, 0.0, 1.0)
    z = 1.6*(prob01-0.5)  # очень упрощённо
    p = _clamp(_sigmoid(z), 0.05, 0.98)
    return p, {"prob01": prob01}

def dynamic_threshold(symbol: str, side: str, zone: dict, prob: float, rr_ratio: float):
    """
    Порог берём из групповых порогов thresholds.json (если есть),
    иначе — из threshold.json, иначе — 0.66.
    """
    try:
        return float(nn_group_threshold(symbol, side))
    except Exception:
        return 0.66

def get_ohlc_15m(symbol, limit=60):
    """
    Возвращает последние 15-минутные свечи в формате [(t,o,h,l,c,v), ...].
    Использует кеширующую функцию fetch_cached_ohlcv, чтобы не делать лишних запросов.
    """
    return fetch_cached_ohlcv(symbol, "15", limit=limit)

def is_strong_breakout(symbol, side, level):
    """
    Сильный пробой 15m:
    - тело свечи >= 50% диапазона
    - объём >= 1.5× среднего по 50 последним 15m
    - свеча пересекает уровень в сторону сделки
    Использует fetch_cached_ohlcv -> кортежи из 5 полей: (o, h, l, c, v)
    """
    ohlc_15 = fetch_cached_ohlcv(symbol, "15", limit=60)
    if not ohlc_15 or len(ohlc_15) < 51:
        return False

    # каждая свеча: (o, h, l, c, v) — БЕЗ времени
    try:
        o, h, l, c, v = ohlc_15[-1]
    except ValueError:
        # если вдруг пришла «кривая» свеча, не ломаемся
        return False

    rng = h - l
    if rng <= 0:
        return False

    body_ratio = abs(c - o) / rng
    crosses = (c > level and o < level) if side == "long" else (c < level and o > level)

    # средний объём по 50 предыдущим свечам
    vols = []
    for bar in ohlc_15[-51:-1]:
        try:
            vols.append(float(bar[4]))
        except (IndexError, ValueError, TypeError):
            continue
    if not vols:
        return False
    avg50 = sum(vols) / len(vols)

    vol_ok = v >= 1.5 * avg50

    return (body_ratio >= 0.50) and crosses and vol_ok


    reasons = []
    if side == 'long':
        reasons.append("цена выше уровня поддержки")
        if zone.get("consolidation"): reasons.append("идёт консолидация у уровня")
        if zone.get("type") == "Поджатие": reasons.append("наблюдается поджатие к уровню")
        if zone.get("type") == "Отскок после импульса": reasons.append("цена резко подошла к уровню — возможен отскок")
        if zone.get("type") == "Затухание": reasons.append("волатильность снижается — возможен выход из фазы накопления")
        return "🟢 LONG, потому что " + ", ".join(reasons)
    else:
        reasons.append("цена у уровня сопротивления")
        if zone.get("consolidation"): reasons.append("идёт консолидация у уровня")
        if zone.get("type") == "Поджатие": reasons.append("наблюдается поджатие снизу")
        if zone.get("type") == "Отскок после импульса": reasons.append("цена резко подошла к уровню — возможен отскок вниз")
        if zone.get("type") == "Затухание": reasons.append("волатильность снижается")
        return "🔴 SHORT, потому что " + ", ".join(reasons)


# ✅ Обновлён: теперь поддерживается несколько зон входа, ключ alerted построен по каждой зоне


# ------------------------------ НАСТРОЙКИ ------------------------------
import os
import json
import time
import requests
import telebot
import threading
import base64
import pandas as pd
from datetime import datetime
from schedule import Scheduler
import schedule
sched = schedule
from tradingview_ta import TA_Handler, Interval, Exchange

ENABLE_DAILY_DIGEST = False  # выключает ежедневный дайджест
STRICT_NN     = globals().get('STRICT_NN', False)
NN_SOFT_EPS   = globals().get('NN_SOFT_EPS', 0.005)
PROB_MIN = 58     # было 60 — часто ровно 60 и резалось
RR_MIN        = globals().get('RR_MIN', 1.50)
EPS_RR        = globals().get('EPS_RR', 1e-3)         # допуск на флоат-ошибку RR
# --- Safe defaults (если не заданы где-то ещё) ---
PREF_BREAKOUT = False
PROB_MIN      = globals().get('PROB_MIN', 60.0)
ALLOW_REBOUND = True
STRICT_TREND_GATE = False  # False => достаточно, чтобы тренд ок либо по инструменту, либо по BTC



import json, os, time
THR_DEFAULT = 0.01
THR_PATH = "./nn_eval_out/thresholds.json"
_thr_cache = {"global_f1": THR_DEFAULT}
_thr_mtime = 0.0

def _load_thr_cache():
    global _thr_cache, _thr_mtime
    try:
        st = os.stat(THR_PATH).st_mtime
        if st != _thr_mtime:
            with open(THR_PATH, "r", encoding="utf-8") as f:
                _thr_cache = json.load(f)
            _thr_mtime = st
    except Exception:
        _thr_cache = {"global_f1": THR_DEFAULT}

def get_nn_threshold(symbol: str, side: str) -> float:
    # A) групповые из EVAL_DIR/thresholds.json или ART_DIR/thresholds.json
    try:
        g = nn_group_threshold(symbol, side)
        if isinstance(g, (int, float)) and g > 0:
            return float(g)
    except Exception:
        pass
    # B) локальный thresholds.json рядом с ботом
    try:
        _load_thr_cache()
        grp_key = "majors" if is_major(symbol) else "alts"
        for k in (f"{symbol}:{side}", symbol, grp_key, "prec60", "global_f1"):
            v = _thr_cache.get(k)
            if isinstance(v, (int, float)) and v > 0:
                return float(v)
    except Exception:
        pass
    # C) глобальный из EVAL_DIR (threshold_auto.json)
    try:
        if isinstance(GLOBAL_THR, (int, float)) and GLOBAL_THR > 0:
            return float(GLOBAL_THR)
    except Exception:
        pass
    # D) дефолт
    return THR_DEFAULT

# === NN RUNTIME (склейка c eval/trainer артефактами) =========================
from pathlib import Path
import json, math, numpy as np

# === Директории артефактов ===
# Модель/скейлер остаются в train-папке:
ART_DIR  = Path(r"C:\Users\music\OneDrive\Desktop\Бот скрипты\out_nn_bot\nn_train_out_cal2")
# Порог(и), калибровка, отчёты — в eval-папке:
EVAL_DIR = Path(r"C:\Users\music\OneDrive\Desktop\Бот скрипты\out_nn_bot\nn_eval_out_auto_inv")
# feature order
FEATURES = []
try:
    FEATURES = json.load(open(ART_DIR/"feature_order.json","r",encoding="utf-8"))
except Exception as e:
    print(f"[nn] feature_order.json missing: {e}")


def _vol_ratio(vol_series, n=20):
    try:
        arr = np.asarray(vol_series[-n:], dtype=float)
        med = float(np.median(arr[:-1])) if len(arr) > 1 else 1.0
        med = med if med > 0 else 1.0
        return float(arr[-1] / med)
    except Exception:
        return 1.0

def classify_po(side: str, price: float, zone: dict, ohlcv_15m: list):
    """
    Возвращает 'Импульсный пробой' или 'Отбой' по положению входа относительно уровня
    и факту недавнего пересечения уровня в 15m данных.
    """
    lvl = float(zone.get('lvl') or zone.get('level_1h') or zone.get('level_4h') or price)
    atr = float(zone.get('atr') or 0.0)
    tol = max(atr * 0.1, lvl * 0.0005)  # допуск: кусок ATR или 5 бп

    # входной коридор, если уже рассчитан
    z = zone.get('zone')
    if isinstance(z, (list, tuple)) and len(z) >= 2:
        entry_low, entry_high = float(z[0]), float(z[1])
    else:
        entry_low = entry_high = price

    # флаги "мы уже по ту сторону уровня"
    if side.lower() == 'long':
        beyond = entry_low >= (lvl + tol)
    else:  # short
        beyond = entry_high <= (lvl - tol)

    # проверка недавнего пересечения (2 последние свечи)
    crossed_up = crossed_down = False
    try:
        # ohlcv: [ts, open, high, low, close, volume]
        o1, c1 = float(ohlcv_15m[-2][1]), float(ohlcv_15m[-2][4])
        o2, c2 = float(ohlcv_15m[-1][1]), float(ohlcv_15m[-1][4])
        # был ли ап-кросс (из-под уровня вверх) или даун-кросс
        crossed_up   = (o1 < lvl <= c1) or (o2 < lvl <= c2)
        crossed_down = (o1 > lvl >= c1) or (o2 > lvl >= c2)
    except Exception:
        pass

    breakout = False
    if side.lower() == 'long':
        breakout = beyond or crossed_up
    else:
        breakout = beyond or crossed_down

    return 'Импульсный пробой' if breakout else 'Отбой'

def _extract_ohlcv_compact(candle):
    """
    Вернёт (o, h, l, c, v) из любой разумной формы:
      - [ts, o, h, l, c, v]
      - [o, h, l, c, v]
      - [o, h, l, c]
      - {"open":..,"high":..,"low":..,"close":..,"volume":..}
      - {"o":..,"h":..,"l":..,"c":..,"v":..}
    Если чего-то нет — v=0.0.
    """
    if candle is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # dict
    if isinstance(candle, dict):
        o = float(candle.get("open", candle.get("o", 0.0)))
        h = float(candle.get("high", candle.get("h", 0.0)))
        l = float(candle.get("low",  candle.get("l", 0.0)))
        c = float(candle.get("close", candle.get("c", 0.0)))
        v = float(candle.get("volume", candle.get("v", 0.0)))
        return o, h, l, c, v

    # list/tuple
    if isinstance(candle, (list, tuple)):
        n = len(candle)
        if n >= 6:
            _, o, h, l, c, v = candle[:6]
            return float(o), float(h), float(l), float(c), float(v)
        if n == 5:
            o, h, l, c, v = candle
            return float(o), float(h), float(l), float(c), float(v)
        if n == 4:
            o, h, l, c = candle
            return float(o), float(h), float(l), float(c), 0.0

    # fallback
    return 0.0, 0.0, 0.0, 0.0, 0.0


def _vol_ratio_from_ohlcv(ohlcv_15m, n=20):
    """Считает отношение последнего объёма к медиане предыдущих n-1.
       Корректно работает, даже если нет объёма (v=0) — тогда вернёт 1.0."""
    try:
        tail = ohlcv_15m[-n:] if len(ohlcv_15m) >= n else ohlcv_15m
        vols = []
        for c in tail:
            _, _, _, _, v = _extract_ohlcv_compact(c)
            vols.append(float(v or 0.0))
        if not vols:
            return 1.0
        med = float(np.median(vols[:-1])) if len(vols) > 1 else 1.0
        med = med if med > 0 else 1.0
        return float(vols[-1] / med)
    except Exception:
        return 1.0

# scaler
def _load_scaler():
    pkl = ART_DIR/"nn_scaler.pkl"
    if pkl.exists():
        try:
            import joblib
            scaler = joblib.load(pkl)
            return lambda X: scaler.transform(X)
        except Exception as e:
            print(f"[nn] nn_scaler.pkl failed: {e}")
    st = ART_DIR/"scaler_stats.json"
    if st.exists():
        try:
            js = json.load(open(st,"r",encoding="utf-8"))
            mean  = np.asarray(js.get("mean") or js.get("data_mean") or js.get("feature_means"), dtype=float)
            scale = np.asarray(js.get("scale") or js.get("data_scale") or js.get("feature_scales") or js.get("var"), dtype=float)
            if "var" in js:  # иногда сохраняют дисперсию
                scale = np.sqrt(np.maximum(scale, 1e-12))
            scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
            def _tr(X):
                X = np.asarray(X, dtype=float)
                return (X - mean) / scale
            return _tr
        except Exception as e:
            print(f"[nn] scaler_stats.json failed: {e}")
    return lambda X: np.asarray(X, dtype=float)

SCALE = _load_scaler()

# calibration (Platt: sigmoid(a*logit(p)+b))
def _logit(p): 
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p/(1-p))

def _identity(p): return p

# 1) Калибровка: пробуем из eval, затем train; иначе identity
CAL = _identity
def _load_calibration():
    for p in [EVAL_DIR/"calibration.json", ART_DIR/"calibration.json"]:
        try:
            if p.exists():
                cjs = json.load(open(p,"r",encoding="utf-8"))
                a = float(cjs.get("a", 1.0)); b = float(cjs.get("b", 0.0))
                def _cal(x):
                    z = a * _logit(np.asarray(x, dtype=float)) + b
                    return 1.0/(1.0 + np.exp(-z))
                print(f"[nn] calibration loaded -> {p}")
                return _cal
        except Exception as e:
            print(f"[nn] calibration load failed for {p}: {e}")
    print("[nn] calibration: identity")
    return _identity
CAL = _load_calibration()

# 2) Возможная инверсия вероятностей, если в eval были запущены с --auto-invert
ORIENT_INV = False
try:
    ori = json.load(open(EVAL_DIR/"proba_orientation.json","r",encoding="utf-8"))
    ORIENT_INV = (str(ori.get("orientation","")).lower() == "inverted")
    if ORIENT_INV:
        print("[nn] probability orientation: INVERTED")
except Exception:
    pass

# thresholds
# Глобальный порог: сначала из eval (nn_eval_auto_threshold.json / threshold_auto.json),
# затем из train (threshold.json), иначе дефолт 0.66
def _load_global_thr():
    for p in [EVAL_DIR/"nn_eval_auto_threshold.json",
              EVAL_DIR/"threshold_auto.json",
              ART_DIR/"threshold.json"]:
        try:
            if p.exists():
                js = json.load(open(p,"r",encoding="utf-8"))
                # поддержка форматов: {"threshold": 0.24} или {"metric":"f1","threshold":0.24}
                thr = js.get("threshold", None)
                if thr is not None:
                    print(f"[nn] global threshold loaded -> {p} :: {thr}")
                    return float(thr)
        except Exception as e:
            print(f"[nn] global thr load failed for {p}: {e}")
    print("[nn] global threshold: default 0.66")
    return 0.66
GLOBAL_THR = _load_global_thr()

# Групповые пороги: сначала из eval\thresholds.json, затем train\thresholds.json
def _load_group_thresholds():
    for p in [EVAL_DIR/"thresholds.json", ART_DIR/"thresholds.json"]:
        try:
            if p.exists():
                js = json.load(open(p,"r",encoding="utf-8"))
                print(f"[nn] group thresholds loaded -> {p}")
                return js
        except Exception as e:
            print(f"[nn] group thresholds load failed for {p}: {e}")
    print("[nn] group thresholds: not found")
    return None
GROUP_THR = _load_group_thresholds()


# sklearn model
MODEL = None; POS_IDX = 1
try:
    import joblib
    MODEL = joblib.load(ART_DIR/"nn_mlp.pkl")
    if hasattr(MODEL, "classes_"):
        cls = np.asarray(MODEL.classes_)
        POS_IDX = int(np.where(cls == 1)[0][0]) if 1 in cls else 1
    print(f"[nn] sklearn model loaded: POS_IDX={POS_IDX}")
except Exception as e:
    print(f"[nn] no sklearn model: {e} (бот будет использовать эвристику)")

def _predict_proba(X):
    if MODEL is None:
        raise RuntimeError("MODEL is None")
    P = MODEL.predict_proba(X)[:, POS_IDX]
    return np.asarray(P, dtype=float)

# безопасная импутация до SCALE
def _impute_safe(X_df):
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    # попробуем средние из тренировки
    try:
        sstats = json.load(open(ART_DIR/"scaler_stats.json","r",encoding="utf-8"))
        means = np.asarray(sstats.get("mean") or sstats.get("data_mean") or sstats.get("feature_means"), dtype=float)
        arr = X_df.values.astype(float)
        bad = ~np.isfinite(arr)
        if bad.any() and means.shape[0] == arr.shape[1]:
            arr[bad] = np.take(means, np.where(bad)[1])
            X_df = pd.DataFrame(arr, columns=X_df.columns)
    except Exception:
        pass
    # остатки — медианой, затем nan_to_num
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(X_df.median(numeric_only=True))
    arr = np.nan_to_num(X_df.values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    return arr

# групповый ключ для порога
def _group_key(symbol: str, side: str) -> str:
    kind = "major" if is_major(symbol) else "alt"
    return f"{side}|{kind}"

def nn_passes(p, thr) -> bool:
    if p is None or thr is None:
        return True
    if p >= thr:
        return True
    # мягкий проход
    if p >= max(0.0, float(thr) - float(NN_SOFT_EPS)):
        return True
    return False

def nn_group_threshold(symbol: str, side: str) -> float:
    if GROUP_THR:
        gk = _group_key(symbol, side)
        try:
            # поддерживаем оба формата:
            # 1) {"groups":{"short|alt":0.32, ...}}
            # 2) {"short|alt":{"threshold":0.32,...}, ...}
            if "groups" in GROUP_THR and isinstance(GROUP_THR["groups"], dict):
                node = GROUP_THR["groups"].get(gk)
            else:
                node = GROUP_THR.get(gk)
            if node is not None:
                if isinstance(node, dict):
                    val = node.get("threshold") or node.get("thr") or node.get("value")
                    return float(val)
                else:
                    return float(node)
        except Exception:
            pass
    return GLOBAL_THR
# =====================================================================



LAST_SENT_TS = {}
DEDUP_SECONDS = 2 * 60 * 60  # 2 часа
import time
# --- Helper: pretty Russian formatting for reasons (like Telegram) ---
def _format_reasons_ru(reasons:list, zone:dict)->str:
    """
    Приводит список предпосылок к читабельной русской строке, как в Telegram.
    - Дубликаты убираем, порядок стабилизируем.
    - Добавляем короткие эмодзи-маркеры для ключевых типов.
    """
    if not reasons:
        return ""
    uniq = sorted(set([str(r).strip() for r in reasons if str(r).strip()]))
    emap = {
        "Консолидация": "📦",
        "Пробой": "⚡",
        "Ложный пробой": "🎣",
        "Отбой": "↩️",
        "Нарастающий объём": "📈",
        "Дивергенция": "🪞",
        "Импульс": "🚀",
        "Импульсный пробой": "🚀⚡",
        "Поддержка": "🧱",
        "Сопротивление": "🧱",
        "ATR": "📏",
        "Завал объёма": "📉",
    }
    parts = []
    for r in uniq:
        mark = ""
        for key, emoji in emap.items():
            if key.lower() in r.lower():
                mark = emoji
                break
        parts.append(f"{mark} {r}".strip())
    txt = ", ".join(parts)
    # Добавим короткий хвост по консолидации, если есть подробный дебаг
    dbg = (zone or {}).get("consolidation_debug_str") or ""
    if dbg:
        # делаем лаконичнее: type= -> тип=; atr= -> ATR=
        dbg_ru = (dbg.replace("type=resistance","тип=сопротивление")
                      .replace("type=support","тип=поддержка")
                      .replace("tol=","допуск=")
                      .replace("atr=","ATR=")
                      .replace("nl/nh","nl/nh")
                      .replace("ca/cb","ca/cb"))
        txt = f"{txt} | {dbg_ru}"
    return txt

# === NETWORK SETUP: disable proxies and use a single session without proxies ===
import os as _os
import requests as _requests
from telebot import apihelper as _apihelper
from requests.adapters import HTTPAdapter as _HTTPAdapter
from urllib3.util.retry import Retry as _Retry

# 1) Remove any proxy environment variables
for _k in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"):
    _os.environ.pop(_k, None)

# 2) Single requests session without proxies
HTTP = _requests.Session()
HTTP.trust_env = False
HTTP.proxies = {}
HTTP.headers.update({"User-Agent": "signal-bot/1.0"})
HTTP.mount("https://", _HTTPAdapter(max_retries=_Retry(total=3, backoff_factor=0.5, status_forcelist=[429,500,502,503,504])))

# 3) Telebot: force no-proxy session
_apihelper.SESSION = _requests.Session()
_apihelper.SESSION.trust_env = False
_apihelper.SESSION.proxies = {}
_apihelper.proxy = None
# === /NETWORK SETUP ===



# ===================== ДОБАВЛЕНО: ХЕЛПЕРЫ ТРЕНДА/ATR/ЛОГА =====================
import csv

def _ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def get_df(symbol, interval="60", limit=250):
    data = fetch_cached_ohlcv(symbol, interval, limit=limit)
    if not data: 
        return None
    import pandas as _pd
    return _pd.DataFrame(data, columns=["open","high","low","close","volume"])

def atr_from_df(df, period=14):
    high = df["high"].astype(float); low = df["low"].astype(float); close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = (high - low).abs().combine((high - prev_close).abs(), max).combine((low - prev_close).abs(), max)
    return tr.rolling(window=period, min_periods=period).mean()

def is_trend_confirmed(df_1h, side):
    # EMA20 vs EMA200, минимум 1 из 3 последних свечей
    if df_1h is None or len(df_1h) < 205: 
        return False
    fast = _ema(df_1h["close"], 20)
    slow = _ema(df_1h["close"], 200)
    cond = (fast > slow) if side == "long" else (fast < slow)
    recent = cond.tail(3).astype(bool).sum()
    return recent >= 1

MAJORS = {"BTCUSDT", "ETHUSDT", "BNBUSDT"}
def is_major(symbol: str) -> bool:
    return symbol in MAJORS

def trend_gate(symbol, side, btc_symbol="BTCUSDT"):
    df_inst = get_df(symbol, "60", 260)
    df_btc  = get_df(btc_symbol, "60", 260)
    ok_inst = is_trend_confirmed(df_inst, side)
    ok_btc  = is_trend_confirmed(df_btc, side) if df_btc is not None else True
    # Дополнительная проверка по разрыву EMA20/EMA200
    try:
        if df_inst is not None and len(df_inst) >= 205:
            f_inst = _ema(df_inst["close"], 20).iloc[-1]
            s_inst = _ema(df_inst["close"], 200).iloc[-1]
            gap_inst = abs(float(f_inst) - float(s_inst)) / max(1e-9, abs(float(s_inst)))
            req_gap = 0.005 if is_major(symbol) else 0.003  # 0.5% majors, 0.3% alts
            if gap_inst < req_gap:
                ok_inst = False
        if df_btc is not None and len(df_btc) >= 205:
            f_btc = _ema(df_btc["close"], 20).iloc[-1]
            s_btc = _ema(df_btc["close"], 200).iloc[-1]
            gap_btc = abs(float(f_btc) - float(s_btc)) / max(1e-9, abs(float(s_btc)))
            if gap_btc < 0.0020:  # BTC всегда строго 0.5%
                ok_btc = False
    except Exception:
        pass
    # Для альтов допускаем «нейтраль» при очень узком зазоре <=0.2%
    if (not is_major(symbol)) and (not ok_inst) and df_inst is not None and len(df_inst) >= 205:
        try:
            f = _ema(df_inst["close"], 20).iloc[-1]
            s = _ema(df_inst["close"], 200).iloc[-1]
            gap = abs(float(f) - float(s)) / max(1e-9, abs(float(s)))
            if gap <= 0.002:
                ok_inst = True
        except Exception:
            pass
    return ok_inst and ok_btc, {"inst_ok": ok_inst, "btc_ok": ok_btc}



# === NEW: ATR-based leverage =========================================
def calc_leverage_atr(symbol: str, price: float, atr_value: float, rr_ratio=None) -> int:
    """
    Динамическое плечо от волатильности (ATR):
    - База: x10 для BTC/ETH/BNB, x12 для остальных
    - atr_pct = atr_value / price
    - Плечо ~ k / atr_pct, затем ограничение в [x3 .. base]
    - Если RR < 2 — дополнительно режем плечо вдвое (не ниже x3)
    """
    base = 10 if symbol.split("USDT")[0] in ("BTC", "ETH", "BNB") else 12
    k = 0.05 if base == 10 else 0.06  # таргет ~x10 при ~0.5–0.6% ATR
    atr_pct = (atr_value / max(1e-9, price)) if atr_value is not None else 0
    lev_raw = int(round(k / max(1e-6, atr_pct))) if atr_pct > 0 else base
    lev = max(3, min(base, lev_raw))
    if rr_ratio is not None and rr_ratio < 2.0:
        lev = max(3, lev // 2)
    return int(lev)
# ======================================================================
SKIPPED_LOG_CSV = "skipped_signals_log.csv"
def log_skipped_signal(symbol, side, reason, extra=None):
    header = ["timestamp","symbol","side","reason","extra"]
    row = [time.strftime("%Y-%m-%d %H:%M:%S"), symbol, side, reason, json.dumps(extra or {}, ensure_ascii=False)]
    write_header = not os.path.exists(SKIPPED_LOG_CSV)
    with open(SKIPPED_LOG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)
    # попытка загрузить на GitHub (не критично при ошибке)
    try:
        upload_to_github(SKIPPED_LOG_CSV, "skipped_signals_log.csv", f"append skipped {symbol} {side} ({reason})")
    except Exception as _e:
        pass

# Память SL: храним фактический SL и сравниваем с допуском + направлением
recent_sl_hits = {}

def record_sl_hit(symbol, sl_value):
    from datetime import datetime
    recent_sl_hits.setdefault(symbol, []).append((sl_value, datetime.now()))

def was_recent_sl_hit(symbol, new_sl, side, hours=4, pct_tol=0.0008):
    from datetime import datetime, timedelta
    now = datetime.now()
    records = recent_sl_hits.get(symbol, [])
    for old_sl, ts in records:
        if now - ts > timedelta(hours=hours):
            continue
        rel = abs(new_sl - old_sl) / max(1e-9, abs(old_sl))
        if rel <= pct_tol:
            return True
        # направленный допуск: deeper для long ок; выше для short ок — не считаем совпадением
        if side == "long" and new_sl <= old_sl:
            return False
        if side == "short" and new_sl >= old_sl:
            return False
    return False
# ======================================================================

import ta

# Telegram
TOKEN = '7786476338:AAEVnaL6xBkQjVo9cqv_dXhIprEzL2XD9bI'
bot = telebot.TeleBot(TOKEN)

# GitHub
GITHUB_TOKEN = "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"
REPO_OWNER = "toevich"
REPO_NAME = "crypto-signals"
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# ------------------------- ЗАГРУЗКА НА GITHUB --------------------------
def upload_to_github(file_path, repo_file_path, commit_message):
    api_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{repo_file_path}"
    with open(file_path, "rb") as f:
        content = base64.b64encode(f.read()).decode()
    get_resp = HTTP.get(api_url, headers=HEADERS)
    sha = get_resp.json().get("sha") if get_resp.status_code == 200 else None
    data = {"message": commit_message, "content": content}
    if sha:
        data["sha"] = sha
    put_resp = HTTP.put(api_url, headers=HEADERS, json=data)
    if put_resp.status_code not in [200, 201]:
        print("❌ Ошибка загрузки:", put_resp.text)

# -------------------------- УРОВНИ ПОДТВЕРЖДЕНИЯ --------------------------
def fetch_ohlcv(symbol, interval, limit=100):
    try:
        url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={interval}&limit={limit}"
        resp = HTTP.get(url)
        data = resp.json()["result"]["list"]
        return [[float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])] for x in data]  # O, H, L, C, V
    except:
        return []

def detect_common_levels(symbol):
    candles_4h = fetch_ohlcv(symbol, interval="240")
    candles_1h = fetch_ohlcv(symbol, interval="60")
    candles_15m = fetch_ohlcv(symbol, interval="15")
    if not candles_4h or not candles_1h or not candles_15m:
        return []

    def levels_from(data):
        levels = set()
        for i in range(2, len(data) - 2):
            high = data[i][1]
            low = data[i][2]
            if high > data[i-1][1] and high > data[i+1][1]:
                levels.add(round(high, 4))
            if low < data[i-1][2] and low < data[i+1][2]:
                levels.add(round(low, 4))
        return levels

    levels_4h = levels_from(candles_4h)
    levels_1h = levels_from(candles_1h)
    levels_15m = levels_from(candles_15m)

    def near_match(levels_a, levels_b):
        result = set()
        for a in levels_a:
            for b in levels_b:
                if abs(a - b) / b < 0.0030:  # ±0.1%
                    result.add(round((a + b) / 2, 4))
        return result

    strict_match = levels_4h & levels_1h & levels_15m
    soft_match = near_match(levels_4h, levels_1h) & near_match(levels_1h, levels_15m)

    return list(strict_match | soft_match)

def confirm_entry_15m(symbol, level, side):
    candles_15m = fetch_cached_ohlcv(symbol, "15")
    if not candles_15m:
        return False
    last_close = candles_15m[-1][3]
    near = abs(last_close - level) / max(level, 1e-9) < 0.0020  # ±0.20% допуска (было жёстче)
    if side == "long":
        return (last_close > level) or near
    elif side == "short":
        return (last_close < level) or near
    return False

def get_volume_indicator(symbol):
    try:
        candles = fetch_cached_ohlcv(symbol, "1", limit=100)
        if not candles or len(candles) < 2:
            return "⚪ нет данных"
        try:
            if len(candles[-1]) < 5:
                return "⚪ недостаточно данных"
            last_vol = float(candles[-1][4])
            volumes = []
            for c in candles[:-1]:
                try:
                    volumes.append(float(c[4]))
                except (IndexError, ValueError):
                    continue
        except Exception as e:
            print(f"⚠️ Ошибка обработки объемов: {e}")
            return "⚪ ошибка объема"
        if len(volumes) == 0:
            return "⚪ нет среднего объема"
        avg_vol = sum(volumes) / len(volumes)
        if last_vol >= avg_vol * 1.5:
            return f"🟢 Объём: {last_vol:.0f} (высокий)"
        elif last_vol >= avg_vol * 0.7:
            return f"🟡 Объём: {last_vol:.0f} (средний)"
        else:
            return f"🔴 Объём: {last_vol:.0f} (низкий)"
    except Exception as e:
        print(f"⚠️ Ошибка получения объема для {symbol}: {e}")
        return "⚪ ошибка объема"



def get_market_data():
    url = "https://api.bybit.com/v5/market/tickers?category=linear"
    response = HTTP.get(url)
    data = response.json()["result"]["list"]
    result = {}
    for item in data:
        symbol = item["symbol"]
        volume = float(item.get("turnover24h", 0))
        if volume >= 10_000_000:
            result[symbol] = {
                "price": float(item["lastPrice"]),
                "volume": volume
            }
    result = dict(sorted(result.items(), key=lambda x: -x[1]['volume']))
    return dict(list(result.items())[:80])

ohlcv_cache = {}
def fetch_cached_ohlcv(symbol, interval, limit=100):
    key = (symbol, interval)
    if key in ohlcv_cache:
        return ohlcv_cache[key]
    data = fetch_ohlcv(symbol, interval, limit)
    ohlcv_cache[key] = data
    return data

def _mk_features_for_bot(symbol: str, side: str) -> np.ndarray:
    """
    Стараемся построить признаки так, чтобы совпасть с feature_order.json.
    Используем твои кэши свечей: 1m/15m/60m. Не валимся, если чего-то нет.
    """
    import numpy as _np
    import pandas as _pd

    # Забираем куски из кеша (можешь увеличить лимиты при желании)
    m1  = fetch_cached_ohlcv(symbol, "1",  limit=240) or []
    m15 = fetch_cached_ohlcv(symbol, "15", limit=240) or []
    h1  = fetch_cached_ohlcv(symbol, "60", limit=240) or []

    # Быстрые агрегаты
    def _last_close(arr): 
        try: return float(arr[-1][3])
        except Exception: return _np.nan
    def _atr_like(arr, n=14):
        try:
            df = pd.DataFrame(arr, columns=["open","high","low","close","volume"])
            high = df["high"].astype(float); low = df["low"].astype(float); close = df["close"].astype(float)
            prev_close = close.shift(1)
            tr = (high - low).abs().combine((high - prev_close).abs(), max).combine((low - prev_close).abs(), max)
            return float(tr.rolling(window=n, min_periods=n).mean().iloc[-1])
        except Exception:
            return _np.nan
    def _sma(arr, n=20):
        try:
            import pandas as _pd
            df = pd.DataFrame(arr, columns=["open","high","low","close","volume"])
            return float(df["close"].rolling(window=n, min_periods=n).mean().iloc[-1])
        except Exception:
            return _np.nan

    # Простейшие кандидаты (под многие пайплайны)
    feats = {}
    c1 = _last_close(m1); c15 = _last_close(m15); c60 = _last_close(h1)
    feats["c1"]  = c1
    feats["c15"] = c15
    feats["c60"] = c60
    feats["atr1"]  = _atr_like(m1, 14)
    feats["atr15"] = _atr_like(m15,14)
    feats["atr60"] = _atr_like(h1, 14)
    feats["sma20_15"] = _sma(m15, 20)
    feats["sma200_60"] = _sma(h1, 200)
    try:
        feats["ema20_60"]  = float(pd.Series([x[3] for x in h1]).ewm(span=20, adjust=False).mean().iloc[-1])
        feats["ema200_60"] = float(pd.Series([x[3] for x in h1]).ewm(span=200, adjust=False).mean().iloc[-1])
    except Exception:
        feats["ema20_60"]  = np.nan
        feats["ema200_60"] = np.nan

    # Сторона в виде битов (часто в пайплайнах)
    feats["is_long"]  = 1 if side == "long" else 0
    feats["is_short"] = 1 if side == "short" else 0

    # Сводим к нужному порядку
    row = []
    for name in FEATURES:
        val = feats.get(name, 0.0)
        try:
            row.append(float(val))
        except Exception:
            row.append(0.0)
    X_df = pd.DataFrame([row], columns=FEATURES)
    return _impute_safe(X_df)

# ----------------------------- ПОДПИСЧИКИ ------------------------------
chat_ids_file = "chat_ids.json"
if os.path.exists(chat_ids_file):
    try:
        with open(chat_ids_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except Exception:
        loaded = []
    # в памяти держим set, в файл пишем список
    chat_ids = set(loaded if isinstance(loaded, list) else [])
else:
    chat_ids = set()

@bot.message_handler(func=lambda message: True)
def handle_new_user(message):
    user_id = message.chat.id
    if user_id not in chat_ids:
        chat_ids.add(user_id)
        with open(chat_ids_file, "w", encoding="utf-8") as f:
            json.dump(sorted(list(chat_ids)), f, ensure_ascii=False)
        bot.send_message(user_id, "✅ Подписка на сигналы активирована!")
    else:
        bot.send_message(user_id, "🔔 Уже подписаны.")

def _notify_http_fallback(uid: int, text: str):
    try:
        import requests
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": uid, "text": text})
    except Exception as e:
        print(f"❌ HTTP fallback провалился для {uid}:", e)

def notify(text):
    for uid in list(chat_ids):
        sent = False
        for _ in range(3):  # up to 3 retries
            try:
                bot.send_message(uid, text)
                sent = True
                break
            except Exception as e:
                print(f"⚠️ Ошибка отправки {uid}, попытка повторить: {e}")
                time.sleep(1.2)
        if not sent:
            _notify_http_fallback(uid, text)


# ---------------------------- ПРОГНОЗ И ЦЕНЫ ---------------------------
def load_forecast():
    url = "https://raw.githubusercontent.com/toevich/crypto-signals/main/forecast.json"
    try:
        resp = HTTP.get(url, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
        last_updated = raw.get("last_updated", "не указано")
        forecast = {}
        for sym, data in raw.items():
            if sym == "last_updated":
                continue
            forecast[sym] = {"long": [], "short": []}
            for side in ["long", "short"]:
                for zone in data.get(side, []):
                    if "zone" in zone:
                        forecast[sym][side].append({
                            "zone": tuple(zone["zone"]),
                            "tp": float(zone["tp"]),
                            "sl": float(zone["sl"]),
                            "leverage": int(zone["leverage"]),
                            "po": zone.get("po", "N/A"),
                            "ATR":float(zone["ATR"]),
                            "ATR_rem":float(zone["ATR_rem"]),
                            "price":float(zone["price"]),
                        })
        return forecast, last_updated
    except Exception as e:
        print("❌ Ошибка загрузки прогноза:", e)
        return {}, "не указано"

def get_prices():
    url = "https://api.bybit.com/v5/market/tickers?category=linear"
    try:
        r = HTTP.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()["result"]["list"]
        return {x["symbol"]: float(x["lastPrice"]) for x in data}
    except Exception as e:
        print(f"⚠️ Ошибка при получении цен: {e}")
        return {}

# -------------------------- МОНИТОРИНГ ЦЕН ----------------------------
from concurrent.futures import ThreadPoolExecutor, as_completed


def is_consolidating(symbol, level, atr):
    try:
        df_1h = fetch_cached_ohlcv(symbol, "60", limit=10)
        if not df_1h or len(df_1h) < 6:
            return False
        df = pd.DataFrame(df_1h, columns=["open", "high", "low", "close", "volume"])

        near_level = df.apply(lambda row: abs((row["high"] + row["low"]) / 2 - level) < atr * 0.20, axis=1)
        narrow_range = (df["high"] - df["low"]) < atr * 0.30

        return (near_level & narrow_range).sum() >= 7
    except Exception as e:
        print(f"[DEBUG] ⚠️ Ошибка анализа консолидации {symbol}: {e}")
        return False
        df = pd.DataFrame(df_1h, columns=["open", "high", "low", "close", "volume"])

        near_level = df.apply(lambda row: abs((row["high"] + row["low"]) / 2 - level) < atr * 0.20, axis=1)
        narrow_range = (df["high"] - df["low"]) < atr * 0.30

        return (near_level & narrow_range).sum() >= 7
    except Exception as e:
        print(f"[DEBUG] ⚠️ Ошибка анализа консолидации {symbol}: {e}")
        return False
        df = pd.DataFrame(df_1h, columns=["open", "high", "low", "close", "volume"])
        touches = 0
        for _, row in df.iterrows():
            mid = (row['high'] + row['low']) / 2
            if abs(mid - level) <= atr * 0.2 and (row['high'] - row['low']) < atr * 0.3:
                touches += 1
        return touches >= 3
    except Exception as e:
        print(f"[DEBUG] ⚠️ Ошибка анализа консолидации {symbol}: {e}")
        return False


def detect_po_type(df, level, side):
    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        body = abs(last["close"] - last["open"])
        range_ = last["high"] - last["low"]
        body_ratio = body / (range_ + 1e-9)
        avg_vol = df["volume"].iloc[-10:].mean()
        vol = last["volume"]

        if side == "long":
            if last["close"] > level and prev["close"] > level and vol > avg_vol:
                return "Пробой"
            elif last["low"] < level and last["close"] > last["open"]:
                return "Отбой"
        if side == "short":
            if last["close"] < level and prev["close"] < level and vol > avg_vol:
                return "Пробой"
            elif last["high"] > level and last["close"] < last["open"]:
                return "Отбой"
        return "Отбой"
    except:
        return "Отбой"
def strong_confirmation_15m(symbol, side):
    candles = fetch_cached_ohlcv(symbol, "15", limit=2)
    if not candles or len(candles) < 2:
        return False
    o, h, l, c, v = candles[-1]
    body = abs(c - o)
    range_ = h - l
    if range_ == 0:
        return False
    body_ratio = body / range_
    if side == "long" and c > o and body_ratio > 0.40:
        return True
    if side == "short" and c < o and body_ratio > 0.40:
        return True
    return False

recent_sl_hits = {}

def was_recent_sl_hit(symbol, level, hours=2):
    from datetime import datetime, timedelta
    now = datetime.now()
    records = recent_sl_hits.get(symbol, [])
    return any(abs(level - lvl) / lvl < 0.002 and now - ts < timedelta(hours=hours) for lvl, ts in records)

def record_sl_hit(symbol, level):
    from datetime import datetime
    if symbol not in recent_sl_hits:
        recent_sl_hits[symbol] = []
    recent_sl_hits[symbol].append((level, datetime.now()))

# --- PATCH 1/5: helper для ATR 24h ---
def compute_atr24h_fields(symbol: str, current_price: float):
    """
    Возвращает (ATR24, ATR24_rem, ATR24_rem_pct).
    ATR24 — диапазон за ~24 часа по 15m: (max(high)-min(low))*1.05
    ATR24_rem — остаток: ATR24 - |price_now - price_24h_ago|
    ATR24_rem_pct — доля остатка от ATR24 (0..1)
    """
    try:
        # 24ч ≈ 96 баров по 15m; берём 97 чтобы был старт и конец окна
        m15 = fetch_cached_ohlcv(symbol, "15", limit=97) or []
        if not m15 or len(m15) < 97:
            m15 = fetch_ohlcv(symbol, interval="15", limit=97) or []
        if m15 and len(m15) >= 97:
            window = m15[-97:]
            hi = max(float(b[1]) for b in window)
            lo = min(float(b[2]) for b in window)
            atr5 = (hi - lo) * 1.05
            p_start = float(window[0][3])  # close ~24ч назад
            p_now = float(current_price)
            dist = abs(p_now - p_start)
            rem = max(0.0, atr5 - dist)
            rem_pct = rem / max(1e-9, atr5)
            return round(atr5, 6), round(rem, 6), rem_pct
    except Exception:
        pass
    return 0.0, 0.0, 0.0
# --- /PATCH 1/5 ---

def find_next_level(levels, ref_price: float, side: str):
    """
    Возвращает ближайший УРОВЕНЬ дальше по направлению сделки.
    side: 'long' -> ближайшее сопротивление выше, 'short' -> поддержка ниже.
    """
    if not levels:
        return None
    lvls = sorted(set(float(x) for x in levels))
    if side == "long":
        for lv in lvls:
            if lv > ref_price + 1e-9:
                return lv
    else:
        for lv in reversed(lvls):
            if lv < ref_price - 1e-9:
                return lv
    return None
# --- PATCH 1/4: ATR(5D) helper ---
def compute_atr5d_fields(symbol: str, current_price: float):
    """
    Возвращает (ATR5, ATR5_rem, ATR5_rem_pct) на дневном ТФ.
    ATR5 — средний True Range 5 ПОСЛЕДНИХ ЗАВЕРШЁННЫХ дней.
    ATR5_rem — «остаток» ATR5 на сегодня = ATR5 - |price_now - day_open_today|.
    ATR5_rem_pct — доля остатка от ATR5 (0..1).
    """
    try:
        d1 = fetch_cached_ohlcv(symbol, "D", limit=8) or []  # нужно >=6-7 баров
    except Exception:
        d1 = []
    if not d1 or len(d1) < 6:
        return 0.0, 0.0, 0.0

    # разделим: завершённые дни и текущий
    completed = d1[:-1]            # все кроме последнего (текущего)
    today = d1[-1]                 # [o,h,l,c,v] текущего дня (может быть незавершён)
    if len(completed) < 5:
        return 0.0, 0.0, 0.0

    # считаем TR классически: TR = max(h-l, |h - prev_close|, |l - prev_close|)
    trs = []
    for k in range(-5, 0):  # последние 5 завершённых дней
        o,h,l,c,*_ = completed[k]
        prev_c = completed[k-1][3] if k-1 >= -len(completed) else c
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)
    atr5 = sum(trs) / 5.0

    # «потрачено» сегодня — от открытия дня до текущей цены
    day_open = float(today[0])
    p_now = float(current_price)
    spent = abs(p_now - day_open)

    rem = max(0.0, atr5 - spent)
    rem_pct = rem / max(1e-9, atr5)
    return round(atr5, 6), round(rem, 6), rem_pct
# --- /PATCH 1/4 ---


def analyze_symbol(symbol, info):
    result = {}
    price = info["price"]
    # NN extra features defaults (always defined)
    ema20_dist_pct = None
    ema200_dist_pct = None
    atr15_pct = None
    try:
        df = fetch_cached_ohlcv(symbol, "60", limit=150)
        if not df:
            return symbol, None
        df = pd.DataFrame(df, columns=["open", "high", "low", "close", "volume"])
        df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        macd = ta.trend.MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        rsi = round(df["rsi"].iloc[-1], 2)
        macd_val = round(df["macd"].iloc[-1], 6)
        macd_signal_val = round(df["macd_signal"].iloc[-1], 6)
    except Exception as e:
        print(f"⚠️ Ошибка анализа {symbol}: {e}")
        return symbol, None

    levels = detect_common_levels(symbol)
    if not levels:
        return symbol, None

    zones_long = []
    zones_short = []

    for level in levels:
        if was_recent_sl_hit(symbol, level):
            continue
        # --- PATCH 2/5: расчёт ATR(1H) и ATR(24h) ---
        # ATR(1H, 14) — для механики зон/толерансов (как было)
        atr = round((max(df["high"][-14:]) - min(df["low"][-14:])) * 1.05, 6)
        atr_rem_h1 = round(atr - (df["high"].iloc[-1] - df["low"].iloc[-1]), 6)

        # ATR(5D) — для «остатка топлива»
        atr5, atr5_rem, atr5_rem_pct = compute_atr5d_fields(symbol, price)
        # --- /PATCH 2/5 ---


        if level > price and confirm_entry_15m(symbol, level, "long") and strong_confirmation_15m(symbol, "long"):
            dist = abs(price - level) / price
            if dist > 0.005:   # больше 0.5% — слишком далеко от уровня
                continue
            near_level = abs(price - level) / price < 0.012
            # адаптивная ширина зоны для LONG
            min_w, max_w = (0.0020, 0.0030) if is_consolidating(symbol, level, atr) and near_level else (0.0035, 0.0055)
            zone = [round(level * (1 - max_w), 6), round(level * (1 - min_w), 6)]
            # заранее определим тип сигнала для расчёта k_box
            signal_type = auto_detect_type(df, {"ATR": atr, "zone": zone}, "long")
            # === SMART TP for LONG ===
            entry_mid = (zone[0] + zone[1]) / 2.0

            # 1) Структурная цель: следующий 1H/4H уровень выше
            buf = max(0.15 * atr, 0.002 * level)  # фронт-ран до уровня
            next_lv = find_next_level(levels, level, "long")
            tp_struct = (next_lv - buf) if next_lv else None

            # 2) Мерный ход из бокса (последние 24 бара 1H)
            box_h = float(max(df["high"][-24:]) - min(df["low"][-24:]))
            k_box = 0.6 if (isinstance(signal_type, str) and "Ложный пробой" in signal_type) else 1.0
            tp_mm = entry_mid + k_box * box_h

            # 3) Волатильностная цель от ATR(1H)
            tp_atr = entry_mid + 1.2 * atr

            # 4) Консервативный выбор цели
            tp_candidates = [x for x in (tp_struct, tp_mm, tp_atr) if x is not None]
            tp = min(tp_candidates) if tp_candidates else (entry_mid + 1.2 * atr)

            # SL чуть шире, чтобы не выбивало шумом
            sl = round(zone[0] - 0.8 * atr, 6)

            # 5) RR-клиппинг в коридор [1.5 … 3.0]
            min_rr, max_rr = 1.5, 3.0
            rr = (tp - entry_mid) / max(1e-9, entry_mid - sl)
            if rr < min_rr:
                tp = entry_mid + min_rr * (entry_mid - sl)
            elif rr > max_rr:
                tp = entry_mid + max_rr * (entry_mid - sl)

            tp = round(tp, 6)
            rr = (tp - entry_mid) / max(1e-9, entry_mid - sl)

            # 6) Частичный выход: TP1 на половине пути
            tp1 = round(entry_mid + 0.5 * (tp - entry_mid), 6)
            leverage = int(calc_leverage_atr(symbol, price, atr, rr))
            zones_long.append({
                "patterns": detect_candle_pattern(df),
                "level_4h": level,
                "level_1h": level,
                "level_15m": level,
                "zone": zone, "sl": sl,
                "tp1": tp1,
                "tp2": tp,   # основной
                "rr": rr,
                "leverage": leverage,
                "po": "Пробой" if price > level else "Отбой",
                "ATR": atr, "ATR_rem": atr_rem_h1,
                "ATR5": atr5, "ATR5_rem": atr5_rem, "ATR5_rem_pct": atr5_rem_pct,
                "price": price,
                "consolidation": is_consolidating(symbol, level, atr),
                "type": signal_type,
                # extra NN features (safe)
                "ema20_dist_pct": ema20_dist_pct,
                "ema200_dist_pct": ema200_dist_pct,
                "atr15_pct": atr15_pct,
            })

        if level < price and confirm_entry_15m(symbol, level, "short") and strong_confirmation_15m(symbol, "short"):
            near_level = abs(price - level) / price < 0.01
            # адаптивная ширина зоны для SHORT
            min_w, max_w = (0.0020, 0.0030) if is_consolidating(symbol, level, atr) and near_level else (0.0035, 0.0055)
            zone = [round(level * (1 + min_w), 6), round(level * (1 + max_w), 6)]
            # заранее определим тип сигнала для расчёта k_box
            signal_type = auto_detect_type(df, {"ATR": atr, "zone": zone}, "short")
            # === SMART TP for SHORT ===
            entry_mid = (zone[0] + zone[1]) / 2.0

            # 1) Структурная цель: ближайшая поддержка ниже
            buf = max(0.15 * atr, 0.002 * level)
            next_lv = find_next_level(levels, level, "short")
            tp_struct = (next_lv + buf) if next_lv else None

            # 2) Мерный ход из бокса
            box_h = float(max(df["high"][-24:]) - min(df["low"][-24:]))
            k_box = 0.6 if (isinstance(signal_type, str) and "Ложный пробой" in signal_type) else 1.0
            tp_mm = entry_mid - k_box * box_h

            # 3) Волатильностная цель
            tp_atr = entry_mid - 1.2 * atr

            # 4) Консервативный выбор цели
            tp_candidates = [x for x in (tp_struct, tp_mm, tp_atr) if x is not None]
            tp = max(tp_candidates) if tp_candidates else (entry_mid - 1.2 * atr)

            # SL чуть шире
            sl = round(zone[1] + 0.8 * atr, 6)

            # 5) RR-клиппинг [1.5 … 3.0]
            min_rr, max_rr = 1.5, 3.0
            rr = (entry_mid - tp) / max(1e-9, sl - entry_mid)
            if rr < min_rr:
                tp = entry_mid - min_rr * (sl - entry_mid)
            elif rr > max_rr:
                tp = entry_mid - max_rr * (sl - entry_mid)

            tp = round(tp, 6)
            rr = (entry_mid - tp) / max(1e-9, sl - entry_mid)

            # 6) Частичный выход: TP1 на половине
            tp1 = round(entry_mid - 0.5 * (entry_mid - tp), 6)

            leverage = int(calc_leverage_atr(symbol, price, atr, rr))
            zones_short.append({
                "patterns": detect_candle_pattern(df),
                "level_4h": level,
                "level_1h": level,
                "level_15m": level,
                "zone": zone, "sl": sl, 
                "tp1": tp1,
                "tp2": tp,   # основной
                "rr": rr,
                "leverage": leverage,
                "po": "Пробой" if price < level else "Отбой",
                "ATR": atr, "ATR_rem": atr_rem_h1,
                "ATR5": atr5, "ATR5_rem": atr5_rem, "ATR5_rem_pct": atr5_rem_pct,
                "price": price,
                "consolidation": is_consolidating(symbol, level, atr),
                "type": signal_type,
                # extra NN features (safe)
                "ema20_dist_pct": ema20_dist_pct,
                "ema200_dist_pct": ema200_dist_pct,
                "atr15_pct": atr15_pct,
            })

    if zones_long or zones_short:
        result = {"long": zones_long, "short": zones_short}
        return symbol, result
    return symbol, None

def build_live_forecast(data):
    result = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(analyze_symbol, symbol, info) for symbol, info in data.items()]
        for future in as_completed(futures):
            symbol, forecast = future.result()
            if forecast:
                result[symbol] = forecast
    return result


def monitor():
    global forecast
    market_data = get_market_data()
    forecast = build_live_forecast(market_data)
    forecast_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    from collections import defaultdict
    alerted = {"long": set(), "short": set()}
    last_reload = time.time()
    last_sent_ts = {}          # key -> timestamp последней отправки
    last_in_counter = {}       # key -> счётчик подряд-попаданий в зону (для анти-дребезга)
    last_exit_ts = {}          # key -> когда реально «вышли» из расширенной зоны

    REENTRY_COOLDOWN_SEC = 300 * 60   # 300 минут между повторными сигналами по той же зоне
    HOLD_TICKS_IN_ZONE   = 2         # минимум 2 последовательных тика в зоне для подтверждения
    HYSTERESIS_FACTOR    = 1.0       # выйти надо на 50% ширины зоны за её пределы


    while True:
        now_time = time.time()
        if now_time - last_reload > 45:
            market_data = get_market_data()
            forecast = build_live_forecast(market_data)
            forecast_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            last_reload = now_time

        if not forecast:
            print("⚠️ Прогноз пуст — повторная попытка через 15 секунд")
            time.sleep(15)
            market_data = get_market_data()
            forecast = build_live_forecast(market_data)
            forecast_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            last_reload = time.time()
            continue

        prices = get_prices()
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        print("[DEBUG] Текущий прогноз:", forecast)
        for symbol, data in forecast.items():
            price = prices.get(symbol)
            if not price:
                print(f"⛔ Нет цены для {symbol}")
                continue
            for side in ["long", "short"]:
                zones = data.get(side, [])
                if not zones:
                    print(f"{now} ⛔ {symbol} {side.upper()} — зон нет")
                for zone in zones:
                    zone_range = zone["zone"]
                    z_low, z_high = float(zone_range[0]), float(zone_range[1])
                    width = abs(z_high - z_low)
                    # FAST ENTRY: сильный пробой уровня — даём быстрый сигнал малыми допусками
                    lvl_mid = (z_low + z_high) / 2.0
                    if is_strong_breakout(symbol, side, lvl_mid):
                        print(f"[FAST] 🚀 Сильный пробой {symbol} {side.upper()} у уровня {lvl_mid}")
                        send_signal(symbol, price, zone, side, forecast_time)
                        # не дожидаемся hold ticks — это отдельный тип входа
                        continue
                    
                    in_zone   = (z_low <= price <= z_high)
                    ext_low   = z_low  - HYSTERESIS_FACTOR * width
                    ext_high  = z_high + HYSTERESIS_FACTOR * width
                    outside_ext = (price < ext_low) or (price > ext_high)
                    key = f"{symbol}_{side}_{z_low}_{z_high}"
                    was_alerted = key in alerted[side]
                    now_ts = time.time()
                    if in_zone:
                        last_in_counter[key] = last_in_counter.get(key, 0) + 1
                    else:
                        last_in_counter[key] = 0
                    if was_alerted and outside_ext:
                        alerted[side].remove(key)
                        last_exit_ts[key] = now_ts
                        print(f"🔁 ВЫХОД (re-arm): {symbol} {side.upper()} — {zone_range}, price={price}")
                    last_sent = last_sent_ts.get(key, 0.0)
                    cooldown_ok = (now_ts - last_sent) >= REENTRY_COOLDOWN_SEC
                    print(f"{now} ▶️ {symbol} {side.upper()} :: price={price} zone={zone_range} in_zone={in_zone} alerted={was_alerted} hold={last_in_counter.get(key,0)} cd_ok={cooldown_ok}")
                    if in_zone and not was_alerted and last_in_counter.get(key,0) >= HOLD_TICKS_IN_ZONE and cooldown_ok:
                        print(f"[DEBUG] ✅ SEND {symbol} {side.upper()} — устойчивый вход в зону {zone_range}")
                        send_signal(symbol, price, zone, side, forecast_time)
                        alerted[side].add(key)
                        last_sent_ts[key] = now_ts

                    z_low, z_high = float(zone_range[0]), float(zone_range[1])
                    width = abs(z_high - z_low)

                  # базовые статусы
                    in_zone   = (z_low <= price <= z_high)
                    # «расширенная зона» для гистерезиса (нужно дальше выйти, чтобы «разармировать»)
                    ext_low   = z_low  - HYSTERESIS_FACTOR * width
                    ext_high  = z_high + HYSTERESIS_FACTOR * width
                    outside_ext = (price < ext_low) or (price > ext_high)

                    key = f"{symbol}_{side}_{z_low}_{z_high}"
                    was_alerted = key in alerted[side]
                    now_ts = time.time()

                # счётчик последовательных попаданий внутрь «узкой» зоны
                    if in_zone:
                        last_in_counter[key] = last_in_counter.get(key, 0) + 1
                    else:
                        last_in_counter[key] = 0

                # перезарядка (разармирование) только после выхода из расширенной зоны
                    if was_alerted and outside_ext:
                        alerted[side].remove(key)
                        last_exit_ts[key] = now_ts
                        print(f"🔁 ВЫХОД (re-arm): {symbol} {side.upper()} — {zone_range}, price={price}")

                # проверка кулдауна повторного входа
                    last_sent = last_sent_ts.get(key, 0.0)
                    cooldown_ok = (now_ts - last_sent) >= REENTRY_COOLDOWN_SEC

                    print(f"{now} ▶️ {symbol} {side.upper()} :: price={price} zone={zone_range} in_zone={in_zone} alerted={was_alerted} hold={last_in_counter.get(key,0)} cd_ok={cooldown_ok}")

                # условие отправки: внутри зоны достаточное время, не заспамлено, и зона «заряжена»
                    if in_zone and not was_alerted and last_in_counter.get(key,0) >= HOLD_TICKS_IN_ZONE and cooldown_ok:
                        print(f"[DEBUG] ✅ SEND {symbol} {side.upper()} — устойчивый вход в зону {zone_range}")
                        send_signal(symbol, price, zone, side, forecast_time)
                        alerted[side].add(key)
                        last_sent_ts[key] = now_ts

def detect_candle_pattern(df):
    try:
        patterns = []
        last = df.iloc[-1]
        prev = df.iloc[-2]
        body = abs(last["close"] - last["open"])
        range_ = last["high"] - last["low"]
        upper_shadow = last["high"] - max(last["close"], last["open"])
        lower_shadow = min(last["close"], last["open"]) - last["low"]

        # Пин-бар
        if body < range_ * 0.3:
            if upper_shadow > body * 2:
                patterns.append("Пин-бар (медвежий)")
            if lower_shadow > body * 2:
                patterns.append("Пин-бар (бычий)")

        # Доджи
        if body / (range_ + 1e-9) < 0.1:
            patterns.append("Доджи")

        # Поглощение
        if last["close"] > last["open"] and prev["close"] < prev["open"]:
            if last["open"] < prev["close"] and last["close"] > prev["open"]:
                patterns.append("Бычье поглощение")
        if last["close"] < last["open"] and prev["close"] > prev["open"]:
            if last["open"] > prev["close"] and last["close"] < prev["open"]:
                patterns.append("Медвежье поглощение")

        return patterns
    except Exception as e:
        print(f"[DEBUG] Ошибка в detect_candle_pattern: {e}")
        return []

def auto_detect_type(df, zone, side):
    result = []
    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        tail_ratio = abs(last["close"] - last["open"]) / (last["high"] - last["low"] + 1e-9)

        atr = zone["ATR"] if isinstance(zone, dict) else 0.0
        rsi = last["rsi"] if "rsi" in df.columns else 50
        macd = last["macd"] if "macd" in df.columns else 0
        macd_signal = last["macd_signal"] if "macd_signal" in df.columns else 0
        vol = last["volume"]
        avg_vol = df["volume"].iloc[-20:].mean()

        if tail_ratio < 0.2:
            result.append("Разворотная свеча")
        if rsi < 30 and side == "long":
            result.append("Отскок после импульса")
        if rsi > 70 and side == "short":
            result.append("Отскок после импульса")
        if (macd > macd_signal) and rsi > 55 and side == "long":
            result.append("Импульсный пробой")
        if (macd < macd_signal) and rsi < 45 and side == "short":
            result.append("Импульсный пробой")
        if vol > avg_vol * 1.5:
            result.append("Нарастающий объём")
        if atr and atr < df["close"].mean() * 0.005:
            result.append("Затухание")
        if rsi > 50 and df["close"].iloc[-1] < df["close"].iloc[-2] and side == "short":
            result.append("Ложный пробой")
        if rsi < 50 and df["close"].iloc[-1] > df["close"].iloc[-2] and side == "long":
            result.append("Ложный пробой")

        return result[0] if result else "Стандарт"
    except Exception as e:
        print(f"[DEBUG] Ошибка в auto_detect_type: {e}")
        return "Стандарт"

# ---------------------------- СИГНАЛЫ И ЛОГ ----------------------------
log_file = "signals_log.xlsx"
def log_signal(symbol, direction, price, zone, forecast_time):
    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "direction": direction.upper(),
        "price": price,
        "tp": zone["tp"],
        "sl": zone["sl"],
        "leverage": zone["leverage"],
        "po": zone.get("po", "N/A"),
        "forecast_time": forecast_time,
        "zone_low": zone["zone"][0],
        "zone_high": zone["zone"][1],
        "price_forecast": zone.get("price", "N/A"),
        "level_4h": zone.get("level_4h", "N/A"),
        "level_1h": zone.get("level_1h", "N/A"),
        "level_15m": zone.get("level_15m", "N/A"),
        "type": zone.get("type", "N/A"),
        "consolidation": zone.get("consolidation", "N/A"),
        "patterns": ", ".join(zone.get("patterns", [])) if zone.get("patterns") else "",
        "pct_diff_log": json.dumps(zone.get("pct_diff_log", []), ensure_ascii=False),
        "consolidation_lvl_used": zone.get("consolidation_lvl_used", "N/A"),
        "consolidation_debug": json.dumps(zone.get("consolidation_debug", {}), ensure_ascii=False),
        "probability_pct": zone.get("probability_pct", "N/A")
    , "entry_reasons_ru": zone.get("entry_reasons_ru","")
    }
    if os.path.exists(log_file):
        df = pd.read_excel(log_file)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_excel(log_file, index=False)
    upload_to_github(log_file, "signals_log.xlsx", f"Добавлен сигнал {symbol} ({direction.upper()})")

def calculate_atr(symbol, interval="15", period=14):
    candles = fetch_ohlcv(symbol, interval=interval, limit=period + 1)
    if len(candles) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        high = candles[i][1]
        low = candles[i][2]
        prev_close = candles[i - 1][3]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    return round(sum(trs) / len(trs), 6)


# ------------------------- ОЦЕНКА ВЕРОЯТНОСТИ --------------------------
def _volume_bucket(note: str) -> str:
    if "🟢" in note: return "high"
    if "🟡" in note: return "mid"
    if "🔴" in note: return "low"
    return "na"

def normalize_zone_kind(zone: dict, price: float) -> str:
    lvl = float(zone.get('lvl') or zone.get('level') or zone.get('lvl_price') or 0.0)
    tol = float(zone.get('tol', 0.0))
    kind = (zone.get('type') or zone.get('kind') or '').lower()  # 'support'|'resistance'|...

    # Если реальное положение цены противоречит типу – подправим.
    # Цена заметно ниже уровня -> это сопротивление; выше -> поддержка.
    if price < (lvl - tol) and 'support' in kind:
        kind = 'resistance'
    elif price > (lvl + tol) and 'resistance' in kind:
        kind = 'support'

    zone['type'] = kind
    zone['kind'] = kind
    zone['lvl'] = lvl
    zone['tol'] = tol
    return kind

def adjust_entry_for_zone(side: str, zone: dict, entry_low: float, entry_high: float):
    lvl = float(zone.get('lvl') or zone.get('level') or 0.0)
    tol = float(zone.get('tol', 0.0))
    atr = float(zone.get('atr', 0.0))
    kind = (zone.get('type') or zone.get('kind') or '').lower()
    # буфер “на сторону уровня”: берём погрубее, чтобы не оставаться на неверной стороне
    buf = max(tol, 0.10 * atr, lvl * 0.0007)

    lo, hi = float(entry_low), float(entry_high)

    if side == 'long':
        if 'support' in kind:
            # Отбой LONG от поддержки: вход должен быть НЕ НИЖЕ уровня (лучше чуть выше)
            lo = max(lo, lvl)              # минимум — на уровне
            hi = max(hi, lvl + 0.25 * buf) # верх входа немного выше уровня
        else:
            # Пробой LONG через сопротивление: только ВЫШЕ уровня+буфер
            lo = max(lo, lvl + buf)
            hi = max(hi, lo)
    else:  # short
        if 'resistance' in kind:
            # Отбой SHORT от сопротивления: вход должен быть НЕ ВЫШЕ уровня
            hi = min(hi, lvl)              # максимум — на уровне
            lo = min(lo, lvl - 0.25 * buf) # низ входа немного ниже уровня
        else:
            # Пробой SHORT через поддержку: только НИЖЕ уровня-буфер
            hi = min(hi, lvl - buf)
            lo = min(lo, hi)

    return lo, hi

def _trendgate_override(symbol: str, side: str, zone: dict) -> bool:
    """Allow sending even if trend_gate failed, when quality is decent."""
    try:
        rr = 2.0
        if 'tp' in zone and 'zone' in zone:
            price_mid = sum(zone['zone'])/2.0
            rr = abs(zone['tp'] - price_mid) / max(1e-9, abs(price_mid - zone.get('sl', price_mid)))
        prob = zone.get('probability_pct')
        if prob is None:
            prob = 60  # assume decent if not computed yet
        # If probability >= 58 or RR >= 2.2 — let it pass once.
        return (prob >= 58) or (rr >= 2.2)
    except Exception:
        return False

def compute_probability(symbol: str, side: str, zone: dict, rr_ratio: float) -> int:
    """
    Эвристическая оценка вероятности (0–100):
    - тип сигнала/паттерны, консолидация у уровня
    - сила объёма за 1м
    - подтверждение 15m-свечой
    - соотношение TP/SL (RR)
    """
    score = 50
    st = zone.get("type", "Стандарт")
    st = ", ".join(st) if isinstance(st, list) else st
    if "Импульсный пробой" in st: score += 12
    if "Нарастающий объём" in st: score += 8
    if "Поджатие" in st: score += 6
    if "Отскок после импульса" in st: score += 5
    if "Разворотная свеча" in st: score -= 10
    if zone.get("consolidation"): score += 6
    vol_note = get_volume_indicator(symbol)  # уже есть в проекте
    vb = _volume_bucket(vol_note)
    if vb == "high": score += 8
    elif vb == "mid": score += 3
    elif vb == "low": score -= 8
    try:
        if strong_confirmation_15m(symbol, side):  # уже есть в проекте
            score += 7
    except Exception:
        pass
    if rr_ratio >= 2.5: score += 8
    elif rr_ratio >= 2.0: score += 5
    elif rr_ratio < 1.5: score -= 6
    score = max(20, min(90, score))
    return int(round(score))



def calculate_atr_1h(symbol, period=14):
    """ATR по 1H свечам для символа."""
    try:
        df = fetch_ohlcv(symbol, interval="60", limit=period + 100)
        if not df or len(df) < period + 1:
            return 0.0
        import pandas as _pd
        _df = _pd.DataFrame(df, columns=["open","high","low","close","volume"])
        high = _df["high"].astype(float); low = _df["low"].astype(float); close = _df["close"].astype(float)
        prev_close = close.shift(1)
        tr = (high - low).abs().combine((high - prev_close).abs(), max).combine((low - prev_close).abs(), max)
        atr = tr.rolling(window=period, min_periods=period).mean()
        return float(atr.iloc[-1])
    except Exception as e:
        print(f"[ATR] error {symbol}: {e}")
        return 0.0

def relaxed_trend_gate(symbol, side, btc_symbol="BTCUSDT"):
    ok, ctx = trend_gate(symbol, side, btc_symbol)
    inst_ok = bool(ctx.get("inst_ok"))
    btc_ok = bool(ctx.get("btc_ok"))
    if is_major(symbol):
        return (inst_ok and btc_ok), ctx  # мажоры: нужен согласованный тренд
    else:
        return inst_ok, ctx               # альты: достаточно тренда инструмента
def explain_reason(symbol, side, zone):
    """
    Возвращает строку с перечислением ТОЛЬКО тех предпосылок (из 13),
    которые реально выполняются по данным.
    Использует H4/H1/M5/D и параметры зоны (ATR, ATR_rem, consolidation, уровни).
    """
    reasons = []

    # Базовые данные и уровень
    lvl = zone.get("level_1h") or zone.get("level_4h") or zone.get("level_15m")
    if lvl is None and isinstance(zone.get("zone"), (list, tuple)) and len(zone.get("zone")) >= 2:
        lvl = sum(zone["zone"])/2.0
    atr = float(zone.get("ATR", 0.0)) if zone.get("ATR") is not None else 0.0

    # Загрузка свечей
    try:
        h4 = fetch_cached_ohlcv(symbol, "240", limit=10) or []
        h1 = fetch_cached_ohlcv(symbol, "60", limit=20) or []
        m5 = fetch_cached_ohlcv(symbol, "5", limit=20) or []
        d1 = fetch_cached_ohlcv(symbol, "D", limit=2) or []
    except Exception:
        h4, h1, m5, d1 = [], [], [], []
    pct_log = zone.setdefault('pct_diff_log', [])



    def pct_diff(a, b, tag=None):
        try:
            res = abs(float(a) - float(b)) / max(1e-9, abs(float(b)))
            try:
                pct_log.append({'tag': tag, 'a': float(a), 'b': float(b), 'res': float(res)})
            except Exception:
                pass
            return res
        except Exception:
            try:
                pct_log.append({'tag': tag, 'a': a, 'b': b, 'res': None, 'error': True})
            except Exception:
                pass
            return 1.0

    # 1) Паранормальные H4/H1 + мелкие M5
    try:
        big_h4 = any(abs(c[3]-c[0]) > max(atr, 0.006 * c[0]) for c in h4[-3:]) if h4 else False
        big_h1 = any(abs(c[3]-c[0]) > max(atr*0.6, 0.004 * c[0]) for c in h1[-4:]) if h1 else False
        small_m5 = all((c[1]-c[2])/max(1e-9, c[0]) < 0.002 for c in m5[-5:]) if m5 else False
        if (big_h4 or big_h1) and small_m5:
            reasons.append("подход на паранормальных барах")
    except Exception:
        pass

    # 2) Закрытие дневной свечи прямо на уровне
    try:
        if d1 and lvl is not None:
            close = d1[-1][3]
            if pct_diff(close, lvl, tag='d1_close_vs_level') < 0.0015:
                reasons.append("дневная свеча закрылась на уровне")
    except Exception:
        pass

    # 3) Начало ATR: топлива достаточно (ATR_rem > 60% ATR)
    try:
        atr_total = float(zone.get("ATR5", 0) or 0)
        atr_rem = float(zone.get("ATR5_rem", 0) or 0)
        if atr_total > 0 and atr_rem >= 0.6 * atr_total:
            reasons.append("много топлива (ATR5D в начале хода)")
    except Exception:
        pass

    # 4) Ближний ретест уровня (последние 6 H1 касались уровня)
    try:
        if h1 and lvl is not None:
            touches = 0
            tol = max(0.002*lvl, 0.3*atr)  # 0.2% или 0.3*ATR
            for bar in h1[-6:]:
                mid = (bar[1]+bar[2])/2
                if abs(mid - lvl) <= tol:
                    touches += 1
            if touches >= 1:
                reasons.append("ближний ретест уровня")
    except Exception:
        pass

    # 5) После сильного движения (10–20% день) нет глубокого отката
    try:
        if d1:
            o,h,l,c,v = d1[-1]
            rng = max(1e-9, h - l)
            day_move = abs(c - o) / max(1e-9, o)
            # нет глубокого отката — закрытие в верхней (для long) или нижней (для short) 20% диапазона
            if day_move >= 0.10:
                if side == "long" and (h - c) / rng <= 0.2:
                    reasons.append("нет отката после сильного движения")
                if side == "short" and (c - l) / rng <= 0.2:
                    reasons.append("нет отката после сильного падения")
    except Exception:
        pass

    # 6) Монета целый день идёт в одном направлении
    try:
        if d1:
            o,h,l,c,v = d1[0]
            if c > o:
                reasons.append("весь день в рост")
            elif c < o:
                reasons.append("весь день в падение")
    except Exception:
        pass

    # 7) Нет отката после ложного пробоя
    try:
        if d1 and len(d1) >= 1:
            o,h,l,c,v = d1[-1]
            if side == "long":
                if c > o:
                    reasons.append("весь день в рост")
                # если c < o — НЕ добавляем «весь день в падение» для long
            elif side == "short":
                if c < o:
                    reasons.append("весь день в падение")
    except Exception:
        pass

    # 8) Наличие выкупной свечи (H1 сильная бычья/медвежья)
    try:
        if h1:
            o,h,l,c,v = h1[-1]
            rng = max(1e-9, h - l)
            body = abs(c - o)
            upper_w = h - max(c,o)
            lower_w = min(c,o) - l
            if side == "long" and c > o and lower_w > body and body/rng > 0.4:
                reasons.append("выкупная свеча")
            if side == "short" and c < o and upper_w > body and body/rng > 0.4:
                reasons.append("продажная свеча")
    except Exception:
        pass

    # 9) Поджатие с прилипанием к уровню
    try:
        if h1 and lvl is not None and len(h1) >= 6:
            tol = max(0.002*lvl, 0.3*atr)
            closes = [bar[3] for bar in h1[-6:]]
            near = [abs(c - lvl) <= tol for c in closes]
            trend_up = all(closes[i] >= closes[i-1] for i in range(1, len(closes)))
            trend_dn = all(closes[i] <= closes[i-1] for i in range(1, len(closes)))
            if side == "long" and any(near) and trend_up:
                reasons.append("поджатие к уровню")
            if side == "short" and any(near) and trend_dn:
                reasons.append("поджатие к уровню сверху")
    except Exception:
        pass

    # 10) V-формация (минимум/максимум в центре)
    try:
        if h1 and len(h1) >= 7:
            closes = [bar[3] for bar in h1[-7:]]
            idx_min = closes.index(min(closes))
            idx_max = closes.index(max(closes))
            if side == "long" and 2 <= idx_min <= 4 and closes[-1] > closes[0]:
                reasons.append("V-формация")
            if side == "short" and 2 <= idx_max <= 4 and closes[-1] < closes[0]:
                reasons.append("V-формация")
    except Exception:
        pass

    # 11) После паранормальной свечи возле уровня нет глубокого отката
    try:
        if h1:
            o,h,l,c,v = h1[-1]
            rng = max(1e-9, h - l)
            body = abs(c - o)
            if body/rng >= 0.9 and lvl is not None:
                # текущая цена держится в направлении хода
                if side == "long" and c >= o and c >= h - 0.2*rng:
                    reasons.append("нет отката после паранормальной свечи")
                if side == "short" and c <= o and c <= l + 0.2*rng:
                    reasons.append("нет отката после паранормальной свечи")
    except Exception:
        pass

    # 12) Плавный подход к уровню (малые тела, направленно)
    try:
        if h1 and lvl is not None and len(h1) >= 4:
            bodies_ok = all(abs(b[3]-b[0])/max(1e-9,b[0]) < 0.003 for b in h1[-4:])
            if bodies_ok:
                dir_ok = True
                if side == "long":
                    dir_ok = h1[-1][3] >= h1[-4][3]
                else:
                    dir_ok = h1[-1][3] <= h1[-4][3]
                if dir_ok:
                    reasons.append("плавный подход к уровню")
    except Exception:
        pass
    # 13) Классификация консолидации относительно уровня (над/под + поддержка/сопротивление, взаимоисключающе)
    try:
        if h1 and (lvl is not None) and len(h1) >= 8:
            # Последние 8 H1 свечей: [open, high, low, close, volume]
            last8  = h1[-8:]
            opens  = [c[0] for c in last8]
            highs  = [c[1] for c in last8]
            lows   = [c[2] for c in last8]
            closes = [c[3] for c in last8]

            # Фолбэк ATR: средний диапазон, если atr отсутствует
            local_ranges = [(hi - lo) for hi, lo in zip(highs, lows)]
            rng_avg = (sum(local_ranges) / max(1, len(local_ranges))) if local_ranges else 0.0
            atr_eff = float(atr) if (isinstance(atr, (int, float)) and atr is not None and atr > 0) else float(rng_avg)

            lvl_val = float(lvl)

            # Пороги
            tol = max(0.002 * lvl_val, 0.3 * atr_eff)  # допуск к уровню
            small_flags  = [ (hi - lo) / max(1e-9, abs(op)) <= 0.02 for hi, lo, op in zip(highs, lows, opens) ]  # ~2%
            near_mid_flags = [ abs(((hi + lo) / 2.0) - lvl_val) <= max(0.005 * lvl_val, 0.5 * atr_eff) for hi, lo in zip(highs, lows) ]

            # Компактность окна
            rel_range  = (max(highs) - min(lows)) / max(1e-9, lvl_val)
            compact_ok = rel_range <= 0.012  # ~1.2% за 8 свечей

            # Касания уровня
            near_low_flags  = [ abs(lo - lvl_val) <= tol for lo in lows ]   # поддержка
            near_high_flags = [ abs(hi - lvl_val) <= tol for hi in highs ]  # сопротивление

            # Закрытия
            closes_above = [ cl >  lvl_val for cl in closes ]
            closes_below = [ cl <  lvl_val for cl in closes ]

            # Агрегаты
            small_ok = sum(small_flags)  >= 5
            near_ok  = sum(near_mid_flags) >= 5

            nl = sum(near_low_flags)
            nh = sum(near_high_flags)
            ca = sum(closes_above)
            cb = sum(closes_below)

            # Взаимоисключающий тип уровня
            if (nl - nh) >= 1:
                level_type = "support"
            elif (nh - nl) >= 1:
                level_type = "resistance"
            else:
                level_type = "support" if ca >= cb else "resistance"

            # Сохраняем подробный дебаг
            try:
                zone["consolidation_lvl_used"] = lvl_val
                zone["consolidation_debug"] = {
                    "level_type": level_type,
                    "nl": int(nl), "nh": int(nh),
                    "ca": int(ca), "cb": int(cb),
                    "tol": float(tol), "atr_eff": float(atr_eff),
                    "small_ok": bool(small_ok),
                    "near_ok": bool(near_ok),
                    "compact_ok": bool(compact_ok),
                }
            except Exception:
                pass

            # Единственная причина по majority закрытий
            if small_ok and (near_ok or compact_ok):
                if level_type == "support":
                    if ca >= 4:
                        reasons.append("Консолидация над уровнем поддержки")
                    elif cb >= 4:
                        reasons.append("Консолидация под уровнем поддержки")
                else:  # resistance
                    if ca >= 4:
                        reasons.append("Консолидация над уровнем сопротивления")
                    elif cb >= 4:
                        reasons.append("Консолидация под уровнем сопротивления")
    except Exception:
        pass
    # Формирование строки

    # Сохраняем предпосылки для логов/таблицы (русский стиль, как в Telegram)
    try:
        _uniq = sorted(set(reasons))
        if _uniq:
            zone['entry_reasons_list'] = _uniq
            zone['entry_reasons_text'] = ", ".join(_uniq)
            zone['entry_reasons_ru'] = "потому что " + zone['entry_reasons_text']
    except Exception:
        pass
    if not reasons:
        return ""  # ничего не добавляем
    prefix = "🟢 LONG, потому что " if side == "long" else "🔴 SHORT, потому что "
    return prefix + ", ".join(sorted(set(reasons)))

def get_thr_with_src(symbol, side):
    # A) групповые/символьные из EVAL_DIR/ART_DIR
    if GROUP_THR:
        try:
            gk = _group_key(symbol, side)
            node = None
            if isinstance(GROUP_THR, dict):
                if "groups" in GROUP_THR and isinstance(GROUP_THR["groups"], dict):
                    node = GROUP_THR["groups"].get(gk)
                else:
                    node = GROUP_THR.get(gk)
            if node is not None:
                if isinstance(node, dict):
                    val = node.get("threshold") or node.get("thr") or node.get("value")
                else:
                    val = node
                if isinstance(val, (int, float)) and val > 0:
                    return float(val), f"group:{gk}"
        except Exception:
            pass
    # B) GLOBAL_THR из EVAL_DIR
    try:
        if isinstance(GLOBAL_THR, (int, float)) and GLOBAL_THR > 0:
            return float(GLOBAL_THR), "global@EVAL_DIR"
    except Exception:
        pass
    # C) локальный thresholds.json (./nn_eval_out/thresholds.json)
    try:
        _load_thr_cache()
        grp_key = "majors" if is_major(symbol) else "alts"
        for k in (f"{symbol}:{side}", symbol, grp_key, "prec60", "global_f1"):
            v = _thr_cache.get(k)
            if isinstance(v, (int, float)) and v > 0:
                return float(v), k
    except Exception:
        pass
    return THR_DEFAULT, "default"

def _features_quality(arr):
    import numpy as np
    a = np.asarray(arr, dtype=float).ravel()
    return {
        "nonzero": int(np.count_nonzero(np.abs(a) > 1e-12)),
        "uniq":    int(len(np.unique(np.round(a, 6)))),
        "std":     float(np.std(a)),
    }

def _is_degenerate_features(arr) -> bool:
    m = _features_quality(arr)
    return (m["nonzero"] <= 3) or (m["uniq"] <= 2) or (m["std"] < 1e-6)


def send_signal(symbol, price, zone, side, forecast_time):

    global LAST_SENT_TS, DEDUP_SECONDS
    now_ts = time.time()
    last_ts = LAST_SENT_TS.get(symbol)
    if last_ts is not None and (now_ts - last_ts) < DEDUP_SECONDS:
        try:
            print(f"[dedup] Skip signal for {symbol}: sent {(now_ts - last_ts):.0f}s ago (< {DEDUP_SECONDS}s)")
        except Exception:
            pass
        return
    kind = normalize_zone_kind(zone, price)  # kind: 'support' или 'resistance'
    defer_trend_fail = False
    print(f"[DEBUG] ▶ send_signal вызван для {symbol} {side.upper()} по цене {price}")
    # --- Тренд-гейт 1H + BTC-гейт ---
    ok, ctx = relaxed_trend_gate(symbol, side)
    inst_ok = bool(ctx.get("inst_ok", False))
    btc_ok  = bool(ctx.get("btc_ok", False))
    if is_major(symbol):
        if not (inst_ok and btc_ok):
            log_skipped_signal(symbol, side, 'trend_gate_fail_major', ctx)
    
            print(f"[skip] trend_gate_fail_major {symbol} {side} ctx={ctx}")
        else:
            if not inst_ok:
                log_skipped_signal(symbol, side, 'trend_gate_fail_alt', ctx)
    
            print(f"[skip] trend_gate_fail_alt {symbol} {side} ctx={ctx}")
            # Проверка локального тренда по монете (мягкий буфер 0.15%)
    candles = fetch_ohlcv(symbol, interval="15", limit=2)
    if len(candles) >= 2:
        prev_close = candles[-2][3]
        last_close = candles[-1][3]
        delta = (last_close - prev_close) / prev_close
        if side == 'long' and delta < -0.0100:
                log_skipped_signal(symbol, side, 'local_15m_downtrend', {'d15': float(delta)})
                print(f"[skip] local_15m_downtrend {symbol} d15={delta:.3%}")
                return
        if side == 'short' and delta > 0.0100:
                log_skipped_signal(symbol, side, 'local_15m_uptrend', {'d15': float(delta)})
                print(f"[skip] local_15m_uptrend {symbol} d15={delta:.3%}")
                return
            # BTC filter (15m)
    btc_candles = fetch_ohlcv("BTCUSDT", interval="15", limit=2)
    if len(btc_candles) >= 2:
        btc_prev = btc_candles[-2][3]
        btc_last = btc_candles[-1][3]
        btc_delta = (btc_last - btc_prev) / btc_prev
        if side == "long" and btc_delta < -0.0120:
            log_skipped_signal(symbol, side, 'btc_filter_down')
        
            print(f"[skip] btc_filter_down vs BTC 15m {symbol}")
        if side == "short" and btc_delta > 0.0120:
            log_skipped_signal(symbol, side, 'btc_filter_up')
    
            print(f"[skip] btc_filter_up vs BTC 15m {symbol}")# Объём
    #volume_note = get_volume_indicator(symbol)
    #allow = True
    #if is_major(symbol):
    #    allow = ("🟢" in volume_note)
    #else:
        # Для альтов допустим 🟡, но потребуем высокую вероятность >=70 позже
    #    allow = ("🟢" in volume_note) or ("🟡" in volume_note)
    #if not allow:
    #    print(f"⛔ Пропущен сигнал для {symbol} из-за низкого объёма: {volume_note}")
    #    log_skipped_signal(symbol, side, 'low_volume', {'note': volume_note})
    volume_note = get_volume_indicator(symbol)
    # фильтр по объёмам отключён, теперь сигнал не блокируется
    allow = True
    # Расчёт ATR, SL, TP, Leverage (1H-ATR c полом 0.25%)
    atr_1h = calculate_atr_1h(symbol, period=14)
    atr_floor = max(atr_1h, price * 0.0025)
    tp_mult = 3.0 if is_major(symbol) else 2.5

    if side == 'long':
        sl = zone['zone'][0] - atr_floor
        tp = zone['zone'][1] + tp_mult * atr_floor
    else:
        sl = zone['zone'][1] + atr_floor
        tp = zone['zone'][0] - tp_mult * atr_floor

    try:
        X = _mk_features_for_bot(symbol, side)
        q = _features_quality(X)
        if q["nonzero"] <= 3:
            print(f"[nn-debug] {symbol} {side} features look dead: {q}")
    except Exception:
        pass

    rr_ratio = abs(tp - price) / max(1e-9, abs(price - sl))
    leverage = int(calc_leverage_atr(symbol, price, atr_1h, rr_ratio))
    zone['sl'] = round(sl, 6)
    record_sl_hit(symbol, zone['sl'])
    zone['tp'] = round(tp, 6)
    zone['leverage'] = leverage
    # Вероятность
    prob = compute_probability(symbol, side, zone, rr_ratio)
    # Для альтов требуем prob>=70, если объём лишь 🟡
    if (not is_major(symbol)) and ("🟡" in volume_note) and (prob < 70):
        log_skipped_signal(symbol, side, "prob_low_on_yellow_volume", {"prob": prob})

        print(f"[skip] prob_low_on_yellow_volume {symbol} {side} prob={prob}")
        zone['probability_pct'] = prob

    # 1) ИНИЦИАЛИЗАЦИЯ entry_low/high ИЗ zone['zone'] (или около уровня)
    # --- РАННЯЯ КЛАССИФИКАЦИЯ ПО/ОТБОЙ (до любых логов, где используется `po`) ---
    try:
        ohlcv_15m = zone.get("ohlcv_15m") or fetch_cached_ohlcv(symbol, "15", limit=120)
    except Exception:
        ohlcv_15m = []
    try:
        po = classify_po(side, price, zone, ohlcv_15m)
    except Exception:
        po = "N/A"
    zone["po"] = po

    zl = zone.get("zone")
    if isinstance(zl, (list, tuple)) and len(zl) >= 2:
        entry_low, entry_high = float(zl[0]), float(zl[1])
    else:
    # фолбэк вокруг уровня, если зона отсутствует
        lvl = float(zone.get("lvl") or zone.get("level_1h") or zone.get("level_4h") or price)
        span = max(0.001 * max(lvl, 1.0), 0.0005 * max(price, 1.0))
        entry_low, entry_high = lvl - span, lvl + span

# нормализуем порядок (low <= high)
    if entry_low > entry_high:
        entry_low, entry_high = entry_high, entry_low

    # 2) КОРРЕКТИРОВКА ВХОДА С УЧЁТОМ роли уровня (поддержка/сопротивление)
    entry_low, entry_high = adjust_entry_for_zone(side, zone, entry_low, entry_high)

# 3) ЗАПИСЫВАЕМ ОБРАТНО В ЗОНУ (чтобы дальше всё работало консистентно)
    zone["zone"] = [round(entry_low, 6), round(entry_high, 6)]

    entry_low, entry_high = adjust_entry_for_zone(side, zone, entry_low, entry_high)

    # --- Эвристический гейт (prob & RR) ---
    # забираем из zone или из локальных расчётов
    rr   = float(zone.get("rr", rr if 'rr' in locals() else 0))
    prob = float(zone.get("prob", prob if 'prob' in locals() else 0))
    ok_nn = bool(globals().get('last_nn_ok', True))  # или твоя переменная NN-прохода

    MIN_RR = 1.5
    EPS_RR = 1e-3  # допуск на плавающую точку

    # было: if rr < MIN_RR:
    if rr + EPS_RR < MIN_RR:
        log_skipped_signal(symbol, side, 'heuristics_reject', {
            "rr": round(rr, 3),
            "prob": float(prob),
            "nn_passed": bool(ok_nn),
            "po": zone.get("po","N/A"),
        })
        return

    ok_heur = (prob >= PROB_MIN) and (rr >= RR_MIN - EPS_RR)
    if not ok_heur:
        log_skipped_signal(symbol, side, 'heuristics_reject', {
            "rr": round(rr, 3),
            "prob": float(prob),
            "nn_passed": bool(ok_nn),
            "po": zone.get("po","N/A"),
        })
        return
    # --- Heuristics gate (строже после NN) ---
    # --- Heuristics gate (строже после NN) ---
    try:
        baseline_prob = 60 if is_major(symbol) else 65
        baseline_rr = 1.5
        border_prob = baseline_prob - 5
        border_rr = 1.4
        passed_nn = bool(zone.get('nn_used')) and (zone.get('nn_p') is not None) and (zone.get('nn_thr') is not None) and (zone['nn_p'] > zone['nn_thr']) and (zone['nn_p'] >= 0.66)
        heur_ok = (rr_ratio >= baseline_rr) and (prob >= baseline_prob)
        borderline_ok = passed_nn and (rr_ratio >= border_rr) and (prob >= border_prob)
        if not (heur_ok or borderline_ok):
                    log_skipped_signal(symbol, side, 'heuristics_reject', {
            "rr": round(rr, 3),
            "prob": float(prob),
            "nn_passed": bool(ok_nn),
            "po": zone.get("po","N/A"),
        })
        print(f"[skip] heuristics_reject {symbol} {side} rr={round(float(rr_ratio),3)} prob={int(prob)} nn_passed={bool(passed_nn)}")
        return
    except Exception as _he:
        # на всякий случай не роняем рассылку
        pass
        pass


    # Формирование и отправка сообщения (оставляем как было ниже)
    icons = {
        "Отскок после импульса": "💥",
        "Поджатие": "📉",
        "Затухание": "🌫",
        "Импульсный пробой": "🚀",
        "Ложный пробой": "🔄",
        "Нарастающий объём": "📊",
        "Разворотная свеча": "🔃",
        "Divergence": "📐",
        "Стандарт": "⚪"
    }
    signal_type = zone.get("type", "Стандарт")
    if isinstance(signal_type, list):
        signal_type = ", ".join(signal_type)
    тип_иконка = icons.get(signal_type, "⚪")
    patterns_str = ", ".join(zone.get("patterns", [])) if zone.get("patterns") else ""
    reason = explain_reason(symbol, side, zone)


    # === Add EMA20/EMA200 for instrument and BTC, plus BTC price & 15m delta ===
    ema20_inst = ema200_inst = None
    ema20_btc = ema200_btc = None
    btc_price_line = ""
    try:
        _df_inst_1h = get_df(symbol, "60", 260)
        if _df_inst_1h is not None and len(_df_inst_1h) >= 205:
            ema20_inst = float(_ema(_df_inst_1h["close"], 20).iloc[-1])
            ema200_inst = float(_ema(_df_inst_1h["close"], 200).iloc[-1])
    except Exception as _e:
        pass
    try:
        _df_btc_1h = get_df("BTCUSDT", "60", 260)
        if _df_btc_1h is not None and len(_df_btc_1h) >= 205:
            ema20_btc = float(_ema(_df_btc_1h["close"], 20).iloc[-1])
            ema200_btc = float(_ema(_df_btc_1h["close"], 200).iloc[-1])
    except Exception as _e:
        pass
    # BTC last price and 15m delta
    try:
        _btc_15 = fetch_ohlcv("BTCUSDT", interval="15", limit=2)
        if _btc_15 and len(_btc_15) >= 2:
            _btc_prev = float(_btc_15[-2][3]); _btc_last = float(_btc_15[-1][3])
            _btc_delta = (_btc_last - _btc_prev) / max(1e-9, _btc_prev)
            _sgn = "+" if _btc_delta >= 0 else ""
            btc_price_line = f"₿ BTC: ${_btc_last:.2f} (15m Δ {_sgn}{_btc_delta*100:.2f}%)"
    except Exception:
        pass
    # === /Add EMA20/EMA200 and BTC lines ===
    # NN flag (безопасно)
    if 'nn_used' not in zone:
        zone['nn_used'] = bool('_NN_AVAILABLE' in globals() and _NN_AVAILABLE)    

    # NN p/thr (мягко, без падений)
    if zone['nn_used']:
        try:
            p_nn, _ctx = nn_score_for_signal(symbol, side, zone)
            try:
                rr_ratio = abs(tp - price) / max(1e-9, abs(price - sl))
                thr = get_nn_threshold(symbol, side)  # ← сразу берём финальный порог
            except Exception:
                    thr = None
            if p_nn is not None:
                zone['nn_p'] = float(p_nn)
                zone["nn_ctx"] = _ctx   # ← тут был баг: раньше стояло ctx, переменной нет
            if thr is not None:
                zone['nn_thr'] = float(thr)

        except Exception as _nne:
            zone.setdefault('nn_error', str(_nne))


    # NN hard gate — пропускаем, если работаем в фолбэке (нет модели) или p <= thr
    if zone.get('nn_used') and (zone.get('nn_p') is not None) and (zone.get('nn_thr') is not None):
        nn_ctx = zone.get('nn_ctx') or {}
        fallback_mode = bool(nn_ctx.get('fallback')) or (MODEL is None)
        if not fallback_mode:
            if not (zone['nn_p'] > zone['nn_thr']):
                try:
                    log_skipped_signal(symbol, side, 'nn_rejected', {'p_nn': zone['nn_p'], 'thr': zone['nn_thr']})
                except Exception:
                    pass
                print(f"[skip] nn_rejected {symbol} {side} p={zone['nn_p']:.3f} thr={zone['nn_thr']:.3f}")
                return
    
        try:
            nn_used = zone.get('nn_used')
            p_nn = zone.get('nn_p')
            thr = zone.get('nn_thr')
            if nn_used is None:
                nn_used = False
            if (p_nn is not None) and (thr is not None):
                p_pct = f"{float(p_nn)*100:.1f}".replace('.', ',')
                thr_pct = f"{float(thr)*100:.1f}".replace('.', ',')
                nn_line = f"NN: {'ON' if nn_used else 'OFF'} (Успех сделки = {p_pct}%, Допуск успеха сделки = {thr_pct}%)"
            else:
                nn_line = f"NN: {'ON' if nn_used else 'OFF'}"
        except Exception:
                nn_line = f"NN: {'ON' if zone.get('nn_used') else 'OFF'}"
        # --- 15m OHLCV для классификации пробой/отбой ---
        ohlcv_15m = zone.get("ohlcv_15m")  # вдруг уже передали сверху
        if not ohlcv_15m:
            try:
        # если у тебя функция называется иначе, используй её
        # для Bybit чаще подходят интервалы '15' или '15m'
                ohlcv_15m = fetch_cached_ohlcv(symbol, '15', limit=120)
        # если у тебя используется '15m', раскомментируй следующую строку:
        # ohlcv_15m = fetch_cached_ohlcv(symbol, '15m', limit=120)
            except Exception as e:
                print(f"[warn] fetch 15m ohlcv failed for {symbol}: {e}")
                orohlcv_15m = []

# на всякий случай подкинем ключи уровня в zone, если их нет
        if "lvl" not in zone:
            zone["lvl"] = zone.get("level") or zone.get("lvl_price") or 0.0
        if "tol" not in zone:
            zone["tol"] = zone.get("tolerance") or zone.get("tol_abs") or 0.0
        if "atr" not in zone:
            zone["atr"] = zone.get("atr_15m") or zone.get("atr") or 0.0


        # у тебя уже есть zone, side и массив последних 15m свечей
        po = classify_po(side, price, zone, ohlcv_15m)
        zone['type'] = po  # чтобы попало в телеграм-сообщение
        pref_breakout = bool(globals().get('PREF_BREAKOUT', True))
        if pref_breakout and ('пробой' not in str(po).lower()):
            log_skipped_signal(symbol, side, 'type_not_breakout', {
                "po": zone.get("po","N/A"),
                "lvl": zone.get('lvl'),
                "zone": zone.get('zone'),
                "side": side,
            })
            return


        ohlcv_15m = zone.get("ohlcv_15m")
        if not ohlcv_15m:
            try:
        # Подставь правильный интервал для твоей функции
                ohlcv_15m = fetch_cached_ohlcv(symbol, '15', limit=120)  # или '15m'
            except Exception as e:
                print(f"[warn] fetch 15m ohlcv failed for {symbol}: {e}")
                ohlcv_15m = []

        po_str = str(po or '').lower()
        is_breakout = ('пробой' in po_str)

        if PREF_BREAKOUT and not is_breakout:
            log_skipped_signal(symbol, side, 'type_not_breakout', {
                "po": zone.get("po","N/A"),
                "lvl": zone.get('lvl'),
                "zone": zone.get('zone'),
                "side": side,
            })
            return  # оставляем строгий режим, если он включён
# иначе пропуска НЕ делаем — идём дальше
    
        # Безопасное создание списка частей сообщения
        try:
            parts
        except NameError:
            parts = []
        parts = [f"{'🟢 сигнал' if side == 'long' else '🔴 сигнал'} {symbol}"]
    # Append EMA lines & BTC info right after header/reasons
    if (ema20_inst is not None) and (ema200_inst is not None):
        parts.append(f"📐 EMA20/EMA200 (1H): {ema20_inst:.6f} / {ema200_inst:.6f}")
    if (ema20_btc is not None) and (ema200_btc is not None):
        _t = "↑" if ema20_btc > ema200_btc else "↓"
        parts.append(f"₿ BTC EMA20/EMA200 (1H): {ema20_btc:.2f} / {ema200_btc:.2f} {_t}")
    if btc_price_line:
        parts.append(btc_price_line)


    # Показать процент остатка ATR за 24 часа
        try:
            atr_total = float(zone.get("ATR5") or 0)
            atr_rem = float(zone.get("ATR5_rem") or 0)
            if atr_total > 0:
                parts.append(f"⚙ ATR5D остаток: {atr_rem / atr_total * 100:.0f}%")
        except Exception:
            pass
        # reason может отсутствовать — страхуемся
        try:
            reason
        except NameError:
            reason = ''
        if reason:
            parts.append(str(reason))  # добавим строку с предпосылками ТОЛЬКО если они есть
    
        parts += [
            f"🕳 Уровень 4H: ${zone.get('level_4h', '-')}",
            f"🕳 Уровень 1H: ${zone.get('level_1h', '-')}",
            f"🕳 Уровень 15M: ${zone.get('level_15m', '-')}",
            f"Тип сигнала: {тип_иконка} {signal_type}",
            ("🔁 !" if zone.get("consolidation") else ""),
            f"🎯 Вход: ${zone['zone'][0]} – ${zone['zone'][1]}",
            f"🎯 TP: ${tp}",
            f"🛡 SL: ${sl}",
            f"⚙ Плечо: x{leverage}",
            f"📊 Вероятность: {prob}%",
            #nn_line,
            f"{'📉' if zone['po'] == 'Отбой' else '📈'} {zone['po']}",
            f"{volume_note}",
            (f"🕯 Паттерны: {patterns_str}" if patterns_str else ""),
            f"Цена на момент прогноза: ${zone['price']}",
            f"🕒 Прогноз: {forecast_time}",
        ]

        # === NN блок: показать в сообщении + решить PASS/FAIL ===
        # Глобальные дефолты (если вдруг не заданы выше)
        STRICT_NN   = globals().get("STRICT_NN", False)
        NN_SOFT_EPS = globals().get("NN_SOFT_EPS", 0.005)   # 0.5 п.п. в шкале 0..1

        po = classify_po(side, price, zone, ohlcv_15m)
        zone['po'] = po
        is_breakout = ('пробой' in str(po).lower())  # true для 'Импульсный пробой' и 'Пробой'
        parts.append('📈 Пробой' if is_breakout else '📉 Отбой')


        def _as_float_or_none(x):
            try:
                if x is None: return None
                return float(x)
            except Exception:
                return None

        def _to_pct(x):
            x = _as_float_or_none(x)
            if x is None: return None
            return x*100.0 if x <= 1.0 else x  # поддержка и 0..1, и 0..100

        def _nn_passes(p_val, thr_val):
            p_val   = _as_float_or_none(p_val)
            thr_val = _as_float_or_none(thr_val)
            if p_val is None or thr_val is None:
                return True  # нет порогов — не режем
            return (p_val >= thr_val) or (p_val >= (thr_val - NN_SOFT_EPS))

        # читаем значения из zone с безопасными дефолтами
        p   = zone.get("nn_p")
        thr = zone.get("nn_thr")
        grp = str(zone.get("nn_group") or globals().get("NN_GROUP", "global_f1"))

        ok_nn = _nn_passes(p, thr)
        zone["nn_passed"] = bool(ok_nn)

        p_pct   = _to_pct(p)
        thr_pct = _to_pct(thr)

        if p_pct is None or thr_pct is None:
            parts.append(f"🤖 NN: N/A — OFF [{grp}]")
        else:
    # если p/thr валидны — покажем сравнение
            p_f, thr_f = float(p_pct), float(thr_pct)
            cmp = "≥" if _as_float_or_none(p) is not None and _as_float_or_none(thr) is not None and float(p) >= float(thr) else "<"
            parts.append(f"🤖 NN: p={p_f:.2f}% {cmp} thr={thr_f:.2f}% [{grp}] — {'PASS ✅' if ok_nn else 'FAIL ❌'}")

# опционально жёстко режем по NN
        if not ok_nn and STRICT_NN:
            log_skipped_signal(symbol, side, 'nn_rejected', {"p_nn": _as_float_or_none(p), "thr": _as_float_or_none(thr), "group": grp})
            return

    # Показать два тейка и правило брейк-ивена
    try:
        tp1 = zone.get("tp1"); tp2 = zone.get("tp2") or zone.get("tp")
        if tp1 and tp2:
            parts.append(f"🎯 TP1/TP2: {tp1} / {tp2}")
            parts.append("🧷 BE: после TP1 перенос SL в безубыток")
    except Exception:
        pass
    try:
        dbg = zone.get('consolidation_debug')
        lvl_used = zone.get('consolidation_lvl_used')
        if dbg and lvl_used is not None:
            parts.append(
                f"🔎 Консол.: lvl={lvl_used:.6f}, type={dbg.get('level_type')}, nl/nh={dbg.get('nl')}/{dbg.get('nh')}, ca/cb={dbg.get('ca')}/{dbg.get('cb')}, tol={dbg.get('tol'):.6f}, atr={dbg.get('atr_eff'):.6f}"
            )
    except Exception:
        pass
    text = "\n".join([p for p in parts if p])
    notify(text)
    log_signal(symbol, side, price,  zone, forecast_time)
    try:
        LAST_SENT_TS[symbol] = time.time()
    except Exception:
        pass
    
        # Конец send_signal

def send_forecast_digest():
    forecast, updated_at = load_forecast()
    msg = f"📊 Прогноз на {updated_at}\n"

    for symbol, sides in forecast.items():
        for side in ["long", "short"]:
            for zone in sides[side]:
                msg += (
                    f"{symbol}:\n"
                    f"{'🟢 LONG' if side == 'long' else '🔴 SHORT'} \n"
                    f"💰 Вход: ${zone['zone'][0]}–${zone['zone'][1]}\n"
                    f"🎯 TP: ${zone['tp']} | 🛡 SL: ${zone['sl']} | ⚙ x{zone['leverage']}\n"
                    f"{'📉' if side == 'Отбой' else '📈'}${zone['po']}\n"
                    f"\n"
                )
    notify(msg.strip())
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ Ежедневная рассылка прогноза отправлена.")

def run_scheduler():
    if not ENABLE_DAILY_DIGEST:
        return
    sched.every().day.at("20:00").do(send_forecast_digest)
    while True:
        sched.run_pending()
        time.sleep(30)

# ---------------------------- ЗАПУСК БОТА -----------------------------
threading.Thread(target=monitor).start()
if ENABLE_DAILY_DIGEST:
    threading.Thread(target=run_scheduler).start()
print("🤖 Бот запущен.")
bot.polling(none_stop=True)





def is_strong_breakout(symbol: str, side: str, level: float) -> bool:
    try:
        c = fetch_cached_ohlcv(symbol, '15', limit=60)
        if not c or len(c) < 3:
            return False
        o, h, l, cl, v = c[-1]
        rng = h - l
        if rng <= 0:
            return False
        body_ratio = abs(cl - o) / max(1e-9, rng)
        avgv = sum([x[4] for x in c[-51:-1]]) / max(1, len(c[-51:-1]))
        crossed = (cl > level and o <= level) if side == 'long' else (cl < level and o >= level)
        return (body_ratio >= 0.60) and (v >= 2.0 * avgv) and crossed
    except Exception:
        return False
