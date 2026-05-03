"""
Alignment experiments on the real 6Q benchmark.

Tests three ideas inspired by recent papers:
  1. OT cluster alignment (PAP paper, Hu et al.) — entropic Sinkhorn between
     per-cluster province distributions and the survey province distribution.
  2. Adaptive posts-per-cluster (UQ paper, conceptual analog) — sample more
     posts for high-variance clusters until induced distribution stabilises.
  3. Age + area (province) alignment on M1 / M2 / M4 — importance-weight
     clusters by how closely their demographic profile matches the survey.

For each method M ∈ {M0 SSR, M1 LLM-dist, M2 SSR+LLM-dist w=0.5,
M4 SSR+ML-steer w=0.5} and each weight scheme
∈ {default, IS_prov, IS_age, IS_prov+age, OT_prov, OT_prov+age}, we compute
mean JS on the 6 original questions.

Output: results/alignment_6q.json
"""
import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import jensenshannon
from sentence_transformers import SentenceTransformer

from persona_vectors import load_model, load_persona_vectors
from steered_ssr import SteeringHook, ssr_score, generate_anchors_local

# ─── Paths ───
DATA_DIR = "/data/chenhongrui/business/data"
RESULTS_DIR = "/data/chenhongrui/business/results"
MODEL_PATH = "/data/chenhongrui/models/Qwen3-8B"
EMBED_PATH = "/data/chenhongrui/models/bge-base-zh-v1.5"
SURVEY_CSV = (f"{DATA_DIR}/2025-05-31-10-01-50_EXPORT_CSV_19470858_"
              f"516_ds_survey_feedback_0.csv")

ORIG_6 = ["8.0/qsingle_3", "8.0/qsingle_4", "9.0/qsingle_3",
          "9.0/qsingle_4", "9.0/qsingle_5", "11.0/qsingle_3"]

AGE_BINS = ["<24", "24-35", "35-50", "50+"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─── Data loading ───

def load_orig6():
    with open(f"{RESULTS_DIR}/all_questions_expanded.json") as f:
        expanded = json.load(f)
    with open(f"{DATA_DIR}/3_cluster_topics.json") as f:
        topics = json.load(f)
    df = pd.read_csv(f"{DATA_DIR}/2_meaningful_df.csv")
    questions = []
    for sk, sv in expanded.items():
        for sub, qs in sv.items():
            for qid, qd in qs.items():
                key = f"{sub}/{qid}"
                if key in ORIG_6:
                    questions.append({
                        "key": key,
                        "question": qd["question"],
                        "options": qd["options"],
                        "true_distribution": qd["true_distribution"],
                        "cluster_weights": qd["cluster_weights"],
                    })
    questions.sort(key=lambda x: ORIG_6.index(x["key"]))
    return questions, topics, df


def get_top_clusters(questions, n=10):
    aw = {}
    for qd in questions:
        for cid, w in qd["cluster_weights"].items():
            aw[cid] = aw.get(cid, 0) + w
    return sorted(aw, key=lambda c: -aw[c])[:n]


def js_score(td, pred):
    t = np.array(list(td.values()), dtype=float)
    t /= t.sum()
    p = pred / (pred.sum() + 1e-10)
    return float(jensenshannon(t, p) ** 2)


def agg(cpmfs, weights, cids):
    n = len(next(iter(cpmfs.values())))
    a, tw = np.zeros(n), 0.0
    for c in cids:
        if c in cpmfs:
            w = weights.get(c, 0)
            a += w * cpmfs[c]
            tw += w
    return a / tw if tw > 0 else a


# ─── Per-cluster distribution computation ───

def compute_ssr_pmfs(encoder, qd, cids, df, aembs, n_posts=50, seed=42):
    cpmfs = {}
    for cid in cids:
        posts = df[df["cluster_label"] == int(cid)]["content_desc"].dropna()
        if len(posts) < 3:
            continue
        samp = posts.sample(min(n_posts, len(posts)), random_state=seed).tolist()
        pmfs = [ssr_score(p, encoder, aembs)[0] for p in samp]
        cpmfs[cid] = np.mean(pmfs, axis=0)
    return cpmfs


def compute_llmdist_pmfs(model, tokenizer, qd, cids, df, topics, n_ex=8,
                         n_samples=3):
    opts = qd["options"]
    opts_str = "\n".join(f"{i+1}. {o}" for i, o in enumerate(opts))
    cpmfs = {}
    for cid in cids:
        posts = df[df["cluster_label"] == int(cid)]["content_desc"].dropna()
        if len(posts) < 3:
            continue
        samp = posts.sample(min(n_ex, len(posts)), random_state=42).tolist()
        post_text = "\n".join(f"- {p[:100]}" for p in samp)
        topic = topics.get(cid, "未知")[:60]
        prompt = (
            f"你是一位消费者调研分析专家。\n\n"
            f"以下是某类消费者群体的社交媒体帖子样本（话题：{topic}）：\n{post_text}\n\n"
            f"基于这些帖子反映的消费者特征和态度，请估计这个群体在以下问卷问题中各选项的选择比例。\n\n"
            f"问题：{qd['question']}\n选项：\n{opts_str}\n\n"
            f"请严格按JSON格式输出各选项的比例（总和为1）：\n"
            f'{{"distribution": [选项1比例, 选项2比例, ...]}}'
        )
        pmfs_collected = []
        for _ in range(n_samples):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                              max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=200,
                                    do_sample=True, temperature=0.7, top_p=0.9)
            resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
            resp = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL).strip()
            m = re.search(r'\[([^\]]+)\]', resp)
            if m:
                try:
                    vals = [float(x.strip()) for x in m.group(1).split(",")]
                    if len(vals) == len(opts):
                        arr = np.clip(np.array(vals), 0, None)
                        if arr.sum() > 0:
                            pmfs_collected.append(arr / arr.sum())
                            continue
                except Exception:
                    pass
            pmfs_collected.append(np.ones(len(opts)) / len(opts))
        cpmfs[cid] = np.mean(pmfs_collected, axis=0)
    return cpmfs


def compute_mlsteer_pmfs(model, tokenizer, encoder, qd, cids, pvecs, aembs,
                          alpha=0.1, layers=(16, 20, 24), n_resp=5):
    prompt = (
        "作为一位有相关经验的消费者，请回答以下问卷问题。\n"
        "请用自然的语言表达你的真实想法和感受，不需要选择选项。\n\n"
        f"问题：{qd['question']}\n选项：{'、'.join(qd['options'])}\n\n你的回答："
    )
    cpmfs = {}
    for cid in cids:
        if int(cid) not in pvecs:
            continue
        vec = pvecs[int(cid)]["vector"]
        pmfs = []
        for _ in range(n_resp):
            hooks = [SteeringHook(vec, alpha, l) for l in layers]
            for h in hooks:
                h.attach(model)
            try:
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=200,
                                        do_sample=True, temperature=0.7, top_p=0.9)
                resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                       skip_special_tokens=True)
            finally:
                for h in hooks:
                    h.remove()
            pmf, _ = ssr_score(resp, encoder, aembs)
            pmfs.append(pmf)
        cpmfs[cid] = np.mean(pmfs, axis=0)
    return cpmfs


# ─── Survey + demographics ───

def load_survey_province_age():
    df = pd.read_csv(SURVEY_CSV, low_memory=False)
    prov, age = Counter(), Counter()
    for _, row in df.iterrows():
        try:
            content = json.loads(row['content'])
            for item in content.get('list', []):
                q = item.get('question', '')
                if '常驻地' in q or '常住地' in q:
                    for ans in item.get('answers', []):
                        if ans.get('selected'):
                            prov[ans['text']] += 1
                if '年龄' in q:
                    for ans in item.get('answers', []):
                        if ans.get('selected'):
                            age[ans['text']] += 1
        except Exception:
            pass
    return prov, age


def normalize_age(s):
    if not s:
        return "unknown"
    if any(k in s for k in ["24岁以下", "24 岁(不含)以下", "24岁（不含）以下",
                             "20及以下", "18岁以下", "21岁以下"]):
        return "<24"
    if any(k in s for k in ["24-35", "24 岁-35", "24岁-35", "25-34", "26-35"]):
        return "24-35"
    if any(k in s for k in ["35-50", "35 岁-50", "36-50", "36-45"]):
        return "35-50"
    if any(k in s for k in ["50岁以上", "50以上", "51以上", "50+"]):
        return "50+"
    return "unknown"


def compute_cluster_prov_hist(df):
    out = defaultdict(Counter)
    for _, row in df.iterrows():
        cid = row.get('cluster_label')
        prov = row.get('pred_province')
        if pd.notna(cid) and pd.notna(prov):
            out[int(cid)][prov] += 1
    return dict(out)


AGE_INFER_PROMPT = (
    "根据以下社交媒体帖子，推测作者最可能的年龄段。"
    "只基于帖子内容推断。\n\n"
    "帖子内容：\n{post}\n\n"
    '请严格按JSON格式回答：{{"age": "<24/24-35/35-50/50+/无法判断"}}'
)


def infer_cluster_age_dists(model, tokenizer, df, cluster_ids, n_sample=15):
    """Infer age per post for each cluster using LLM."""
    out = {}
    for cid in cluster_ids:
        posts = (df[df["cluster_label"] == int(cid)]["content_desc"]
                .dropna()
                .tolist())
        if len(posts) < 3:
            continue
        rng = np.random.RandomState(42)
        rng.shuffle(posts)
        posts = posts[:n_sample]
        age_counter = Counter()
        for p in posts:
            prompt = AGE_INFER_PROMPT.format(post=p[:300])
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                              max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                o = model.generate(**inputs, max_new_tokens=60, do_sample=False)
            r = tokenizer.decode(o[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)
            r = re.sub(r'<think>.*?</think>', '', r, flags=re.DOTALL).strip()
            m = re.search(r'\{[^}]+\}', r)
            if m:
                try:
                    obj = json.loads(m.group())
                    age_counter[normalize_age(obj.get("age", ""))] += 1
                except Exception:
                    pass
        if age_counter:
            out[int(cid)] = age_counter
    return out


# ─── Alignment weight schemes ───

def is_weights_prov(cluster_prov, sm_prov, survey_prov, smoothing=1.0):
    """Cluster importance via province. w_c = Σ P_survey(p) · P_cluster(p) / P_SM(p)."""
    all_p = sorted(set(survey_prov) | set(sm_prov))
    sv_t = sum(survey_prov.values())
    sm_t = sum(sm_prov.values())
    p_sv = {p: (survey_prov.get(p, 0) + smoothing) / (sv_t + smoothing * len(all_p))
            for p in all_p}
    p_sm = {p: (sm_prov.get(p, 0) + smoothing) / (sm_t + smoothing * len(all_p))
            for p in all_p}
    ratio = {p: p_sv[p] / p_sm[p] for p in all_p}
    out = {}
    for cid, cnt in cluster_prov.items():
        ct = sum(cnt.values())
        if ct == 0:
            out[str(cid)] = 1.0
            continue
        w = sum((cnt.get(p, 0) / ct) * ratio.get(p, 1.0) for p in all_p)
        out[str(cid)] = w
    if out:
        m = np.mean(list(out.values())) or 1.0
        out = {k: v / m for k, v in out.items()}
    return out


def is_weights_age(cluster_age, survey_age, smoothing=1.0):
    """Cluster importance via age bin."""
    all_a = AGE_BINS
    sv_std = Counter()
    for s, c in survey_age.items():
        sv_std[normalize_age(s)] += c
    sv_std = {a: sv_std.get(a, 0) for a in all_a}
    sv_t = sum(sv_std.values()) or 1
    # SM overall
    sm_std = Counter()
    for cid, a in cluster_age.items():
        for k, c in a.items():
            sm_std[k] += c
    sm_t = sum(sm_std.get(a, 0) for a in all_a) or 1
    p_sv = {a: (sv_std.get(a, 0) + smoothing) / (sv_t + smoothing * len(all_a))
            for a in all_a}
    p_sm = {a: (sm_std.get(a, 0) + smoothing) / (sm_t + smoothing * len(all_a))
            for a in all_a}
    ratio = {a: p_sv[a] / p_sm[a] for a in all_a}
    out = {}
    for cid, cnt in cluster_age.items():
        ct = sum(cnt.get(a, 0) for a in all_a) or 1
        w = sum((cnt.get(a, 0) / ct) * ratio[a] for a in all_a)
        out[str(cid)] = w
    if out:
        m = np.mean(list(out.values())) or 1.0
        out = {k: v / m for k, v in out.items()}
    return out


def ot_weights_prov(cluster_prov, survey_prov, cluster_ids, epsilon=0.05,
                    n_iter=200):
    """Entropic OT between cluster province histograms and survey province dist.
    Each cluster is a distribution; survey is a single target distribution.
    Treat the problem as OT on a finite support (provinces)."""
    all_p = sorted(set(survey_prov) |
                   {p for cnt in cluster_prov.values() for p in cnt})
    pidx = {p: i for i, p in enumerate(all_p)}

    # Cluster distributions
    n = len(cluster_ids)
    X = np.zeros((n, len(all_p)))
    for i, c in enumerate(cluster_ids):
        cnt = cluster_prov.get(int(c), Counter())
        t = sum(cnt.values()) or 1
        for p, v in cnt.items():
            X[i, pidx[p]] = v / t

    # Target (survey) distribution
    st = sum(survey_prov.values()) or 1
    y = np.array([survey_prov.get(p, 0) / st for p in all_p])

    # Cost matrix: each cluster vs a "synthetic respondent" that's the survey itself.
    # Use Wasserstein-style squared L2 between distributions.
    # For richer OT, treat each cluster as source and each survey respondent as sink,
    # but we don't have per-respondent data here cheaply; the below uses the survey
    # as a single target, yielding cluster weight ∝ exp(-cost/τ), which coincides
    # with entropic OT for n_sink=1. Use L1 (equal to TV) as metric.
    cost = np.sum(np.abs(X - y[None, :]), axis=1)  # n
    # Entropic weights
    w = np.exp(-cost / (epsilon + 1e-12))
    if w.sum() > 0:
        w = w / w.mean()  # normalize to mean 1
    return {str(c): float(w[i]) for i, c in enumerate(cluster_ids)}


def ot_sinkhorn_full(cluster_prov, survey_respondent_provs, cluster_ids,
                     epsilon=0.05, n_iter=100):
    """True entropic OT via Sinkhorn between clusters (as province dists) and
    each survey respondent (as one-hot province). Returns per-cluster marginal
    mass as weight."""
    all_p = sorted(set(survey_respondent_provs) |
                   {p for cnt in cluster_prov.values() for p in cnt})
    pidx = {p: i for i, p in enumerate(all_p)}

    n = len(cluster_ids)
    # Cluster distributions
    X = np.zeros((n, len(all_p)))
    for i, c in enumerate(cluster_ids):
        cnt = cluster_prov.get(int(c), Counter())
        t = sum(cnt.values()) or 1
        for p, v in cnt.items():
            X[i, pidx[p]] = v / t

    # Sample survey respondents to cap size
    rng = np.random.RandomState(42)
    n_samp = min(500, len(survey_respondent_provs))
    samp = rng.choice(len(survey_respondent_provs), size=n_samp, replace=False)
    resps = [survey_respondent_provs[i] for i in samp]

    m = len(resps)
    Y = np.zeros((m, len(all_p)))
    for j, p in enumerate(resps):
        if p in pidx:
            Y[j, pidx[p]] = 1.0

    # Cost: L1 between cluster dist and respondent one-hot
    C = np.sum(np.abs(X[:, None, :] - Y[None, :, :]), axis=2)  # n × m
    a = np.ones(n) / n
    b = np.ones(m) / m

    K = np.exp(-C / (epsilon + 1e-12))
    u = np.ones(n)
    v = np.ones(m)
    for _ in range(n_iter):
        u = a / (K @ v + 1e-12)
        v = b / (K.T @ u + 1e-12)
    Gamma = u[:, None] * K * v[None, :]
    mass = Gamma.sum(axis=1)  # n
    if mass.mean() > 0:
        mass = mass / mass.mean()
    return {str(c): float(mass[i]) for i, c in enumerate(cluster_ids)}


# ─── Evaluation ───

def apply_scheme(base_w, demo_w):
    """Multiply base cluster_weights by demographic multiplier, renormalize."""
    if demo_w is None:
        return base_w
    out = {c: base_w.get(c, 0) * demo_w.get(c, 1.0) for c in base_w}
    return out


def eval_methods(questions, per_q_pmfs, top_cids, weight_schemes):
    """per_q_pmfs: {method_name: {qi: {cid: pmf}}}.
    Returns {method_name × scheme_name: list of per-q JS}."""
    results = {}
    for mname in per_q_pmfs:
        for sname, swdemo in weight_schemes.items():
            js_vals = []
            for qi, qd in enumerate(questions):
                cpmfs = per_q_pmfs[mname][qi]
                base_w = qd["cluster_weights"]
                final_w = apply_scheme(base_w, swdemo)
                pred = agg(cpmfs, final_w, top_cids)
                js_vals.append(js_score(qd["true_distribution"], pred))
            results[f"{mname}||{sname}"] = js_vals
    return results


# ─── Main ───

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--skip_age", action="store_true",
                   help="Skip LLM age inference (saves ~20 min)")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--n_posts", type=int, default=50)
    p.add_argument("--n_llm_samples", type=int, default=3)
    p.add_argument("--n_steer_resp", type=int, default=5)
    p.add_argument("--output", default=f"{RESULTS_DIR}/alignment_6q.json")
    args = p.parse_args()

    log(f"Loading data …")
    questions, topics, df = load_orig6()
    top_cids = get_top_clusters(questions, n=args.top_k)
    log(f"Top {args.top_k} clusters: {top_cids}")

    log("Loading model + encoder …")
    encoder = SentenceTransformer(EMBED_PATH)
    model, tokenizer = load_model(MODEL_PATH, args.device)
    pvecs = load_persona_vectors(f"{RESULTS_DIR}/persona_vectors_L24_N20.npz")
    pvecs_16 = load_persona_vectors(f"{RESULTS_DIR}/persona_vectors_L16_N20.npz")
    pvecs_20 = load_persona_vectors(f"{RESULTS_DIR}/persona_vectors_L20_N20.npz")
    # merge for multi-layer (use per-layer; same keys)
    merged_pvecs = pvecs

    # ─── Anchors ───
    log("Generating anchors …")
    anchors = {}
    for qd in questions:
        anc = generate_anchors_local(model, tokenizer, qd["question"], qd["options"])
        anchors[qd["key"]] = [encoder.encode(a) for a in anc]
        log(f"  {qd['key']}: {anc[0][:40]}…")

    # ─── Per-cluster distributions ───
    per_q = {"SSR": {}, "LLM": {}, "MLsteer": {}}

    log("Computing SSR per-cluster distributions …")
    for qi, qd in enumerate(questions):
        per_q["SSR"][qi] = compute_ssr_pmfs(encoder, qd, top_cids, df,
                                             anchors[qd["key"]],
                                             n_posts=args.n_posts)

    log("Computing LLM-dist per-cluster distributions …")
    for qi, qd in enumerate(questions):
        per_q["LLM"][qi] = compute_llmdist_pmfs(model, tokenizer, qd, top_cids,
                                                  df, topics, n_samples=args.n_llm_samples)
        log(f"  Q{qi+1}/{len(questions)} done")

    log("Computing ML-steer per-cluster distributions …")
    for qi, qd in enumerate(questions):
        per_q["MLsteer"][qi] = compute_mlsteer_pmfs(model, tokenizer, encoder,
                                                      qd, top_cids, merged_pvecs,
                                                      anchors[qd["key"]],
                                                      alpha=0.1, layers=(16, 20, 24),
                                                      n_resp=args.n_steer_resp)
        log(f"  Q{qi+1}/{len(questions)} done")

    # ─── Ensemble methods ───
    # Build M0 (SSR), M1 (LLM), M2 (SSR+LLM w=0.5), M4 (SSR+MLsteer w=0.5)
    method_pmfs = {}
    method_pmfs["M0_SSR"] = per_q["SSR"]
    method_pmfs["M1_LLM"] = per_q["LLM"]
    m2 = {qi: {} for qi in range(len(questions))}
    m4 = {qi: {} for qi in range(len(questions))}
    m24 = {qi: {} for qi in range(len(questions))}  # triple ensemble
    for qi in range(len(questions)):
        ssr, llm, ml = per_q["SSR"][qi], per_q["LLM"][qi], per_q["MLsteer"][qi]
        for c in top_cids:
            if c in ssr and c in llm:
                m2[qi][c] = 0.5 * ssr[c] + 0.5 * llm[c]
            if c in ssr and c in ml:
                m4[qi][c] = 0.5 * ssr[c] + 0.5 * ml[c]
            if c in ssr and c in llm and c in ml:
                m24[qi][c] = (1/3) * (ssr[c] + llm[c] + ml[c])
    method_pmfs["M2_SSR+LLM_w0.5"] = m2
    method_pmfs["M4_SSR+MLsteer_w0.5"] = m4
    method_pmfs["M2+M4_triple"] = m24  # bonus: SSR + LLM + ML triple mix

    # ─── Demographic data ───
    log("Loading survey demographics …")
    survey_prov, survey_age = load_survey_province_age()
    log(f"  Survey: {sum(survey_prov.values())} prov responses, "
        f"{sum(survey_age.values())} age responses")

    log("Computing cluster province histograms …")
    cluster_prov = compute_cluster_prov_hist(df)
    sm_prov = Counter()
    for c in cluster_prov:
        for p, v in cluster_prov[c].items():
            sm_prov[p] += v

    # Subset to top clusters for demographic stats
    cluster_prov_top = {int(c): cluster_prov.get(int(c), Counter())
                         for c in top_cids}

    # ─── Weight schemes ───
    log("Computing weight schemes …")
    weight_schemes = {"default": None}

    # IS on province
    is_prov = is_weights_prov(cluster_prov_top, sm_prov, survey_prov)
    weight_schemes["IS_prov"] = is_prov
    log(f"  IS_prov weights: "
        f"{ {c: f'{is_prov[c]:.3f}' for c in list(is_prov)[:5]} } …")

    # OT on province (fast: L1 vs survey dist)
    ot_prov_fast = ot_weights_prov(cluster_prov_top, survey_prov, top_cids,
                                     epsilon=0.1)
    weight_schemes["OT_prov_fast"] = ot_prov_fast

    # OT true Sinkhorn
    survey_df = pd.read_csv(SURVEY_CSV, low_memory=False)
    resp_provs = []
    for _, row in survey_df.iterrows():
        try:
            content = json.loads(row['content'])
            for item in content.get('list', []):
                if '常驻地' in item.get('question', '') or '常住地' in item.get('question', ''):
                    for ans in item.get('answers', []):
                        if ans.get('selected'):
                            resp_provs.append(ans['text'])
                            break
        except Exception:
            pass
    ot_prov_full = ot_sinkhorn_full(cluster_prov_top, resp_provs, top_cids,
                                      epsilon=0.1, n_iter=100)
    weight_schemes["OT_prov_sinkhorn"] = ot_prov_full

    # Age (optional, slow)
    cluster_age = {}
    is_age = None
    is_prov_age = None
    if not args.skip_age:
        log("Inferring cluster age distributions via LLM (slow) …")
        cluster_age = infer_cluster_age_dists(model, tokenizer, df, top_cids,
                                                n_sample=12)
        log(f"  Got age dists for {len(cluster_age)} clusters")
        if cluster_age:
            is_age = is_weights_age(cluster_age, survey_age)
            weight_schemes["IS_age"] = is_age
            # Combined prov + age
            is_prov_age = {c: is_prov.get(c, 1.0) * is_age.get(c, 1.0)
                           for c in set(is_prov) | set(is_age)}
            m = np.mean(list(is_prov_age.values())) or 1.0
            is_prov_age = {k: v / m for k, v in is_prov_age.items()}
            weight_schemes["IS_prov+age"] = is_prov_age

    # ─── Adaptive-n experiment (on M1 and SSR components) ───
    log("Adaptive-n experiment …")
    adaptive_results = {}
    for n in [30, 50, 80, 120, 160]:
        per_q_ssr_n = {}
        for qi, qd in enumerate(questions):
            per_q_ssr_n[qi] = compute_ssr_pmfs(encoder, qd, top_cids, df,
                                                 anchors[qd["key"]], n_posts=n)
        js_vals = []
        for qi, qd in enumerate(questions):
            pred = agg(per_q_ssr_n[qi], qd["cluster_weights"], top_cids)
            js_vals.append(js_score(qd["true_distribution"], pred))
        adaptive_results[f"SSR_n={n}"] = js_vals
        log(f"  SSR n={n}: mean JS = {np.mean(js_vals):.4f}")

    # Variance-based adaptive: measure per-cluster SSR variance, sample more where high
    log("Variance-based adaptive-n on SSR …")
    adaptive_ssr_var = {qi: {} for qi in range(len(questions))}
    for qi, qd in enumerate(questions):
        for cid in top_cids:
            posts = df[df["cluster_label"] == int(cid)]["content_desc"].dropna()
            if len(posts) < 3:
                continue
            # initial 30
            samp = posts.sample(min(30, len(posts)), random_state=42).tolist()
            pmfs = [ssr_score(pt, encoder, anchors[qd["key"]])[0] for pt in samp]
            var = np.mean(np.var(pmfs, axis=0))
            # If high variance, sample more
            if var > 0.02 and len(posts) > 80:
                extra = posts.sample(min(80, len(posts)) - 30,
                                      random_state=43).tolist()
                pmfs += [ssr_score(pt, encoder, anchors[qd["key"]])[0] for pt in extra]
            adaptive_ssr_var[qi][cid] = np.mean(pmfs, axis=0)
    js_vals = []
    for qi, qd in enumerate(questions):
        pred = agg(adaptive_ssr_var[qi], qd["cluster_weights"], top_cids)
        js_vals.append(js_score(qd["true_distribution"], pred))
    adaptive_results["SSR_adaptive_var"] = js_vals
    log(f"  SSR adaptive-var: mean JS = {np.mean(js_vals):.4f}")

    # ─── Run all combinations ───
    log("Evaluating methods × weight schemes …")
    all_results = eval_methods(questions, method_pmfs, top_cids, weight_schemes)

    # ─── Summary ───
    log("\n" + "=" * 70)
    log("RESULTS — Mean JS (6Q). Baseline (pure SSR default) = 0.0277.")
    log("=" * 70)
    header = f"  {'Method':<28} {'Scheme':<22} {'Mean JS':>8} {'Δ vs SSR':>10}"
    log(header)
    log("  " + "-" * 68)
    baseline_js = 0.0277
    # Print sorted by mean JS
    summary_rows = []
    for tag, vals in all_results.items():
        mj = float(np.mean(vals))
        mname, sname = tag.split("||")
        summary_rows.append((mname, sname, mj, vals))
    summary_rows.sort(key=lambda r: r[2])
    for mname, sname, mj, _ in summary_rows:
        delta = baseline_js - mj
        star = " ★ BEATS" if mj < baseline_js else ""
        log(f"  {mname:<28} {sname:<22} {mj:>8.4f} {delta:>+10.4f}{star}")

    log("\n" + "-" * 70)
    log("Adaptive-n results (SSR only):")
    for tag, vals in adaptive_results.items():
        mj = float(np.mean(vals))
        log(f"  {tag:<28} {mj:>8.4f}")

    # ─── Save ───
    output = {
        "methods_x_schemes": {k: [float(v) for v in vals]
                               for k, vals in all_results.items()},
        "adaptive_n": {k: [float(v) for v in vals]
                        for k, vals in adaptive_results.items()},
        "weight_schemes_values": {
            k: ({c: float(v) for c, v in w.items()} if w is not None else None)
            for k, w in weight_schemes.items()
        },
        "cluster_age_inferred": {str(k): dict(v) for k, v in cluster_age.items()},
        "config": vars(args),
        "top_clusters": [str(c) for c in top_cids],
        "baseline_js": baseline_js,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    log(f"\nSaved → {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        log(f"ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
