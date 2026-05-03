"""
PS-SSR Pipeline — single-file end-to-end implementation.

Goal: take new questionnaire data (and optionally a new social-media corpus)
and produce predicted survey distributions via the validated M0 / M1 / M2
methods at the same hyperparameters used in the original 6Q paper benchmark.

Stages provided
---------------
  1. cluster_posts(...)            — embed (bge-small-zh) → UMAP → HDBSCAN → drop noise
  2. generate_cluster_topics(...)  — one-sentence Chinese topic per cluster (LLM)
  3. generate_anchors(...)         — per-question option anchor sentences (LLM)
  4. compute_cluster_weights(...)  — per-question soft cluster relevance
                                     (LLM "is_relevant?" gate × topic-SSR similarity)
  5. compute_m0_pmfs(...)          — per-cluster, per-post SSR mean → cluster PMF
  6. compute_m1_pmfs(...)          — per-cluster LLM-distribution prompt → cluster PMF
  7. aggregate(...)                — cluster-mass-weighted sum over top-K clusters
  8. run_question / run_questionnaire — convenience wrappers for end-to-end predictions
  9. evaluate(...)                 — JS / K_xy / C_xy

Library usage
-------------
    from pssr_pipeline import PSSRPipeline

    pipe = PSSRPipeline(
        ssr_embed_path   = "/data/.../bge-base-zh-v1.5",
        llm_model_path   = "/data/.../Qwen3-8B",
        device           = "cuda:0",
        cluster_embed_path = "BAAI/bge-small-zh-v1.5",  # only needed if clustering
    )

    # If you already have clusters + topics + cluster_weights, skip ahead:
    results = pipe.run_questionnaire(
        questions=questions,           # list[dict]: {key, question, options,
                                       #             true_distribution, cluster_weights}
        posts_df=posts_df,             # pandas DF with content_desc + cluster_label
        cluster_topics=cluster_topics, # {cid_str: topic_str}
        methods=("m0", "m1", "m2"),
        top_k=10, n_posts=50, n_samples=3, ensemble_w=0.5, seed=42,
    )

    # If clusters are not yet computed:
    df_clustered = pipe.cluster_posts(raw_posts_df)
    cluster_topics = pipe.generate_cluster_topics(df_clustered)
    # Then compute cluster_weights per question (if not already provided):
    for q in questions:
        q["cluster_weights"] = pipe.compute_cluster_weights(
            q["question"], q["options"], cluster_topics)

CLI usage
---------
    python pssr_pipeline.py cluster  --posts raw.csv  --output clustered.csv
    python pssr_pipeline.py topics   --posts clustered.csv  --output topics.json
    python pssr_pipeline.py predict  --questions q.json  --posts clustered.csv \\
                                     --topics topics.json  --output predictions.json

The `predict` command auto-fills `cluster_weights` if missing in the questions JSON.
"""

import argparse
import json
import os
import random
import re
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import jensenshannon
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


# ══════════════════════════════════════════════════════════════════════
# Default paths (overridable via __init__ or CLI)
# ══════════════════════════════════════════════════════════════════════
DEFAULT_SSR_EMBED = "/data/chenhongrui/models/bge-base-zh-v1.5"
DEFAULT_CLUSTER_EMBED = "BAAI/bge-small-zh-v1.5"
DEFAULT_LLM = "/data/chenhongrui/models/Qwen3-8B"

# ══════════════════════════════════════════════════════════════════════
# Prompts (verified against survey_ssr_exp.ipynb cell 26 + dump_*.py)
# ══════════════════════════════════════════════════════════════════════
TOPIC_PROMPT = (
    "你是一名用户调研分析专家。\n\n"
    "我会给你一组来自同一个语义簇（cluster）的用户评论。\n"
    "请你用一句话描述该 cluster 的主题（基于整体内容，而不是个别评论）。\n\n"
    "请严格按以下 JSON 格式输出（注意字段和值都必须一致）：\n\n"
    '{{\n  "cluster_topic": "<一句话主题描述>"\n}}\n\n'
    "下面是该 cluster 的评论（每条之间用换行分隔）：\n{CLUSTER_COMMENTS}\n\n"
    "请严格按以下 JSON 格式输出：\n"
    '{{\n  "cluster_topic": "<一句话主题描述>"\n}}'
)

ANCHOR_PROMPT = (
    "你是一名专业的问卷语义建模专家。\n\n"
    "我会给你一个问卷题目和该题目的所有选项。\n"
    "你的任务是为每个选项生成一个 \"语义锚点句子（anchor sentence）\"。\n\n"
    "要求：\n"
    "1. 每个锚点句子必须是自然语言表达，而不是重复选项文本本身。\n"
    "2. 每个锚点必须清晰表达该选项所代表的态度或立场。\n"
    "3. 锚点句子要尽可能具体、生动、带情绪或明确倾向。\n"
    "4. 输出顺序必须与选项顺序完全一致。\n"
    "5. 锚点句子越像真实用户说的话越好。\n\n"
    "请严格按以下 JSON 格式输出：\n"
    '{{\n  "anchors": [\n    "<对应选项1 的 anchor sentence>",\n'
    '    "<对应选项2 的 anchor sentence>",\n    ...\n  ]\n}}\n\n'
    "下面是问卷题目：\n{QUESTION}\n\n"
    "下面是该题目的选项（按顺序）：\n{OPTIONS}"
)

RELEVANCE_PROMPT = (
    "你是一名用户调研分析专家。\n\n"
    "我会给你一组来自同一个语义簇（cluster）的主题，以及一个问卷题目。\n"
    "请你判断该 cluster 的主题是否与该问卷题目所测量的内容相关。\n\n"
    "请严格按以下 JSON 格式输出：\n"
    '{{\n  "is_relevant": "相关 或 不相关"\n}}\n\n'
    "判断原则：\n"
    "- 只要评论内容能表达对问卷题目所测维度的看法、情绪、态度或体验，即视为\"相关\"。\n"
    "- 如果评论是闲聊、吐槽客服、物流、活动、广告、无意义文本、表情或与产品评价无关，则视为\"不相关\"。\n"
    "- 判断基于该 cluster 的整体趋势，而不是个别评论。\n\n"
    "下面是问卷题目：\n{QUESTION}\n\n"
    "下面是问卷题目可供选择的答案：\n{OPTIONS}\n\n"
    "下面是该 cluster 的主题：\n{CLUSTER_TOPIC}\n\n"
    "请严格按以下 JSON 格式输出：\n"
    '{{\n  "is_relevant": "相关 或 不相关"\n}}'
)

M1_PROMPT = (
    "你是一位消费者调研分析专家。\n\n"
    "以下是某类消费者群体的社交媒体帖子样本（话题：{TOPIC}）：\n{POSTS}\n\n"
    "基于这些帖子反映的消费者特征和态度，"
    "请估计这个群体在以下问卷问题中各选项的选择比例。\n\n"
    "问题：{QUESTION}\n选项：\n{OPTIONS_NUMBERED}\n\n"
    "请严格按JSON格式输出各选项的比例（总和为1）：\n"
    '{{"distribution": [选项1比例, 选项2比例, ...]}}'
)


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════
def _set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _cosine(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def _strip_thinking(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


# ─── Hardened JSON-distribution parser (4-strategy cascade) ─────────────
def _normalize_arr(arr, n_opts):
    arr = np.clip(np.array(arr, float), 0, None)
    if len(arr) != n_opts or arr.sum() <= 0:
        return None
    return arr / arr.sum()


def parse_distribution(resp: str, n_opts: int):
    """Returns (pmf, strategy) or (None, 'fail').
    Used for parsing M1's per-cluster JSON output."""
    text = _strip_thinking(resp)
    # S1: full JSON object containing "distribution"
    for m in re.finditer(r'\{[^{}]*"distribution"[^{}]*\}', text, flags=re.DOTALL):
        try:
            arr = json.loads(m.group(0)).get("distribution")
            if arr is not None:
                pmf = _normalize_arr(arr, n_opts)
                if pmf is not None:
                    return pmf, "s1_json_obj"
        except Exception:
            continue
    # S2: bare [a, b, c, ...] with exact n_opts numerics
    for m in re.finditer(r'\[([^\[\]]+)\]', text):
        nums = re.findall(r'-?\d+(?:\.\d+)?', m.group(1))
        if len(nums) == n_opts:
            pmf = _normalize_arr([float(x) for x in nums], n_opts)
            if pmf is not None:
                return pmf, "s2_bracket_exact"
    # S3: last n_opts numeric tokens anywhere
    nums = re.findall(r'-?\d*\.\d+|-?\d+', text)
    if len(nums) >= n_opts:
        pmf = _normalize_arr([float(x) for x in nums[-n_opts:]], n_opts)
        if pmf is not None:
            return pmf, "s3_last_n_nums"
    # S4: percent-style "选项1: 30%" lines
    pct = re.findall(r'(?:选项)?\s*(\d+)[.:：]\s*(\d+(?:\.\d+)?)\s*%', text)
    if pct:
        vals = [0.0] * n_opts
        for idx_str, p in pct:
            try:
                idx = int(idx_str) - 1
                if 0 <= idx < n_opts:
                    vals[idx] = float(p)
            except Exception:
                continue
        pmf = _normalize_arr(vals, n_opts)
        if pmf is not None:
            return pmf, "s4_percent_lines"
    return None, "fail"


# ─── SSR core ────────────────────────────────────────────────────────────
def ssr_score(text_or_emb, encoder, anchor_embeddings, temp: float = 0.1,
              is_embedding: bool = False) -> np.ndarray:
    """Temp-softmax SSR over option anchors. Returns PMF.

    text_or_emb: str → encoded by `encoder`; or pre-computed np.ndarray with is_embedding=True.
    """
    v = text_or_emb if is_embedding else encoder.encode(text_or_emb)
    sims = np.array([_cosine(v, a) for a in anchor_embeddings], dtype=float)
    exp = np.exp((sims - sims.max()) / temp)
    return exp / exp.sum()


# ─── Metrics ─────────────────────────────────────────────────────────────
def _safe_norm(t):
    t = np.array(t, float)
    return t / (t.sum() + 1e-10)


def js_divergence(true, pred) -> float:
    return float(jensenshannon(_safe_norm(true), _safe_norm(pred)) ** 2)


def k_xy(true, pred) -> float:
    return 1 - float(np.max(np.abs(np.cumsum(_safe_norm(true)) - np.cumsum(_safe_norm(pred)))))


def c_xy(true, pred) -> float:
    a, b = _safe_norm(true), _safe_norm(pred)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def evaluate(pred, true) -> dict:
    return {"js": js_divergence(true, pred),
            "k_xy": k_xy(true, pred),
            "c_xy": c_xy(true, pred)}


# ══════════════════════════════════════════════════════════════════════
# Pipeline class
# ══════════════════════════════════════════════════════════════════════
class PSSRPipeline:
    """End-to-end PS-SSR (M0 / M1 / M2) pipeline.

    Holds the LLM (Qwen3-8B) and the SSR sentence-transformer (bge-base-zh).
    A separate cluster-embedder (bge-small-zh) is loaded lazily only if
    `cluster_posts` is called — for downstream prediction on pre-clustered
    data, it is not needed.
    """

    def __init__(
        self,
        ssr_embed_path: str = DEFAULT_SSR_EMBED,
        llm_model_path: str = DEFAULT_LLM,
        device: str = "cuda:0",
        cluster_embed_path: str = DEFAULT_CLUSTER_EMBED,
    ):
        self.device = device
        self._cluster_embed_path = cluster_embed_path
        self._cluster_encoder = None  # lazy
        _log(f"Loading SSR encoder ({ssr_embed_path}) on {device} …")
        self.encoder = SentenceTransformer(ssr_embed_path, device=device)
        _log(f"Loading LLM ({llm_model_path}) on {device} …")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map=device,
        )
        self.model.eval()
        _log("Pipeline ready.")

    # ─── Lazy: cluster encoder ──────────────────────────────────────────
    def _get_cluster_encoder(self):
        if self._cluster_encoder is None:
            _log(f"Loading cluster encoder ({self._cluster_embed_path}) …")
            self._cluster_encoder = SentenceTransformer(
                self._cluster_embed_path, device=self.device)
        return self._cluster_encoder

    # ─── LLM generate helper ────────────────────────────────────────────
    def _llm_generate(self, prompt: str, max_new_tokens: int = 200,
                      temperature: float = 0.7, top_p: float = 0.9,
                      do_sample: bool = True) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=do_sample, temperature=temperature, top_p=top_p,
            )
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True)

    # ════════════════════════════════════════════════════════════════════
    # Stage 1: Clustering
    # ════════════════════════════════════════════════════════════════════
    def cluster_posts(
        self,
        posts_df: pd.DataFrame,
        content_col: str = "content_desc",
        umap_dim: int = 50,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.0,
        hdbscan_min_cluster_size: int = 50,
        hdbscan_min_samples: int = 50,
        hdbscan_epsilon: float = 0.1,
        seed: int = 42,
        drop_noise: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Embed → UMAP → HDBSCAN. Adds 'cluster_label'; drops noise (-1) by default."""
        from sklearn.cluster import HDBSCAN
        from umap import UMAP

        encoder = self._get_cluster_encoder()
        df = posts_df.copy()
        df[content_col] = df[content_col].astype(str)
        comments = df[content_col].tolist()
        if verbose:
            _log(f"Embedding {len(comments):,} posts with cluster encoder …")
        embs = []
        bs = 100
        for i in range(0, len(comments), bs):
            embs.extend(encoder.encode(comments[i:i+bs], show_progress_bar=False))
        embs = np.array(embs)

        if verbose:
            _log(f"UMAP → {umap_dim} dims …")
        reducer = UMAP(n_components=umap_dim, random_state=seed,
                        n_neighbors=umap_n_neighbors, min_dist=umap_min_dist,
                        verbose=False, n_jobs=-1)
        embs_red = reducer.fit_transform(embs)

        if verbose:
            _log(f"HDBSCAN (min_cluster_size={hdbscan_min_cluster_size}) …")
        clusterer = HDBSCAN(
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
            cluster_selection_epsilon=hdbscan_epsilon,
            metric="euclidean", cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(embs_red)
        df["cluster_label"] = labels
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        if verbose:
            _log(f"  {n_clusters} clusters, {n_noise} noise points "
                 f"({100*n_noise/len(labels):.1f}%)")
        if drop_noise:
            df = df[df["cluster_label"] != -1].reset_index(drop=True)
        return df

    # ════════════════════════════════════════════════════════════════════
    # Stage 2: Per-cluster topic generation
    # ════════════════════════════════════════════════════════════════════
    def generate_cluster_topics(
        self,
        posts_df: pd.DataFrame,
        cluster_col: str = "cluster_label",
        content_col: str = "content_desc",
        n_per_cluster: int = 20,
        temperature: float = 0.0,
    ) -> dict[str, str]:
        """One-sentence Chinese topic per cluster via Qwen3-8B."""
        topics = {}
        cids = sorted(set(posts_df[cluster_col].unique()))
        for cid in cids:
            comments = posts_df[posts_df[cluster_col] == cid][content_col].dropna().tolist()
            if not comments:
                continue
            sample = comments[:n_per_cluster]
            prompt = TOPIC_PROMPT.format(CLUSTER_COMMENTS="\n".join(sample))
            resp = self._llm_generate(
                prompt, max_new_tokens=200,
                temperature=max(temperature, 1e-3),
                do_sample=temperature > 0,
            )
            resp = _strip_thinking(resp)
            topic = None
            m = re.search(r'\{[^{}]*"cluster_topic"[^{}]*\}', resp, flags=re.DOTALL)
            if m:
                try:
                    topic = json.loads(m.group(0)).get("cluster_topic")
                except Exception:
                    pass
            if topic is None:
                topic = resp[:80]  # fallback: first 80 chars of response
            topics[str(cid)] = topic
            _log(f"  cluster {cid}: {topic[:50]}…")
        return topics

    # ════════════════════════════════════════════════════════════════════
    # Stage 3: Anchor generation per question
    # ════════════════════════════════════════════════════════════════════
    def generate_anchors(self, question: str, options: list[str]) -> list[str]:
        """Generate anchor sentences for one question's options. Deterministic."""
        prompt = ANCHOR_PROMPT.format(QUESTION=question, OPTIONS="、".join(options))
        resp = self._llm_generate(prompt, max_new_tokens=500, do_sample=False)
        resp = _strip_thinking(resp)
        # try {"anchors": [...]}
        m = re.search(r'\{"anchors"\s*:\s*\[.*?\]\s*\}', resp, re.DOTALL)
        if m:
            try:
                arr = json.loads(m.group()).get("anchors")
                if isinstance(arr, list) and len(arr) == len(options):
                    return arr
            except Exception:
                pass
        # try any [..., ...] of right length
        m = re.search(r'\[(.*?)\]', resp, re.DOTALL)
        if m:
            try:
                arr = json.loads("[" + m.group(1) + "]")
                if isinstance(arr, list) and len(arr) == len(options):
                    return arr
            except Exception:
                pass
        # fallback
        _log(f"  Anchor parse failed for '{question[:30]}…'; using template fallback")
        return [f"关于{question}，我认为答案是{opt}" for opt in options]

    # ════════════════════════════════════════════════════════════════════
    # Stage 4: Cluster weights (per question, soft-relevance × topic-SSR)
    # ════════════════════════════════════════════════════════════════════
    def compute_cluster_weights(
        self,
        question: str,
        options: list[str],
        cluster_topics: dict[str, str],
        eps: float = 0.1,
    ) -> dict[str, float]:
        """For each cluster: w_c = topic_SSR(question, topic_c) × (1 if relevant else eps).

        Matches the original notebook cell 31 derivation (Step C / Step D).
        """
        cids = list(cluster_topics.keys())
        topic_embs = [self.encoder.encode(cluster_topics[c]) for c in cids]
        # SSR(question_text → topic_embeddings) gives a PMF over clusters
        topic_pmf = ssr_score(question, self.encoder, topic_embs)
        # LLM relevance gate
        weights = {}
        for idx, c in enumerate(cids):
            prompt = RELEVANCE_PROMPT.format(
                QUESTION=question, OPTIONS="、".join(options),
                CLUSTER_TOPIC=cluster_topics[c])
            resp = _strip_thinking(
                self._llm_generate(prompt, max_new_tokens=80, do_sample=False))
            relevant = "相关" in resp and "不相关" not in resp
            w = topic_pmf[idx] * (1.0 if relevant else eps)
            weights[c] = float(w)
        # Normalize so they sum to 1
        total = sum(weights.values()) + 1e-10
        return {c: w / total for c, w in weights.items()}

    # ════════════════════════════════════════════════════════════════════
    # Stage 5–6: M0 / M1 per-cluster PMFs
    # ════════════════════════════════════════════════════════════════════
    def compute_m0_pmfs(
        self,
        question_anchor_embs: list,
        cids: list[str],
        posts_df: pd.DataFrame,
        cluster_col: str = "cluster_label",
        content_col: str = "content_desc",
        n_posts: int = 50,
        sample_seed: int = 42,
    ) -> dict[str, np.ndarray]:
        """M0 = SSR per post → mean within cluster (per-post SSR)."""
        out = {}
        for cid in cids:
            posts = posts_df[posts_df[cluster_col] == int(cid)][content_col].dropna()
            if len(posts) < 3:
                continue
            samp = posts.sample(min(n_posts, len(posts)), random_state=sample_seed).tolist()
            pmfs = [ssr_score(p, self.encoder, question_anchor_embs) for p in samp]
            out[cid] = np.mean(pmfs, axis=0)
        return out

    def compute_m1_pmfs(
        self,
        question: str,
        options: list[str],
        cids: list[str],
        posts_df: pd.DataFrame,
        cluster_topics: dict[str, str],
        cluster_col: str = "cluster_label",
        content_col: str = "content_desc",
        n_ex: int = 8,
        n_samples: int = 3,
        sample_seed: int = 42,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        """M1 = LLM emits one PMF per cluster (per-cluster prompting).
        Returns (cluster→pmf, parse-strategy-counts).
        """
        n_opts = len(options)
        opts_str = "\n".join(f"{i+1}. {o}" for i, o in enumerate(options))
        out = {}
        strats = {"s1_json_obj": 0, "s2_bracket_exact": 0,
                   "s3_last_n_nums": 0, "s4_percent_lines": 0, "fail": 0}
        for cid in cids:
            posts = posts_df[posts_df[cluster_col] == int(cid)][content_col].dropna()
            if len(posts) < 3:
                continue
            samp = posts.sample(min(n_ex, len(posts)), random_state=sample_seed).tolist()
            post_text = "\n".join(f"- {p[:100]}" for p in samp)
            topic = cluster_topics.get(cid, "未知")[:60]
            prompt = M1_PROMPT.format(
                TOPIC=topic, POSTS=post_text,
                QUESTION=question, OPTIONS_NUMBERED=opts_str)
            pmfs = []
            for _ in range(n_samples):
                resp = self._llm_generate(
                    prompt, max_new_tokens=max_new_tokens,
                    temperature=temperature, top_p=top_p, do_sample=True)
                pmf, strat = parse_distribution(resp, n_opts)
                strats[strat] += 1
                if pmf is None:
                    pmf = np.ones(n_opts) / n_opts
                pmfs.append(pmf)
            out[cid] = np.mean(pmfs, axis=0)
        return out, strats

    # ════════════════════════════════════════════════════════════════════
    # Stage 7: Aggregate
    # ════════════════════════════════════════════════════════════════════
    @staticmethod
    def aggregate(per_cluster_pmfs: dict[str, np.ndarray],
                   cluster_weights: dict[str, float],
                   cids: list[str]) -> np.ndarray:
        """Cluster-mass-weighted sum over a chosen cluster list."""
        if not per_cluster_pmfs:
            raise ValueError("per_cluster_pmfs is empty")
        n = len(next(iter(per_cluster_pmfs.values())))
        acc, tw = np.zeros(n), 0.0
        for c in cids:
            if c in per_cluster_pmfs:
                w = cluster_weights.get(c, 0.0)
                acc += w * per_cluster_pmfs[c]
                tw += w
        return acc / tw if tw > 0 else acc

    # ════════════════════════════════════════════════════════════════════
    # Stage 8: end-to-end on one question / a full questionnaire
    # ════════════════════════════════════════════════════════════════════
    def get_top_clusters(self, questions: list[dict], top_k: int = 10) -> list[str]:
        """Pick the top-K clusters by aggregated cluster_weights across all questions.
        Mirrors `get_top_clusters` in dump_*.py.
        """
        agg = {}
        for q in questions:
            for cid, w in q.get("cluster_weights", {}).items():
                agg[cid] = agg.get(cid, 0) + w
        return sorted(agg, key=lambda c: -agg[c])[:top_k]

    def run_question(
        self,
        question: dict,
        posts_df: pd.DataFrame,
        cluster_topics: dict[str, str],
        top_cids: list[str],
        anchors: list[str] | None = None,
        methods: tuple[str, ...] = ("m0", "m1", "m2"),
        n_posts: int = 50,
        n_samples: int = 3,
        ensemble_w: float = 0.5,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Run M0/M1/M2 on one question. `question` must contain:
            - 'question' (str), 'options' (list[str]),
            - 'cluster_weights' (dict[cid_str, float]),
            - 'true_distribution' (optional, for metrics).
        """
        if anchors is None:
            anchors = self.generate_anchors(question["question"], question["options"])
        anchor_embs = [self.encoder.encode(a) for a in anchors]

        out: dict[str, Any] = {
            "question": question.get("question"),
            "options": question["options"],
            "anchors": anchors,
        }

        ssr_p = self.compute_m0_pmfs(
            anchor_embs, top_cids, posts_df,
            n_posts=n_posts, sample_seed=seed,
        ) if "m0" in methods or "m2" in methods else {}

        llm_p, m1_strats = ({}, None)
        if "m1" in methods or "m2" in methods:
            llm_p, m1_strats = self.compute_m1_pmfs(
                question["question"], question["options"],
                top_cids, posts_df, cluster_topics,
                n_samples=n_samples, sample_seed=seed,
            )

        cw = question["cluster_weights"]
        if "m0" in methods:
            out["pred_m0"] = self.aggregate(ssr_p, cw, top_cids).tolist()
        if "m1" in methods:
            out["pred_m1"] = self.aggregate(llm_p, cw, top_cids).tolist()
            out["m1_parse_strats"] = m1_strats
        if "m2" in methods:
            m2_per_cluster = {
                c: ensemble_w * ssr_p[c] + (1 - ensemble_w) * llm_p[c]
                for c in top_cids if c in ssr_p and c in llm_p
            }
            out["pred_m2"] = self.aggregate(m2_per_cluster, cw, top_cids).tolist()

        if "true_distribution" in question and question["true_distribution"] is not None:
            true_vec = (list(question["true_distribution"].values())
                         if isinstance(question["true_distribution"], dict)
                         else question["true_distribution"])
            out["true"] = [float(x) for x in true_vec]
            for m in methods:
                pred_key = f"pred_{m}"
                if pred_key in out:
                    out[m] = evaluate(out[pred_key], true_vec)
        return out

    def run_questionnaire(
        self,
        questions: list[dict],
        posts_df: pd.DataFrame,
        cluster_topics: dict[str, str],
        methods: tuple[str, ...] = ("m0", "m1", "m2"),
        top_k: int = 10,
        n_posts: int = 50,
        n_samples: int = 3,
        ensemble_w: float = 0.5,
        seed: int = 42,
        anchors_cache: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        """End-to-end on a list of questions.

        If any question lacks `cluster_weights`, this method computes them
        on-the-fly via `compute_cluster_weights` (slower; ~1 LLM call per cluster
        per question).
        """
        # 1. Fill in missing cluster_weights
        for q in questions:
            if "cluster_weights" not in q or not q["cluster_weights"]:
                _log(f"  Computing cluster_weights for: {q.get('key', q['question'][:30])} …")
                q["cluster_weights"] = self.compute_cluster_weights(
                    q["question"], q["options"], cluster_topics)

        # 2. Decide top-K cluster list (shared across questions, like dump_*.py)
        top_cids = self.get_top_clusters(questions, top_k=top_k)
        _log(f"Top {top_k} clusters: {top_cids}")

        # 3. Anchors (deterministic, seeded)
        _set_seed(42)
        anchors = anchors_cache.copy() if anchors_cache else {}
        for q in questions:
            key = q.get("key", q["question"])
            if key not in anchors:
                anchors[key] = self.generate_anchors(q["question"], q["options"])
                _log(f"  anchor[{key}]: {anchors[key][0][:40]}…")

        # 4. Per-question run
        _set_seed(seed)
        results: dict[str, Any] = {
            "config": {"methods": list(methods), "top_k": top_k,
                       "n_posts": n_posts, "n_samples": n_samples,
                       "ensemble_w": ensemble_w, "seed": seed},
            "top_clusters": top_cids,
            "anchors": anchors,
            "per_question": {},
        }
        agg_metrics = {m: {"js": [], "k_xy": [], "c_xy": []} for m in methods}
        for q in questions:
            key = q.get("key", q["question"][:50])
            t0 = time.time()
            qr = self.run_question(
                q, posts_df, cluster_topics, top_cids,
                anchors=anchors[key], methods=methods,
                n_posts=n_posts, n_samples=n_samples,
                ensemble_w=ensemble_w, seed=seed,
            )
            results["per_question"][key] = qr
            line = f"  [{key}]"
            for m in methods:
                if m in qr:
                    line += f"  {m.upper()} JS={qr[m]['js']:.4f}"
                    for mt in ("js", "k_xy", "c_xy"):
                        agg_metrics[m][mt].append(qr[m][mt])
            _log(line + f"  elapsed={time.time()-t0:.0f}s")

        # 5. Summary across questions
        results["summary"] = {}
        for m in methods:
            if agg_metrics[m]["js"]:
                results["summary"][m] = {
                    "js": float(np.mean(agg_metrics[m]["js"])),
                    "k_xy": float(np.mean(agg_metrics[m]["k_xy"])),
                    "c_xy": float(np.mean(agg_metrics[m]["c_xy"])),
                    "n_questions": len(agg_metrics[m]["js"]),
                }
        return results


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════
def _cli_cluster(args):
    df = pd.read_csv(args.posts)
    pipe = PSSRPipeline(ssr_embed_path=args.ssr_embed, llm_model_path=args.llm,
                         device=args.device, cluster_embed_path=args.cluster_embed)
    out = pipe.cluster_posts(df, content_col=args.content_col,
                              hdbscan_min_cluster_size=args.min_cluster_size,
                              hdbscan_min_samples=args.min_samples,
                              umap_dim=args.umap_dim, drop_noise=True)
    out.to_csv(args.output, index=False)
    _log(f"Wrote {len(out):,} clustered posts → {args.output}")


def _cli_topics(args):
    df = pd.read_csv(args.posts)
    pipe = PSSRPipeline(ssr_embed_path=args.ssr_embed, llm_model_path=args.llm,
                         device=args.device)
    topics = pipe.generate_cluster_topics(df, content_col=args.content_col,
                                           n_per_cluster=args.n_per_cluster)
    with open(args.output, "w") as f:
        json.dump(topics, f, ensure_ascii=False, indent=2)
    _log(f"Wrote {len(topics)} cluster topics → {args.output}")


def _cli_predict(args):
    with open(args.questions) as f:
        questions = json.load(f)
    # Allow either a list or the original nested {survey:{sub:{q:...}}} format
    if isinstance(questions, dict):
        flat = []
        for sk, sv in questions.items():
            if isinstance(sv, dict):
                for sub, qs in sv.items():
                    if isinstance(qs, dict):
                        for qid, qd in qs.items():
                            qd = dict(qd)
                            qd["key"] = f"{sub}/{qid}"
                            flat.append(qd)
        questions = flat

    df = pd.read_csv(args.posts)
    with open(args.topics) as f:
        cluster_topics = json.load(f)

    pipe = PSSRPipeline(ssr_embed_path=args.ssr_embed, llm_model_path=args.llm,
                         device=args.device)
    methods = tuple(args.methods.split(","))
    results = pipe.run_questionnaire(
        questions, df, cluster_topics,
        methods=methods, top_k=args.top_k,
        n_posts=args.n_posts, n_samples=args.n_samples,
        ensemble_w=args.ensemble_w, seed=args.seed,
    )

    with open(args.output, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    _log(f"Wrote predictions → {args.output}")
    if "summary" in results:
        for m, s in results["summary"].items():
            _log(f"  {m.upper()}  JS={s['js']:.4f}  K={s['k_xy']:.3f}  "
                 f"C={s['c_xy']:.3f}  (n={s['n_questions']})")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PS-SSR pipeline (M0/M1/M2)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # shared
    def common(sp):
        sp.add_argument("--ssr_embed", default=DEFAULT_SSR_EMBED)
        sp.add_argument("--llm", default=DEFAULT_LLM)
        sp.add_argument("--device", default="cuda:0")
        sp.add_argument("--cluster_embed", default=DEFAULT_CLUSTER_EMBED)

    sp_c = sub.add_parser("cluster", help="Embed + UMAP + HDBSCAN; drop noise")
    common(sp_c)
    sp_c.add_argument("--posts", required=True)
    sp_c.add_argument("--output", required=True)
    sp_c.add_argument("--content_col", default="content_desc")
    sp_c.add_argument("--min_cluster_size", type=int, default=50)
    sp_c.add_argument("--min_samples", type=int, default=50)
    sp_c.add_argument("--umap_dim", type=int, default=50)
    sp_c.set_defaults(fn=_cli_cluster)

    sp_t = sub.add_parser("topics", help="Generate one-sentence topic per cluster")
    common(sp_t)
    sp_t.add_argument("--posts", required=True)
    sp_t.add_argument("--output", required=True)
    sp_t.add_argument("--content_col", default="content_desc")
    sp_t.add_argument("--n_per_cluster", type=int, default=20)
    sp_t.set_defaults(fn=_cli_topics)

    sp_p = sub.add_parser("predict", help="End-to-end M0/M1/M2 over a questionnaire")
    common(sp_p)
    sp_p.add_argument("--questions", required=True,
                      help="JSON: list of {key, question, options, cluster_weights?, "
                           "true_distribution?} OR original nested format")
    sp_p.add_argument("--posts", required=True, help="CSV: clustered posts")
    sp_p.add_argument("--topics", required=True, help="JSON: {cid: topic_str}")
    sp_p.add_argument("--output", required=True)
    sp_p.add_argument("--methods", default="m0,m1,m2")
    sp_p.add_argument("--top_k", type=int, default=10)
    sp_p.add_argument("--n_posts", type=int, default=50)
    sp_p.add_argument("--n_samples", type=int, default=3)
    sp_p.add_argument("--ensemble_w", type=float, default=0.5)
    sp_p.add_argument("--seed", type=int, default=42)
    sp_p.set_defaults(fn=_cli_predict)

    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    args.fn(args)
