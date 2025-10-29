# /home/esakit25/work/tensaku/src/tensaku/model_io.py
"""
@module: tensaku.model_io
@role: Model/Tokenizer I/O と推論ユーティリティ（薄い共通層）
@inputs:
  - model_name_or_path: str  # HFの識別子 or ローカルパス
  - num_labels: int          # 分類ラベル数（0..N の N+1）
  - device: "auto" | "cpu" | "cuda" | torch.device
  - dataloader: Iterable[dict]  # keys: input_ids[, attention_mask] など
  - texts: list[str]        # Tokenizerにそのまま渡せるテキスト
  - max_length: int         # トークナイズ長
  - batch_size: int
@outputs:
  - tokenizer: transformers.PreTrainedTokenizerBase
  - model: transformers.PreTrainedModel（SequenceClassification）
  - device: torch.device
  - logits: np.ndarray shape (N, C)
@notes:
  - Tensakuの方針に合わせ、**学習前にtrain/dev分離**、前処理fitはtrainのみ、T/τはdevで推定→pool/testへ固定。
  - 本モジュールは「読み込み・推論・デバイス移送」の薄いユーティリティに限定（学習ループは別）。
  - 依存がない環境でも *輸入時に壊れない* ように try/except を入れてある（実行には torch/transformers が必要）。
  - Gate/Confidence/Calibration/Trust から再利用されることを想定。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple, Union, List, TYPE_CHECKING
import numpy as np

# ---- Optional deps ------------------------------------------------------------------------------

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None     # type: ignore

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except Exception:  # pragma: no cover
    AutoTokenizer = None  # type: ignore
    AutoModelForSequenceClassification = None  # type: ignore

# ==== Typing fix for Pylance ("型式では変数を使用できません") ===========================
# 型ヒント中で実体変数（Tensor 等）を直接使うと Pylance が嫌うため、
# TYPE_CHECKING ブロックでのみ厳密型を import。実行時は Any にフォールバック。
if TYPE_CHECKING:
    from torch import Tensor as TorchTensor  # noqa: F401
    from torch import device as TorchDevice  # noqa: F401
else:
    TorchTensor = Any        # type: ignore
    TorchDevice = Any        # type: ignore
# ===================================================================


# ---- Device helpers ------------------------------------------------------------------------------

def select_device(device: Union[str, TorchDevice] = "auto") -> TorchDevice:
    if torch is None:
        raise RuntimeError("PyTorch is required but not available.")
    if device == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device  # type: ignore[return-value]


def move_batch_to_device(batch: Dict[str, Any], device: TorchDevice) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required.")
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if hasattr(v, "to"):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


# ---- Loaders -------------------------------------------------------------------------------------

def load_tokenizer(model_name_or_path: str, use_fast: bool = True):
    if AutoTokenizer is None:
        raise RuntimeError("transformers is required but not available.")
    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast)
    return tok


def load_cls_model(
    model_name_or_path: str,
    num_labels: int,
    device: Union[str, TorchDevice] = "auto",
    *,
    freeze_base: bool = False,
    dtype: Optional[str] = None,  # "float16" | "bfloat16" | None
):
    if AutoModelForSequenceClassification is None or torch is None:
        raise RuntimeError("transformers/torch are required.")
    dev = select_device(device)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=int(num_labels),
        ignore_mismatched_sizes=True,
    )
    if freeze_base and hasattr(model, "bert"):
        for p in model.bert.parameters():  # type: ignore[attr-defined]
            p.requires_grad = False
    if dtype == "float16":
        model = model.to(dev, dtype=torch.float16)
    elif dtype == "bfloat16":
        model = model.to(dev, dtype=torch.bfloat16)
    else:
        model = model.to(dev)
    model.eval()
    return model, dev


# ---- Dataloader (texts → batches) ----------------------------------------------------------------

def build_text_loader(
    tokenizer,
    texts: List[str],
    *,
    max_length: int = 128,
    batch_size: int = 16,
) -> Iterable[Dict[str, TorchTensor]]:
    """
    依存最小化のため簡易Generatorで返却（DataLoader不要）。
    """
    if torch is None:
        raise RuntimeError("PyTorch is required.")
    n = len(texts)
    for i in range(0, n, batch_size):
        chunk = texts[i : i + batch_size]
        tk = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        yield {"input_ids": tk["input_ids"], "attention_mask": tk.get("attention_mask")}


# ---- Inference -----------------------------------------------------------------------------------

def predict_logits(
    model: Any,
    dataloader: Iterable[Dict[str, Any]],
    *,
    device: Union[str, TorchDevice] = "auto",
    autocast_dtype: Optional[str] = None,  # "float16" | "bfloat16" | None
) -> np.ndarray:
    """
    与えられた dataloader（辞書バッチ）から logits を連結返却。
    """
    if torch is None:
        raise RuntimeError("PyTorch is required.")
    dev = select_device(device)

    no_grad = torch.no_grad
    amp_ctx = (
        torch.autocast(device_type=dev.type, dtype=_to_torch_dtype(autocast_dtype))
        if autocast_dtype is not None
        else _nullcontext()
    )

    outs: List["TorchTensor"] = []
    model.eval()
    with no_grad():
        with amp_ctx:
            for batch in dataloader:
                tb = move_batch_to_device(batch, dev)
                # HFモデル前提の引数名（追加キーは無視）
                out = model(
                    input_ids=tb.get("input_ids"),
                    attention_mask=tb.get("attention_mask"),
                    token_type_ids=tb.get("token_type_ids", None),
                )
                logits = out.logits if hasattr(out, "logits") else out[0]
                outs.append(logits.detach().cpu())
    return torch.cat(outs, dim=0).numpy()  # type: ignore[return-value]


def predict_logits_from_texts(
    tokenizer,
    model: Any,
    texts: List[str],
    *,
    max_length: int = 128,
    batch_size: int = 16,
    device: Union[str, TorchDevice] = "auto",
    autocast_dtype: Optional[str] = None,
) -> np.ndarray:
    loader = build_text_loader(tokenizer, texts, max_length=max_length, batch_size=batch_size)
    return predict_logits(model, loader, device=device, autocast_dtype=autocast_dtype)


# ---- Checkpoint helpers（読み書きは最小限） ---------------------------------------------------------

def save_ckpt(model: Any, path: str) -> None:
    """
    研究運用の簡易保存（HF推奨のsave_pretrainedと併用可）:
      - CPUへ移して state_dict を atomic save（例: outputs/checkpoints_min/last.pt）
    """
    if torch is None:
        raise RuntimeError("PyTorch is required.")
    import os, tempfile
    state = model.state_dict()
    cpu_state = {k: v.detach().cpu() for k, v in state.items()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=".tmp_ckpt_", dir=os.path.dirname(path))
    try:
        with open(tmp_fd, "wb") as f:
            torch.save({"state_dict": cpu_state}, f)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def load_ckpt_if_exists(model: Any, path: str) -> bool:
    """
    学習再開用の軽量ローダ。存在すれば `model.load_state_dict(..., strict=False)` で復元。
    返り値: 読み込めた場合 True。
    """
    if torch is None:
        raise RuntimeError("PyTorch is required.")
    import os
    if not os.path.isfile(path):
        return False
    payload = torch.load(path, map_location="cpu")
    state = None
    for key in ("state_dict", "model_state_dict", "model", "net"):
        if isinstance(payload, dict) and key in payload:
            state = payload[key]
            break
    if state is None and isinstance(payload, dict):
        state = payload  # そのままstate_dictだった場合
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:  # 軽いログ（厳密さは外部で担保）
        print(f"[model_io] missing keys: {len(missing)}")
    if unexpected:
        print(f"[model_io] unexpected keys: {len(unexpected)}")
    return True


# ---- small utils ---------------------------------------------------------------------------------

def _to_torch_dtype(name: Optional[str]):
    if torch is None or name is None:
        return None
    low = name.lower()
    if low in ("fp16", "float16", "half"):
        return torch.float16
    if low in ("bf16", "bfloat16"):
        return torch.bfloat16
    return None


class _nullcontext:
    def __enter__(self): return self
    def __exit__(self, *exc): return False


# ---- Self test (no internet required) ------------------------------------------------------------

if __name__ == "__main__":
    """
    自己テスト方針：
      - FakeTokenizer: texts→固定長のID/Maskを返す
      - FakeModel: (B,L)->logits を返す単純線形層で predict_logits を検証
      - transformers/インターネットが不要な経路のみ通す
    """
    if torch is None:
        raise SystemExit("[skip] torch not available")

    class FakeTokenizer:
        def __init__(self, vocab_size=1000): self.vocab_size=vocab_size
        def __call__(self, texts, padding=True, truncation=True, max_length=16, return_tensors="pt"):
            B = len(texts)
            ids = torch.randint(0, self.vocab_size, (B, max_length))
            mask = torch.ones((B, max_length), dtype=torch.long)
            return {"input_ids": ids, "attention_mask": mask}

    class _Out:
        def __init__(self, logits): self.logits = logits

    class FakeModel(nn.Module):
        def __init__(self, hidden=32, num_labels=7, seq_len=16):
            super().__init__()
            self.seq_len = seq_len
            self.enc = nn.Linear(seq_len, hidden)
            self.head = nn.Linear(hidden, num_labels)
        def forward(self, input_ids=None, attention_mask=None, **_):
            if input_ids is None: raise ValueError("input_ids required")
            x = input_ids.float()
            x = self.enc(x).tanh()
            logits = self.head(x)
            return _Out(logits)

    dev = select_device("auto")
    tok = FakeTokenizer()
    mdl = FakeModel().to(dev).eval()

    texts = ["あいうえお", "かきくけこ", "さしすせそ"]
    loader = build_text_loader(tok, texts, max_length=16, batch_size=2)
    logits = predict_logits(mdl, loader, device=dev)
    assert logits.shape[0] == len(texts), f"shape mismatch: {logits.shape}"
    print("[model_io] self-test OK, logits shape:", logits.shape)
