"""설정 로드: .env, configs/models.yaml, 별칭으로 provider 생성"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .errors import AuthError
from .providers.claude import Claude
from .providers.gpt import GPT
from .providers.openrouter import OpenRouter

PROVIDERS = {"claude": Claude, "gpt": GPT, "openrouter": OpenRouter}
KEYS = {"claude": "ANTHROPIC_API_KEY", "gpt": "OPENAI_API_KEY", "openrouter": "OPENROUTER_API_KEY"}


def find_root() -> Path:
    """configs/models.yaml이 있는 프로젝트 루트 (노트북 위치와 무관하게 찾음)"""
    here = Path.cwd()
    for d in [here, *here.parents, Path(__file__).resolve().parents[2]]:
        if (d / "configs" / "models.yaml").exists():
            return d
    raise FileNotFoundError("configs/models.yaml을 찾을 수 없습니다")


ROOT = find_root()
load_dotenv(ROOT / ".env")


def load_models() -> dict:
    return yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))


def make(alias=None):
    """별칭으로 provider 생성 (None이면 active)"""
    cfg = load_models()
    alias = alias or cfg["active"]
    if alias not in cfg["models"]:
        raise KeyError(f"models.yaml에 없는 별칭: {alias} (가능: {', '.join(cfg['models'])})")
    m = cfg["models"][alias]
    key = KEYS[m["provider"]]
    if os.environ.get(key, "YOUR_").startswith("YOUR_"): # 없거나 .env.example 자리 표시자 그대로
        raise AuthError(
            f"'{alias}' 모델을 쓰려면 {ROOT / '.env'}에 {key}를 넣으세요. "
            f"키가 있는 다른 모델을 쓰려면 configs/models.yaml의 active를 바꾸세요."
        )
    return PROVIDERS[m["provider"]](model=m["model"], **(m.get("options") or {}))


llm = make()
