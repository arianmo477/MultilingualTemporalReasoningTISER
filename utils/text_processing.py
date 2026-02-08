import re

# ==================================================
# UNICODE / TEMPORAL / ENTITY
# ==================================================


#def repair_mangled_unicode(text: str) -> str:
#    if not text or not isinstance(text, str):
#        return text
#    try:
#        return text.encode("latin1").decode("utf8")
#    except Exception:
#        return text


_U_ESCAPE_RE = re.compile(r"u([0-9a-fA-F]{4})")

def repair_mangled_unicode(s: str) -> str:
    if not s:
        return ""
    if _U_ESCAPE_RE.search(s) is None:
        return s
    try:
        s2 = _U_ESCAPE_RE.sub(r"\\u\1", s)
        return s2.encode("utf-8").decode("unicode_escape")
    except Exception:
        return s


# Multilingual temporal patterns
_RANGE_RE_EN = re.compile(r"\b(\d{4})\s*-\s*(\d{4})\b")
_STARTS_RE_EN = re.compile(r"\bstarts at\s+(\d{4})\b", re.IGNORECASE)
_ENDS_RE_EN = re.compile(r"\bends at\s+(\d{4})\b", re.IGNORECASE)

# Italian patterns
_RANGE_RE_IT = re.compile(r"\bdal\s+(\d{4})\s+al\s+(\d{4})\b", re.IGNORECASE)
_STARTS_RE_IT = re.compile(r"\binizia\s+nel\s+(\d{4})\b", re.IGNORECASE)
_ENDS_RE_IT = re.compile(r"\bfinisce\s+nel\s+(\d{4})\b", re.IGNORECASE)

# German patterns
_RANGE_RE_DE = re.compile(r"\bvon\s+(\d{4})\s+bis\s+(\d{4})\b", re.IGNORECASE)
_STARTS_RE_DE = re.compile(r"\bbeginnt\s+(im\s+Jahr\s+)?(\d{4})\b", re.IGNORECASE)
_ENDS_RE_DE = re.compile(r"\bendet\s+(im\s+Jahr\s+)?(\d{4})\b", re.IGNORECASE)

# Persian patterns (using Persian digits)
_RANGE_RE_FA = re.compile(r"\bاز\s+([۰-۹]{4})\s+تا\s+([۰-۹]{4})\b")
_STARTS_RE_FA = re.compile(r"\bشروع\s+در\s+([۰-۹]{4})\b")
_ENDS_RE_FA = re.compile(r"\bپایان\s+در\s+([۰-۹]{4})\b")

def normalize_temporal(text: str, language: str = "en") -> str:
    """
    Normalize temporal expressions in multiple languages.
    
    Args:
        text: Text to normalize
        language: Language code ('en', 'it', 'de', 'fa')
    
    Returns:
        Normalized text
    """
    if not text:
        return text
    text = re.sub(r"\s+", " ", text).strip()
    
    if language == "en":
        text = _RANGE_RE_EN.sub(r"From \1 to \2", text)
        text = _STARTS_RE_EN.sub(r"started in \1", text)
        text = _ENDS_RE_EN.sub(r"ended in \1", text)
    elif language == "it":
        text = _RANGE_RE_IT.sub(r"Dal \1 al \2", text)
        text = _STARTS_RE_IT.sub(r"iniziato nel \1", text)
        text = _ENDS_RE_IT.sub(r"finito nel \1", text)
    elif language == "de":
        text = _RANGE_RE_DE.sub(r"Von \1 bis \2", text)
        text = _STARTS_RE_DE.sub(lambda m: f"begann im Jahr {m.group(2)}", text)
        text = _ENDS_RE_DE.sub(lambda m: f"endete im Jahr {m.group(2)}", text)
    elif language == "fa":
        text = _RANGE_RE_FA.sub(r"از \1 تا \2", text)
        text = _STARTS_RE_FA.sub(r"شروع شده در \1", text)
        text = _ENDS_RE_FA.sub(r"پایان یافته در \1", text)
    
    return text


PAREN_ENTITY_RE = re.compile(r"\(([^()]+)\)")
def make_ent_token(i: int) -> str:
    return f"⟪ENT{i:06d}⟫"

def looks_like_name(text: str) -> bool:
    words = text.strip().split()

    if len(words) > 3:
        return False

    for w in words:
        # allow acronyms like JHU
        if w.isupper():
            continue

        # strip punctuation
        w_clean = re.sub(r"[^\w]", "", w)

        # must start with uppercase
        if not w_clean or not w_clean[0].isupper():
            return False

        # no digits
        if any(c.isdigit() for c in w_clean):
            return False

    return True


def mask_parenthesized_entities(text: str):
    if not text:
        return text, {}

    mapping = {}
    out = text
    idx = 0

    for m in PAREN_ENTITY_RE.finditer(text):
        full = "(" + m.group(1) + ")"
        inner = m.group(1).strip()

        if not looks_like_name(inner):
            continue  #  translate this

        tok = make_ent_token(idx)
        mapping[tok] = full
        out = out.replace(full, f" {tok} ")
        idx += 1

    return out, mapping

def parse_cot(text: str):
    parts = {"reasoning": "", "timeline": "", "reflection": "", "answer": ""}
    for k in parts:
        m = re.search(rf"<{k}>(.*?)</{k}>", text, re.DOTALL | re.IGNORECASE)
        if m:
            parts[k] = m.group(1).strip()
    return parts

def rebuild_cot(parts: dict):
    return (
        f"<reasoning>\n{parts['reasoning']}\n</reasoning>\n"
        f"<timeline>\n{parts['timeline']}\n</timeline>\n"
        f"<reflection>\n{parts['reflection']}\n</reflection>\n"
        f"<answer>\n{parts['answer']}\n</answer>"
    )



def unmask_entities(text: str, mapping: dict) -> str:
    """
    Robustly restores masked parenthesized entities.
    Handles cases where MT slightly corrupts the placeholder
    (spaces, quotes, brackets, etc.).
    """
    if not text or not mapping:
        return text

    for tok, original in mapping.items():
        # Extract numeric ID from token ⟪ENT000001⟫
        m = re.search(r"ENT(\d{6})", tok)
        if not m:
            continue
        ent_id = m.group(1)

        # Robust pattern: allow spaces, quotes, brackets around ENT + id
        pattern = re.compile(
            r"[\s\"'«»⟪⟫\[\]\(\)]*ENT\s*"
            + ent_id +
            r"[\s\"'«»⟪⟫\[\]\(\)]*",
            re.IGNORECASE
        )

        text = pattern.sub(original, text)

    return text