import re
import io
from typing import List, Tuple

import tkinter as tk
from tkinter import ttk

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

try:
    from PIL import Image, ImageTk
    PIL_OK = True
except Exception:
    PIL_OK = False


# ---------- NORMALIZACJA LATEX W CAŁYM DOKUMENCIE ----------
_LATEX_DELIM_FIXES = [
    (re.compile(r"\\\\\["), r"\\["),   # \\[  -> \[
    (re.compile(r"\\\\\]"), r"\\]"),   # \\]  -> \]
    (re.compile(r"\\\\\("), r"\\("),   # \\(  -> \(
    (re.compile(r"\\\\\)"), r"\\)"),   # \\)  -> \)
    (re.compile(r"(?<!\\)\\{2}(?![\\\[\]\(\)\$])"), r"\\"),  # ogólne: podwójny \ -> pojedynczy \ (ostrożnie)
]

def normalize_latex_delims(text: str) -> str:
    """Naprawia typowe zdublowania backslashy w raw-stringach."""
    s = text
    for pat, rep in _LATEX_DELIM_FIXES:
        s = pat.sub(rep, s)
    return s


# ---------- SANITYZACJA WYRAŻENIA DLA MATHTEXT ----------
_SANITIZERS = [
    (re.compile(r"\\text\s*\{([^}]*)\}"), r"\\mathrm{\1}"),
    (re.compile(r"\\operatorname\s*\{([^}]*)\}"), r"\\mathrm{\1}"),
    (re.compile(r"\\argmin\b"), r"\\mathrm{argmin}"),
    (re.compile(r"\\argmax\b"), r"\\mathrm{argmax}"),
    (re.compile(r"\\mathbf\s*\{([^}]*)\}"), r"\\boldsymbol{\1}"),
]

def sanitize_math(expr: str) -> str:
    s = expr.replace("\\\\", "\\")
    for pat, rep in _SANITIZERS:
        s = pat.sub(rep, s)
    return s


# ---------- VIEWER ----------
class MarkdownViewer(ttk.Frame):
    """
    Lekki renderer Markdown + LaTeX (mathtext) do Tkinter Text.
    Obsługuje:
      - nagłówki #..###, listy, blockquote, hr, code fences, inline code;
      - LaTeX: bloki \[ ... \], $$ ... $$ oraz inline: \( ... \), $ ... $.
    """

    # Regexy do tokenizacji wierszowej
    _RE_HR = re.compile(r"^\s*[-*_]{3,}\s*$")
    _RE_HEADER = re.compile(r"^(#{1,6})\s+(.*)$")
    _RE_LIST = re.compile(r"^\s*[-*+]\s+(.*)$")
    _RE_BLOCKQUOTE = re.compile(r"^\s*>\s?(.*)$")
    _RE_FENCE = re.compile(r"^(```|~~~)")

    # Bloki LaTeX w liniach (pełne wiersze)
    _RE_BLOCK_LATEX_START = re.compile(r"^\s*(\\\[|\$\$)\s*")
    _RE_BLOCK_LATEX_END   = {r"\[": re.compile(r"(.*)\\\]\s*$"),
                             "$$":  re.compile(r"(.*)\$\$\s*$")}

    def __init__(self, master):
        super().__init__(master)
        self.text = tk.Text(self, wrap="word", font=("Segoe UI", 11), padx=6, pady=6)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.hsb = ttk.Scrollbar(self, orient="horizontal", command=self.text.xview)
        self.text.configure(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)

        self.text.grid(row=0, column=0, sticky="nsew")
        self.vsb.grid(row=0, column=1, sticky="ns")
        self.hsb.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # style
        self.text.tag_configure("h1", font=("Segoe UI Semibold", 18))
        self.text.tag_configure("h2", font=("Segoe UI Semibold", 16))
        self.text.tag_configure("h3", font=("Segoe UI Semibold", 14))
        self.text.tag_configure("code", font=("Consolas", 11))
        self.text.tag_configure("li", lmargin1=18, lmargin2=36)
        self.text.tag_configure("blockquote", lmargin1=18, lmargin2=18, foreground="#555")
        self.text.tag_configure("hr", foreground="#bbb", underline=True)
        self.text.tag_configure("bold", font=("Segoe UI", 11, "bold"))
        self.text.tag_configure("italic", font=("Segoe UI", 11, "italic"))
        self.text.tag_configure("bolditalic", font=("Segoe UI", 11, "bold italic"))

        # trzymamy referencje do obrazów
        self._images: List[ImageTk.PhotoImage] = []

        self.text.config(state="disabled")

    # ---------- PUBLIC ----------
    def render(self, md: str):
        # 1) normalizacja backslashy (naprawia \[...\], \(...\) itp.)
        md = normalize_latex_delims(md)

        self._images.clear()
        self.text.config(state="normal")
        self.text.delete("1.0", "end")

        lines = md.splitlines()
        i = 0
        in_code = False
        fence_token = None
        code_buf: List[str] = []

        in_block_math = False
        block_math_kind = None   # "\[" albo "$$"
        math_buf: List[str] = []

        while i < len(lines):
            line = lines[i]

            # --- code fences ---
            if not in_block_math and self._RE_FENCE.match(line.strip()):
                tok = self._RE_FENCE.match(line.strip()).group(1)
                if not in_code:
                    in_code = True
                    fence_token = tok
                    code_buf = []
                else:
                    if tok == fence_token:
                        self._insert_code_block("\n".join(code_buf))
                        in_code = False
                        fence_token = None
                        code_buf = []
                i += 1
                continue

            if in_code:
                code_buf.append(line)
                i += 1
                continue

            # --- block LaTeX start? (\[ ...  or  $$ ... ) ---
            if not in_block_math:
                mstart = self._RE_BLOCK_LATEX_START.match(line)
                if mstart:
                    kind = mstart.group(1)   # "\[" lub "$$"
                    block_math_kind = r"\[" if kind.startswith("\\[") else "$$"
                    content = line[mstart.end():]
                    mend = self._RE_BLOCK_LATEX_END[block_math_kind].match(content)
                    if mend:
                        # start i koniec w jednej linii
                        expr = mend.group(1).strip()
                        self._insert_math(expr, block=True)
                    else:
                        in_block_math = True
                        math_buf = [content]
                    i += 1
                    continue

            # --- block LaTeX kontynuacja ---
            if in_block_math:
                mend = self._RE_BLOCK_LATEX_END[block_math_kind].match(line)
                if mend:
                    math_buf.append(mend.group(1))
                    expr = "\n".join(math_buf).strip()
                    self._insert_math(expr, block=True)
                    in_block_math = False
                    block_math_kind = None
                    math_buf = []
                else:
                    math_buf.append(line)
                i += 1
                continue

            # --- hr ---
            if self._RE_HR.match(line):
                self._insert_hr()
                i += 1
                continue

            # --- headers ---
            mh = self._RE_HEADER.match(line)
            if mh:
                level = len(mh.group(1))
                text = mh.group(2).strip()
                self._insert_heading(text, level)
                i += 1
                continue

            # --- blockquote ---
            mb = self._RE_BLOCKQUOTE.match(line)
            if mb:
                self._insert_emphasis(mb.group(1), extra_tags=("blockquote",))
                self.text.insert("end", "\n")
                i += 1
                continue

            # --- list item ---
            ml = self._RE_LIST.match(line)
            if ml:
                self.text.insert("end", "• ", ("li",))
                self._insert_emphasis(ml.group(1), extra_tags=("li",))
                self.text.insert("end", "\n")
                i += 1
                continue

            # --- zwykły paragraf z inline math i inline code ---
            self._insert_paragraph(line)
            i += 1

        self.text.config(state="disabled")

    # ---------- wstawki ----------
    def _insert_heading(self, text: str, level: int):
        tag = "h1" if level == 1 else ("h2" if level == 2 else "h3")
        self.text.insert("end", text + "\n", (tag,))
        self.text.insert("end", "\n")

    def _insert_hr(self):
        self.text.insert("end", "—" * 50 + "\n", ("hr",))

    def _insert_code_block(self, code: str):
        self.text.insert("end", code.rstrip() + "\n", ("code",))
        self.text.insert("end", "\n")

    def _insert_paragraph(self, line: str):
        # 1) inline code `...`
        chunks = self._split_inline_code(line)
        for is_code, chunk in chunks:
            if is_code:
                self.text.insert("end", chunk, ("code",))
                continue
            # 2) inline LaTeX: \( ... \)  oraz  $ ... $
            self._insert_with_inline_math(chunk)
        self.text.insert("end", "\n")

    def _split_inline_code(self, text: str) -> List[Tuple[bool, str]]:
        out: List[Tuple[bool, str]] = []
        i = 0
        while i < len(text):
            m = re.search(r"`", text[i:])
            if not m:
                out.append((False, text[i:]))
                break
            start = i + m.start()
            if start > i:
                out.append((False, text[i:start]))
            m2 = re.search(r"`", text[start+1:])
            if not m2:
                out.append((False, text[start:]))
                break
            end = start + 1 + m2.start()
            out.append((True, text[start+1:end]))
            i = end + 1
        return out

    def _insert_emphasis(self, text: str, extra_tags=()):
        """
        Parsuje **bold** / __bold__ / *italic* / _italic_ i wstawia do Text z odpowiednimi tagami.
        'extra_tags' pozwala dołożyć tag (np. 'li' albo 'blockquote') do każdej porcji tekstu.
        Priorytet: najpierw bold (**, __), potem italic (*, _). Nie wspiera zagnieżdżonych konstrukcji.
        """
        i = 0
        n = len(text)

        # szybkie findery
        bold_re = re.compile(r"(\*\*|__)(.+?)\1")
        ital_re = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)|(?<!_)_(?!_)(.+?)(?<!_)_(?!_)")

        while i < n:
            # Szukaj najbliższego dopasowania (bold lub italic)
            b = bold_re.search(text, i)
            it = ital_re.search(text, i)

            # wybierz wcześniejsze wystąpienie
            cand = []
            if b: cand.append(("bold", b.start(), b))
            if it: cand.append(("italic", it.start(), it))
            if not cand:
                # reszta jako zwykły tekst
                if i < n:
                    self.text.insert("end", text[i:], extra_tags)
                break

            kind, pos, m = min(cand, key=lambda t: t[1])

            # przed fragmentem sformatowanym – zwykły tekst
            if pos > i:
                self.text.insert("end", text[i:pos], extra_tags)

            # sam fragment
            if kind == "bold":
                inner = m.group(2)
                self.text.insert("end", inner, ("bold",) + tuple(extra_tags))
                i = m.end()
            else:
                # italic może zwrócić w grupie 1 lub 2 (gwiazdka vs podkreślnik)
                inner = m.group(1) if m.group(1) is not None else m.group(2)
                self.text.insert("end", inner, ("italic",) + tuple(extra_tags))
                i = m.end()

    def _insert_with_inline_math(self, text: str):
        """
        Dzieli tekst na zwykłe fragmenty i fragmenty LaTeX dla:
          - \( ... \)
          - $ ... $   (pojedyncze $; ignorujemy $$ tutaj)
        """
        parts: List[Tuple[bool, str]] = []
        i = 0
        while i < len(text):
            # znajdź najbliższe \( lub $
            m_par = re.search(r"\\\(", text[i:])
            m_dol = re.search(r"(?<!\$)\$(?!\$)", text[i:])  # pojedynczy $
            # wybierz pierwsze w kolejności
            cand = []
            if m_par: cand.append(("par", i + m_par.start()))
            if m_dol: cand.append(("dol", i + m_dol.start()))
            if not cand:
                parts.append((False, text[i:]))
                break
            kind, pos = min(cand, key=lambda t: t[1])
            if pos > i:
                parts.append((False, text[i:pos]))
            if kind == "par":
                # szukaj zamknięcia \)
                m_end = re.search(r"\\\)", text[pos+2:])
                if not m_end:
                    # brak końca – traktuj jako zwykły tekst
                    parts.append((False, text[pos:]))
                    break
                end = pos + 2 + m_end.start()
                expr = text[pos+2:end]
                parts.append((True, ("par", expr)))
                i = end + 2
            else:
                # pojedynczy $ ... $
                m_end = re.search(r"(?<!\$)\$(?!\$)", text[pos+1:])
                if not m_end:
                    parts.append((False, text[pos:]))
                    break
                end = pos + 1 + m_end.start()
                expr = text[pos+1:end]
                parts.append((True, ("dol", expr)))
                i = end + 1

        # wstaw
        for is_math, payload in parts:
            if not is_math:
                self._insert_emphasis(payload)
            else:
                kind, expr = payload
                self._insert_math(expr, block=False)

    # ---------- render LaTeX jako obraz ----------
    def _insert_math(self, expr: str, block: bool):
        if not PIL_OK:
            self.text.insert("end", f"[LaTeX] {expr}")
            if block:
                self.text.insert("end", "\n")
            return

        # 1) sanityzacja poleceń nieszczególnie lub średnio wspieranych przez mathtext
        expr = sanitize_math(expr)

        # 2) zbuduj figurę wystarczająco dużą, by nic nie zostało ucięte
        #    (później i tak przytniemy do 'tight' w savefig)
        base_size = (4.0, 1.6) if block else (2.4, 1.0)  # szer., wys. w calach
        dpi = 220

        try:
            fig = Figure(figsize=base_size, dpi=dpi)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")

            # \displaystyle tylko dla bloków; inline bez niego (mniej „wysokie”)
            to_draw = rf"$\displaystyle {expr}$" if block else rf"${expr}$"
            ax.text(0.5, 0.5, to_draw, ha="center", va="center", fontsize=14)

            buf = io.BytesIO()
            # KLUCZ: savefig + bbox_inches='tight' -> brak obcinania formuły
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                        pad_inches=0.02, transparent=True)
            buf.seek(0)

            from PIL import Image
            img = Image.open(buf).convert("RGBA")

            # (opcjonalne) dodatkowe ucinanie przez PIL, zwykle niepotrzebne po 'tight'
            bbox = img.getbbox()
            if bbox:
                img = img.crop(bbox)

            tkimg = ImageTk.PhotoImage(img)
            self._images.append(tkimg)
            self.text.image_create("end", image=tkimg)

            if block:
                self.text.insert("end", "\n")

        except Exception as e:
            # pokaż błąd, ale nie zatrzymuj renderu reszty dokumentu
            self.text.insert("end", f"[LaTeX render error] {e}\n")

