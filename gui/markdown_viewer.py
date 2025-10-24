import re
import io
from typing import List, Tuple, Optional

import tkinter as tk
from tkinter import ttk

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

try:
    from PIL import Image, ImageTk
    PIL_OK = True
except Exception:
    PIL_OK = False


# ---------- pomocnicze: sanityzacja LaTeXa dla mathtext ----------
_MATH_TEXT_REPLACERS: List[Tuple[re.Pattern, str]] = [
    # \text{...} -> \mathrm{...}
    (re.compile(r"\\text\s*\{([^}]*)\}"), r"\\mathrm{\1}"),
    # \operatorname{...} -> \mathrm{...}
    (re.compile(r"\\operatorname\s*\{([^}]*)\}"), r"\\mathrm{\1}"),
    # \argmin, \argmax -> \mathrm{argmin}/\mathrm{argmax}
    (re.compile(r"\\argmin\b"), r"\\mathrm{argmin}"),
    (re.compile(r"\\argmax\b"), r"\\mathrm{argmax}"),
    # \boldsymbol{...} zostawiamy (mathtext wspiera), ale dopuszczamy też \mathbf -> \boldsymbol
    (re.compile(r"\\mathbf\s*\{([^}]*)\}"), r"\\boldsymbol{\1}"),
]

def sanitize_math(expr: str) -> str:
    # 1) podwójne backslash'e (np. z r"""...""") -> pojedyncze
    s = expr.replace("\\\\", "\\")
    # 2) zamiany nieobsługiwanych komend
    for pat, rep in _MATH_TEXT_REPLACERS:
        s = pat.sub(rep, s)
    return s


# ---------- prosty renderer Markdown + LaTeX do Text ----------
class MarkdownViewer(ttk.Frame):

    def __init__(self, master):
        super().__init__(master)
        self.text = tk.Text(self, wrap="word", font=("Segoe UI", 11),
                            padx=6, pady=6, undo=False)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.hsb = ttk.Scrollbar(self, orient="horizontal", command=self.text.xview)
        self.text.configure(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)

        self.text.grid(row=0, column=0, sticky="nsew")
        self.vsb.grid(row=0, column=1, sticky="ns")
        self.hsb.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # Tag styles
        self.text.tag_configure("h1", font=("Segoe UI Semibold", 18))
        self.text.tag_configure("h2", font=("Segoe UI Semibold", 16))
        self.text.tag_configure("h3", font=("Segoe UI Semibold", 14))
        self.text.tag_configure("bold", font=("Segoe UI Semibold", 11))
        self.text.tag_configure("code", font=("Consolas", 11))
        self.text.tag_configure("blockquote", lmargin1=18, lmargin2=18, foreground="#555")
        self.text.tag_configure("li", lmargin1=18, lmargin2=36)
        self.text.tag_configure("hr", foreground="#bbb", underline=True)

        # Obrazki LaTeX muszą mieć mocne referencje
        self._images: List[ImageTk.PhotoImage] = []

        # tryb tylko do podglądu – nic nie edytujemy
        self.text.config(state="disabled")

    # ------------ API ------------
    def render(self, md: str):
        """Wyrenderuj podany Markdown + LaTeX do Text."""
        self._images.clear()
        self.text.config(state="normal")
        self.text.delete("1.0", "end")

        # parsujemy linia po linii z obsługą bloków: code/latex
        lines = md.splitlines()
        i = 0
        in_code = False
        code_fence = None
        code_buf: List[str] = []

        in_math_block = False
        math_delim = None
        math_buf: List[str] = []

        while i < len(lines):
            line = lines[i]

            # --- Fenced code blocks ---
            if not in_math_block and (line.strip().startswith("```") or line.strip().startswith("~~~")):
                fence = line.strip()[:3]
                if not in_code:
                    in_code = True
                    code_fence = fence
                    code_buf = []
                else:
                    if fence == code_fence:
                        self._insert_code_block("\n".join(code_buf))
                        in_code = False
                        code_fence = None
                        code_buf = []
                i += 1
                continue
            if in_code:
                code_buf.append(line)
                i += 1
                continue

            # --- Math blocks: \[...\] or $$...$$ ---
            ls = line.strip()
            if not in_code:
                # start?
                if not in_math_block and (ls.startswith("\\[") or ls.startswith("$$")):
                    in_math_block = True
                    math_delim = "\\]" if ls.startswith("\\[") else "$$"
                    # usuń początek
                    start = 2 if math_delim == "\\]" else 2
                    content = line[line.find(ls[0:start])+start:]
                    # jeśli od razu jest koniec w tej samej linii
                    if math_delim in content:
                        before, after = content.split(math_delim, 1)
                        self._insert_math_block(before.strip())
                        if after.strip():
                            self._insert_paragraph(after)
                        in_math_block = False
                        math_delim = None
                        math_buf = []
                    else:
                        math_buf = [content]
                    i += 1
                    continue
                # koniec?
                if in_math_block:
                    if math_delim in line:
                        before, after = line.split(math_delim, 1)
                        math_buf.append(before)
                        self._insert_math_block("\n".join(math_buf).strip())
                        if after.strip():
                            self._insert_paragraph(after)
                        in_math_block = False
                        math_delim = None
                        math_buf = []
                    else:
                        math_buf.append(line)
                    i += 1
                    continue

            # --- pozioma linia ---
            if re.fullmatch(r"\s*[-*_]{3,}\s*", line):
                self._insert_hr()
                i += 1
                continue

            # --- nagłówki ---
            m = re.match(r"^(#{1,6})\s+(.*)$", line)
            if m:
                level = len(m.group(1))
                text = m.group(2).strip()
                self._insert_heading(text, level)
                i += 1
                continue

            # --- blockquote ---
            if line.startswith(">"):
                self._insert_blockquote(line[1:].lstrip())
                i += 1
                continue

            # --- listy ---
            if re.match(r"^\s*[-*+]\s+", line):
                self._insert_list_item(re.sub(r"^\s*[-*+]\s+", "", line))
                i += 1
                continue

            # zwykły paragraf (z inline code i inline math)
            self._insert_paragraph(line)
            i += 1

        self.text.config(state="disabled")

    # ------------ wstawki ------------
    def _insert_heading(self, text: str, level: int):
        tag = "h1" if level == 1 else "h2" if level == 2 else "h3"
        self.text.insert("end", text + "\n", (tag,))
        self.text.insert("end", "\n")

    def _insert_blockquote(self, text: str):
        self.text.insert("end", text + "\n", ("blockquote",))
        self.text.insert("end", "\n")

    def _insert_list_item(self, text: str):
        self.text.insert("end", "• " + text + "\n", ("li",))

    def _insert_hr(self):
        self.text.insert("end", "—" * 40 + "\n", ("hr",))

    def _insert_code_block(self, code: str):
        self.text.insert("end", code.rstrip() + "\n", ("code",))
        self.text.insert("end", "\n")

    def _insert_paragraph(self, line: str):
        # inline code `...`
        parts = self._split_inline(line, r"`", r"`", as_math=False)
        for is_code, chunk in parts:
            if is_code:
                self.text.insert("end", chunk, ("code",))
            else:
                # inline math $...$
                self._insert_with_inline_math(chunk)
        self.text.insert("end", "\n")

    def _insert_with_inline_math(self, text: str):
        parts = self._split_inline(text, r"\$", r"\$", as_math=True)
        for is_math, chunk in parts:
            if is_math:
                self._insert_math_inline(chunk)
            else:
                self.text.insert("end", chunk)

    def _split_inline(self, text: str, left: str, right: str, as_math: bool) -> List[Tuple[bool, str]]:
        """
        Dzielenie tekstu na fragmenty zwykłe i inline (code/math).
        Zwraca listę (is_inline, content).
        """
        out: List[Tuple[bool, str]] = []
        i = 0
        L = len(text)
        while i < L:
            m = re.search(left, text[i:])
            if not m:
                out.append((False, text[i:]))
                break
            start = i + m.start()
            # dorzuć zwykły fragment
            if start > i:
                out.append((False, text[i:start]))
            # znajdź prawy delimiter
            m2 = re.search(right, text[start+1:])
            if not m2:
                # brak zamknięcia – potraktuj jako zwykły tekst
                out.append((False, text[start:]))
                break
            end = start + 1 + m2.start()
            inside = text[start+1:end]
            out.append((True, inside))
            i = end + 1
        return out

    # ---------- LaTeX render ----------
    def _insert_math_block(self, expr: str):
        self._insert_math(expr, newline_after=True)

    def _insert_math_inline(self, expr: str):
        self._insert_math(expr, newline_after=False)

    def _insert_math(self, expr: str, newline_after: bool):
        if not PIL_OK:
            # fallback: pokaż równanie tekstem
            self.text.insert("end", f"[LaTeX] {expr}")
            if newline_after:
                self.text.insert("end", "\n")
            return
        expr = sanitize_math(expr)
        try:
            fig = Figure(figsize=(0.01, 0.01), dpi=220)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            # inline vs block: dla block dajemy \displaystyle (większy)
            if newline_after:
                to_draw = r"$\displaystyle %s$" % expr
            else:
                to_draw = r"$%s$" % expr
            ax.text(0.5, 0.5, to_draw, ha="center", va="center", fontsize=12)
            canvas = FigureCanvasAgg(fig)
            buf = io.BytesIO()
            canvas.print_png(buf)
            buf.seek(0)
            img = Image.open(buf).convert("RGBA")
            bbox = img.getbbox()
            if bbox:
                img = img.crop(bbox)
            tkimg = ImageTk.PhotoImage(img)
            self._images.append(tkimg)  # silna referencja!
            self.text.image_create("end", image=tkimg)
            if newline_after:
                self.text.insert("end", "\n")
        except Exception as e:
            self.text.insert("end", f"[LaTeX render error] {type(e).__name__}: {e}\n")
