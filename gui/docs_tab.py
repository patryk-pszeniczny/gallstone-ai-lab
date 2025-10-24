# ui/docs_tab.py (lub gdzie masz DocsTab)
import tkinter as tk
from tkinter import ttk, filedialog
from assets.docs_text import DOCS_TEXT
from .base import BaseTab
from .markdown_viewer import MarkdownViewer

class DocsTab(BaseTab):
    def build(self):
        bar = ttk.Frame(self.tab); bar.pack(fill="x", pady=(0,6))
        ttk.Label(bar, text="Szukaj:").pack(side="left")
        self.doc_query = tk.StringVar()
        ent = ttk.Entry(bar, textvariable=self.doc_query, width=32); ent.pack(side="left", padx=6)

        ttk.Button(bar, text="Szukaj", command=lambda: self._search(False)).pack(side="left")
        ttk.Button(bar, text="Od początku", command=lambda: self._search(True)).pack(side="left", padx=(4,0))
        ttk.Button(bar, text="Kopiuj wszystko", command=self._copy_all).pack(side="right")
        ttk.Button(bar, text="Zapisz do pliku…", command=self._save_file).pack(side="right", padx=(0,8))

        # ——— UŻYJ VIEWERA ———
        self.viewer = MarkdownViewer(self.tab)
        self.viewer.pack(fill="both", expand=True)

        # Render na starcie
        self.viewer.render(DOCS_TEXT)

    def _search(self, from_start=False):
        # proste „find” po plain-tekście w Text; działa też z obrazkami,
        # bo markery pozycji nie dotyczą obrazów – przeszukujemy to co jest tekstem
        t = self.viewer.text
        t.tag_remove("hit", "1.0", "end")
        q = self.doc_query.get().strip()
        if not q:
            return
        start = "1.0" if from_start else t.index("insert")
        pos = t.search(q, start, nocase=True, stopindex="end")
        if pos:
            end = f"{pos}+{len(q)}c"
            t.tag_add("hit", pos, end)
            t.tag_config("hit", background="#fff29b")
            t.mark_set("insert", end)
            t.see(pos)

    def _copy_all(self):
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(DOCS_TEXT)
            self.root.update_idletasks()
            self.info("Dokumentacja", "Skopiowano całą dokumentację do schowka.")
        except Exception as e:
            self.error("Błąd", f"Nie udało się skopiować: {e}")

    def _save_file(self):
        path = filedialog.asksaveasfilename(defaultextension=".md",
                                            filetypes=[("Markdown","*.md"),("TXT","*.txt"),("All files","*.*")],
                                            parent=self.root)
        if not path: return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(DOCS_TEXT)
            self.info("Dokumentacja", f"Zapisano dokumentację do:\n{path}")
        except Exception as e:
            self.error("Błąd", f"Nie udało się zapisać: {e}")
