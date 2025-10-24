import zipfile
import xml.etree.ElementTree as ET
import pandas as pd

def read_dataset(path: str) -> pd.DataFrame:
    low = path.lower()
    if low.endswith(".csv"):
        return pd.read_csv(path)

    # excel via openpyxl
    try:
        df = pd.read_excel(path, engine="openpyxl")
        if isinstance(df, pd.DataFrame) and df.shape[1] > 0:
            return df
    except Exception:
        pass

    try:
        xls = pd.ExcelFile(path, engine="openpyxl")
        if len(xls.sheet_names) > 0:
            df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
            if isinstance(df, pd.DataFrame) and df.shape[1] > 0:
                return df
    except Exception:
        pass

    return _custom_xlsx_to_df(path)


def _custom_xlsx_to_df(xlsx_path):
    def strip_tag(tag):
        return tag.split("}", 1)[1] if "}" in tag else tag

    z = zipfile.ZipFile(xlsx_path)

    shared_strings = []
    if "xl/sharedStrings.xml" in z.namelist():
        ss = z.read("xl/sharedStrings.xml").decode("utf-8", errors="ignore")
        root = ET.fromstring(ss)
        for si in root:
            texts = []
            for elem in si.iter():
                if strip_tag(elem.tag) == "t" and elem.text is not None:
                    texts.append(elem.text)
            shared_strings.append("".join(texts))

    sheet_name = None
    for n in z.namelist():
        if n.startswith("xl/worksheets/") and n.endswith(".xml"):
            sheet_name = n
            break
    if sheet_name is None:
        raise ValueError("Brak arkusza w XLSX (fallback).")

    xml_data = z.read(sheet_name).decode("utf-8", errors="ignore")
    root = ET.fromstring(xml_data)

    rows_data = {}
    max_col = 0
    for row in root.iter():
        if strip_tag(row.tag) == "row":
            r_idx = int(row.attrib.get("r", "0"))
            cells = {}
            for c in row:
                if strip_tag(c.tag) != "c":
                    continue
                ref = c.attrib.get("r")
                col_letters = "".join(ch for ch in ref if ch.isalpha()) if ref else None
                col_idx = 0
                if col_letters:
                    for ch in col_letters:
                        col_idx = col_idx * 26 + (ord(ch.upper()) - ord('A') + 1)

                t = c.attrib.get("t")
                v_text = None
                for child in c:
                    if strip_tag(child.tag) == "v":
                        v_text = child.text
                        break

                if t == "s" and v_text is not None:
                    si = int(v_text)
                    val = shared_strings[si] if 0 <= si < len(shared_strings) else ""
                else:
                    val = v_text
                if val is not None:
                    try:
                        if isinstance(val, str) and ('.' in val or 'e' in val.lower()):
                            val = float(val)
                        else:
                            val = int(val)
                    except Exception:
                        pass

                cells[col_idx] = val
                max_col = max(max_col, col_idx)
            rows_data[r_idx] = cells

    row_indices = sorted(rows_data.keys())
    table = []
    for r in row_indices:
        row = [rows_data[r].get(c, None) for c in range(1, max_col + 1)]
        table.append(row)

    header = None
    data_start = 0
    for i, row in enumerate(table):
        if any(x is not None for x in row):
            header = [str(x).strip() if x is not None else "" for x in row]
            data_start = i + 1
            break

    data = table[data_start:]
    df = pd.DataFrame(data, columns=header)
    return df


def select_numeric_features(df: pd.DataFrame, target_col: str) -> list[str]:
    return [c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
