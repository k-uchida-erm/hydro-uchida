#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import difflib
from datetime import datetime
from typing import Optional, List, Set


def list_files(root: Path):
    for p in sorted(root.rglob('*')):
        if p.is_file():
            # 正規化: 絶対パスではなく相対に
            yield p.relative_to(root)


def make_diff(base_dir: Path, edited_dir: Path, files_filter: Optional[List[str]] = None) -> str:
    lines = []
    if files_filter is not None:
        all_files = [Path(p) for p in files_filter]
    else:
        base_files = set(list_files(base_dir))
        edited_files = set(list_files(edited_dir))
        all_files = sorted(base_files | edited_files)
    for rel in all_files:
        a = base_dir / rel
        b = edited_dir / rel
        if a.exists():
            a_text = a.read_text(errors='ignore').splitlines(keepends=True)
        else:
            a_text = []
        if b.exists():
            b_text = b.read_text(errors='ignore').splitlines(keepends=True)
        else:
            b_text = []
        if a_text == b_text:
            continue
        diff = difflib.unified_diff(
            a_text,
            b_text,
            fromfile=str(Path('BASE')/rel),
            tofile=str(Path('EDITED')/rel),
        )
        lines.extend(diff)
        if lines and not lines[-1].endswith('\n'):
            lines[-1] += '\n'
    return ''.join(lines)


def diff_stats(base_dir: Path, edited_dir: Path, only_files: Optional[Set[str]] = None) -> str:
    changed = 0
    added = 0
    removed = 0
    base_files = set(list_files(base_dir))
    edited_files = set(list_files(edited_dir))
    all_files = sorted(base_files | edited_files)
    for rel in all_files:
        rel_str = str(rel)
        if only_files is not None and rel_str not in only_files:
            continue
        a = base_dir / rel
        b = edited_dir / rel
        if a.exists():
            a_text = a.read_text(errors='ignore').splitlines()
        else:
            a_text = []
        if b.exists():
            b_text = b.read_text(errors='ignore').splitlines()
        else:
            b_text = []
        if a_text == b_text:
            continue
        changed += 1
        sm = difflib.SequenceMatcher(a=a_text, b=b_text)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'insert':
                added += (j2 - j1)
            elif tag == 'delete':
                removed += (i2 - i1)
            elif tag == 'replace':
                removed += (i2 - i1)
                added += (j2 - j1)
    return f"Changed files: {changed}, +{added} / -{removed}\n"


def _read_text_lines(path: Path):
    return path.read_text(errors='ignore').splitlines(keepends=False)


def _should_ignore(rel_path: Path) -> bool:
    parts = set(rel_path.parts)
    if '__pycache__' in parts:
        return True
    suffix = rel_path.suffix.lower()
    if suffix in {'.pyc', '.png', '.pt', '.npy'}:
        return True
    # ignore large artifacts or plots directories just in case
    if 'plots' in parts or 'result' in parts:
        return True
    return False


def compute_changed_files(base_dir: Path, edited_dir: Path, since: Optional[datetime] = None):
    changed_files = []
    base_files = set(list_files(base_dir))
    edited_files = set(list_files(edited_dir))
    all_files = sorted(base_files | edited_files)
    for rel in all_files:
        a = base_dir / rel
        b = edited_dir / rel
        if _should_ignore(rel):
            continue
        # 変更対象は edited に実体があるもののみ（base にしか無い=除外）
        if not b.exists() or not b.is_file():
            continue
        if since is not None:
            try:
                if datetime.fromtimestamp(b.stat().st_mtime) < since:
                    continue
            except Exception:
                pass
        a_text = _read_text_lines(a) if a.exists() else []
        b_text = _read_text_lines(b)
        if a_text != b_text:
            changed_files.append(str(rel))
    return changed_files


def summarize_changed_files_section(files: List[str], max_files: int = 200) -> str:
    lines = []
    lines.append('[Changed files]\n')
    if not files:
        lines.append('(no changes)\n')
        return ''.join(lines)
    total = len(files)
    show = files[:max_files]
    for p in show:
        lines.append(f'- {p}\n')
    if total > len(show):
        lines.append(f'... and {total - len(show)} more\n')
    return ''.join(lines)


def summarize_hunks_from_patch(patch_text: str, allowed_files: Set[str], max_files: int = 10, max_hunks_per_file: int = 2, max_lines_per_hunk: int = 60) -> str:
    lines = ['\n[Key hunks]\n']
    file_count = 0
    current_file = None
    hunks_for_file = 0
    collecting = False
    hunk_buffer = []

    def flush_hunk():
        nonlocal hunks_for_file, hunk_buffer
        if hunk_buffer:
            # trim hunk if too long
            if len(hunk_buffer) > max_lines_per_hunk:
                cut = hunk_buffer[:max_lines_per_hunk]
                cut.append('...\n')
                lines.extend(cut)
            else:
                lines.extend(hunk_buffer)
            hunk_buffer = []
            hunks_for_file += 1

    for raw in patch_text.splitlines(keepends=True):
        if raw.startswith('+++ '):
            # new file section
            # finalize previous file
            collecting = False
            hunks_for_file = 0
            current_file = None
            if 'EDITED/' in raw:
                candidate = raw.split('EDITED/', 1)[1].strip()
                # このファイルが許可リストに無ければ対象外
                if candidate in allowed_files:
                    current_file = candidate
            continue
        if raw.startswith('@@'):
            if current_file is None:
                continue
            if file_count >= max_files:
                break
            if hunks_for_file == 0:
                # first hunk for this file: print header
                lines.append(f'--- {current_file}\n')
                file_count += 1
            if hunks_for_file >= max_hunks_per_file:
                collecting = False
                continue
            # start new hunk
            collecting = True
            hunk_buffer.append(raw)
            continue
        if collecting:
            # stop if next file header appears
            if raw.startswith('--- ') and 'BASE/' in raw:
                # end of hunk before next headers
                flush_hunk()
                collecting = False
                continue
            # normal hunk content
            hunk_buffer.append(raw)
            # if we reach a blank line separating hunks, flush
            if raw.strip() == '':
                flush_hunk()

    # flush last hunk if any
    if collecting:
        flush_hunk()
    return ''.join(lines)


def append_analysis_summary(result_dir: Path, out_readme: Path):
    # 最新モデルを探して analysis_results.txt を抜粋
    models = sorted(result_dir.glob('model_*'))
    if not models:
        return
    latest = models[-1]
    report = latest / 'analysis_results.txt'
    if not report.exists():
        return
    lines = report.read_text(errors='ignore').splitlines()
    keep = []
    flag = False
    for ln in lines:
        if ln.startswith('[Overall') or ln.startswith('[IC') or ln.startswith('[Per-time'):
            flag = True
            keep.append(ln)
            continue
        if flag:
            keep.append(ln)
    with out_readme.open('a') as f:
        f.write('\n[Key metrics]\n')
        for ln in keep:
            f.write(ln + '\n')


def _read_lock_timestamp(lock_path: Path) -> Optional[datetime]:
    try:
        txt = lock_path.read_text().strip()
        # format: YYYY-MM-DD HH:MM:SS
        return datetime.strptime(txt, '%Y-%m-%d %H:%M:%S')
    except Exception:
        return None


def _clean_previous_sections(readme_path: Path):
    if not readme_path.exists():
        return
    text = readme_path.read_text(errors='ignore')
    # 最初の [Diff summary] 以降を削除して上書き（冪等化）
    idx = text.find('\n[Diff summary]')
    if idx == -1:
        return
    trimmed = text[:idx]
    readme_path.write_text(trimmed, encoding='utf-8')


def _read_started_timestamp_from_readme(readme_path: Path) -> Optional[datetime]:
    try:
        txt = readme_path.read_text(errors='ignore')
        for line in txt.splitlines():
            if line.startswith('Started: '):
                ts = line.split('Started: ', 1)[1].strip()
                return datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='Path to backup(base) dir')
    ap.add_argument('--edited', required=True, help='Path to edited dir')
    ap.add_argument('--result_dir', required=True, help='Path to run result dir (contains model_*)')
    ap.add_argument('--out', required=True, help='Output ablation case dir')
    ap.add_argument('--save-patch', action='store_true', help='Write patch.diff to disk (default: off)')
    ap.add_argument('--since', required=False, help='YYYY-MM-DD HH:MM:SS; if provided, only changes after this time are reported')
    args = ap.parse_args()

    base_dir = Path(args.base)
    edited_dir = Path(args.edited)
    result_dir = Path(args.result_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 既存のREADMEの差分セクションをクリア（冪等化）
    readme = out_dir / 'README.txt'
    if not readme.exists():
        readme.write_text('')
    _clean_previous_sections(readme)

    # ベースライン時刻（優先度: --since > .lock > READMEのStarted行）
    since_ts: Optional[datetime] = None
    if args.since:
        try:
            since_ts = datetime.strptime(args.since, '%Y-%m-%d %H:%M:%S')
        except Exception:
            since_ts = None
    if since_ts is None:
        since_ts = _read_lock_timestamp(out_dir / '.lock')
    if since_ts is None:
        since_ts = _read_started_timestamp_from_readme(readme)

    # 変更ファイル（edited に存在し、開始時刻以後に変更、かつ内容差分あり）
    changed_files = compute_changed_files(base_dir, edited_dir, since=since_ts)

    # patch（ハンク抽出用、テキスト対象の変更ファイルに限定）
    patch = make_diff(base_dir, edited_dir, files_filter=changed_files)
    if args.save_patch:
        (out_dir / 'patch.diff').write_text(patch)

    # README append
    readme = out_dir / 'README.txt'
    with readme.open('a') as f:
        f.write('\n[Diff summary]\n')
        f.write(diff_stats(base_dir, edited_dir, only_files=set(changed_files)))
        f.write('\n')
        f.write(summarize_changed_files_section(changed_files))
        f.write('\n')
        f.write(summarize_hunks_from_patch(patch, allowed_files=set(changed_files)))

    append_analysis_summary(result_dir, readme)


if __name__ == '__main__':
    main()


