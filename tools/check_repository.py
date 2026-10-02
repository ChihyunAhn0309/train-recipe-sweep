"""Check source syntax, skill metadata and local documentation link targets."""
import ast
import json
from pathlib import Path
import re
from urllib.parse import unquote


def main():
    root = Path(__file__).resolve().parents[1]
    skill = (root / 'SKILL.md').read_text(encoding='utf-8')
    if not skill.startswith('---\nname: train-recipe-sweep\n'):
        raise ValueError('Missing or unexpected skill frontmatter')
    if not re.search(r'^description: .+', skill, re.MULTILINE):
        raise ValueError('Missing skill description')
    for name in ('agents/openai.yaml', 'LICENSE', 'README.md'):
        if not (root / name).is_file():
            raise ValueError('Missing required file: ' + name)
    files = [p for folder in ('scripts', 'tools') for p in (root / folder).glob('*.py')]
    for path in files:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path.relative_to(root)))
    json.loads((root / 'scripts/example-study.json').read_text(encoding='utf-8'))
    documents = [root / 'README.md', root / 'SKILL.md', root / 'CONTRIBUTING.md']
    documents += list((root / 'references').glob('*.md')) + list((root / 'docs').glob('*.md'))
    checked = 0
    for document in documents:
        content = document.read_text(encoding='utf-8')
        for target in re.findall(r'\]\(([^)]+)\)', content):
            target = target.strip('<>').split('#', 1)[0]
            if not target or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', target):
                continue
            resolved = (document.parent / unquote(target)).resolve()
            if not resolved.is_relative_to(root) or not resolved.exists():
                raise ValueError(f'Broken or escaping link in {document.name}: {target}')
            checked += 1
    print(json.dumps({'source_files_parsed': len(files), 'documents_checked': len(documents),
                      'local_links_checked': checked, 'status': 'passed',
                      'actual_gpu_training': False}, indent=2))


if __name__ == '__main__':
    main()
