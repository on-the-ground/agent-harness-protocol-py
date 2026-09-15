# PyPI 릴리스 절차

이 저장소는 배포본 두 개를 만든다.

| 배포본 | 위치 | 태그 |
|---|---|---|
| `agent-harness-protocol` | 저장소 루트 | `v<version>` |
| `agent-harness-protocol-langgraph` | `implementations/langgraph` | `langgraph-v<version>` |

adapter는 `agent-harness-protocol==<version>`에 고정돼 있으므로 **항상 core를 먼저**
올린다. PyPI는 같은 파일 이름을 다시 받지 않으므로, 올린 뒤 고칠 것이 생기면 버전을
올려야 한다.

## 사전 조건

- GitHub 저장소 `on-the-ground/agent-harness-protocol-py`가 공개돼 있어야 한다. 두
  배포본의 `Project-URL`과 PyPI에 표시되는 README 링크가 이 저장소를 가리킨다.
- PyPI 계정과 API token, 또는 Trusted Publishing 설정. 두 프로젝트 이름은 첫 업로드 때
  생성된다.
- 버전은 두 곳에서 같이 올린다: `pyproject.toml`의 `version`, 패키지 `__init__.py`의
  `__version__`. adapter는 core 고정 버전(`agent-harness-protocol==...`)도 함께 바꾼다.

## 1. 검증

```bash
# core (저장소 루트)
python -m pip install -e ".[test]" ruff pyright
python -m pytest
ruff check . && ruff format --check .
pyright

# adapter
cd implementations/langgraph
python -m pip install -e "../..[test]" -e ".[test]"
python -m pytest -W error
ruff check . && ruff format --check .
pyright
```

protocol과 conformance suite는 스펙이다. 릴리스 준비 중에 수정하지 않는다.

## 2. 빌드와 점검

```bash
python -m pip install build twine
python -m build . --outdir dist
python -m build implementations/langgraph --outdir implementations/langgraph/dist
python -m twine check --strict dist/* implementations/langgraph/dist/*
```

확인할 것:

- core wheel에는 `agent_harness_protocol/`만, 의존성은 없음(`test` extra 제외)
- core sdist에 `implementations/`와 `docs/langgraph-*.md`가 없음
- adapter wheel에는 `agent_harness_protocol_langgraph/`만, `py.typed` 포함
- 두 배포본 모두 `License-Expression: Apache-2.0`과 `LICENSE` 포함

## 3. 설치 검증

빌드한 wheel을 새 환경에 설치하고, 소스가 아닌 설치본으로 테스트를 돌린다.
`tests/`와 `implementations/langgraph/tests/`를 저장소 밖 임시 디렉터리로 복사한 뒤
그 디렉터리에서 실행한다.

```bash
python -m venv /tmp/ahp-check
/tmp/ahp-check/bin/python -m pip install --find-links dist \
  "implementations/langgraph/dist/agent_harness_protocol_langgraph-<version>-py3-none-any.whl[test]"
/tmp/ahp-check/bin/python -m pytest -o asyncio_mode=auto \
  -o asyncio_default_fixture_loop_scope=function <복사한 core tests> <복사한 adapter tests>
```

0.2.3에서 확인한 조합:

| Python | langchain-core | langgraph | 결과 |
|---|---|---|---|
| 3.11, 3.12, 3.13 | 1.6.3 | 1.2.11 | 99 passed |
| 3.11 | 1.2.8 (하한) | 1.1.0 (하한) | 99 passed |

`langchain-core` 하한이 1.2.8인 이유: `langgraph` 1.1.0을 설치해도 resolver는 최신
`langgraph-checkpoint`(4.2.0)를 고르는데, 이 버전은 `langchain-core` 1.2.7 이하에서
import 단계부터 실패한다.

## 4. TestPyPI

```bash
python -m twine upload --repository testpypi dist/*
python -m twine upload --repository testpypi implementations/langgraph/dist/*

python -m venv /tmp/ahp-testpypi
/tmp/ahp-testpypi/bin/python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  agent-harness-protocol-langgraph==<version>
/tmp/ahp-testpypi/bin/python -c "import agent_harness_protocol_langgraph as m; print(m.__version__)"
```

TestPyPI 페이지에서 README 렌더링과 링크를 확인한다.

## 5. PyPI

```bash
python -m twine upload dist/*
python -m twine upload implementations/langgraph/dist/*

git tag v<version>
git tag langgraph-v<version>
git push origin main v<version> langgraph-v<version>
```

core 업로드가 PyPI에 반영된 것을 확인한 뒤 adapter를 올린다.
