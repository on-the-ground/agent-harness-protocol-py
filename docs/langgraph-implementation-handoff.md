# LangChain + LangGraph adapter 작업 인수인계

## 목표

Python AHP Port를 실제 LangChain + LangGraph harness 구현으로 검증한다. 구현체는
`BaseChatModel`을 LangGraph `StateGraph` 안에서 실행해야 하며, conformance test를
통과시키기 위한 가짜 `AgentHarness`나 reference/mock harness를 만들지 않는다.

테스트에서 외부 API 비용과 비결정성을 제거할 필요는 있다. 이 경우 제어 가능한
LangChain `BaseChatModel`을 **네이티브 모델 경계의 test double**로 사용하는 것은
허용한다. 테스트 대상은 언제나 실제 `LangGraphHarness`, 실제 `StateGraph`, 실제
LangGraph checkpointer 경로여야 한다. fixture가 공개 task 상태를 읽어 관찰 사실을
조작하거나 task 내부를 직접 완료시키면 안 된다.

## 저장소 상태

- 저장소: `agent-harness-protocol-py`
- 기반 커밋: `43e0c25 feat: add Python AHP protocol and conformance suites`
- 현재 `main`은 `origin/main`보다 1개 커밋 앞서 있으며 아직 push하지 않았다.
- core protocol/conformance는 위 커밋에 들어 있다.
- LangGraph 구현은 아직 **untracked / 미커밋** 상태다.
- 임시 API 조사 파일 `.tmp_inspect.py`는 제거했다.

현재 생성된 파일:

```text
implementations/langgraph/
├── pyproject.toml
└── src/agent_harness_protocol_langgraph/
    ├── __init__.py
    ├── _fanout.py
    ├── harness.py
    └── py.typed
```

아직 adapter 테스트, adapter README, 상위 README/docs 갱신은 작성하지 않았다.

## 배포와 디렉터리 경계

core protocol wheel과 구현체 wheel을 분리한다.

- core: `agent-harness-protocol`
- adapter: `agent-harness-protocol-langgraph`
- Python import: `agent_harness_protocol_langgraph`
- adapter 위치: `implementations/langgraph`
- 두 패키지 버전은 현재 `0.2.3`

adapter 의존성 초안:

```toml
agent-harness-protocol==0.2.3
langchain-core>=1.2,<2
langgraph>=1.1,<2
```

이 구조는 Kotlin 저장소의 protocol/implementations 분리를 따른다. core wheel에
LangChain이나 LangGraph가 들어가면 안 된다.

로컬 `.venv`에는 다음 버전이 설치됐다.

- `langchain-core 1.6.3`
- `langgraph 1.2.11`

## 확정한 설계

### 실제 실행 경로

`LangGraphHarness`는 생성자에서 LangChain `BaseChatModel`을 받는다. 각 AHP session은
다음 실제 graph를 컴파일한다.

```text
START -> model node -> END
```

state는 LangGraph `MessagesState`, 저장은 `InMemorySaver`, 실행은 compiled graph의
`ainvoke()`를 사용한다. model node는 누적 messages 앞에 현재 `SessionSpec.instructions`
를 `SystemMessage`로 붙인 뒤 `BaseChatModel.ainvoke()`를 호출한다.

각 session은 고유 `thread_id`를 사용한다. 따라서 같은 session의 순차 task는 이전
Human/AI message를 이어 받고, 다른 session은 격리된다. `release()` 때 해당 thread의
checkpoint를 `adelete_thread()`로 지운다.

### 의도적으로 지원하지 않는 기능

현재 graph는 model-only다. 아래 기능은 구현한 척하지 않고 `SupportReport`에서
`Unsupported`, validation에서 `INCOMPATIBLE`로 거절한다.

- caller approval interaction
- caller question interaction
- harness 재생성 이후 persistence
- workspace와 skills
- filesystem/network execution constraint
- structured output validation

지원한다고 선언한 기능:

- diagnostics
- context retention (`PROVIDER_DEFAULT`, `EPHEMERAL`)
- user history visibility (`PROVIDER_DEFAULT`, `HIDDEN`)

`ApprovalRequirement.DENY_ALL`은 model-only graph에 effect/tool route가 없어서 수용한다.
그 외 approval mode는 거절한다.

### 취소의 정직성

LangChain coroutine이 취소됐다는 사실만으로 원격 provider 작업 종료가 확인됐다고
일반화하면 안 된다. 그래서 다음 공개 adapter 설정을 추가했다.

```python
CancellationSemantics.UNCONFIRMED
CancellationSemantics.COROUTINE_TERMINATION_CONFIRMS_WORK_STOPPED
```

기본값은 `UNCONFIRMED`다. 이 경우 coroutine cancellation 뒤 결과는
`Unresolved(CANCELLATION_UNCONFIRMED)`다. 로컬 모델처럼 coroutine 종료가 네이티브 작업
종료를 실제로 증명하는 구성만 두 번째 값을 선택해 `Cancelled`를 반환한다.

conformance의 명시적 취소 시나리오는 제어 모델이 cancellation을 직접 관찰하므로
두 번째 설정으로 실행한다. 이 설정을 단지 테스트를 통과시키려고 일반 provider에
기본 적용하면 안 된다.

### task 수명과 관찰

현재 초안은 다음을 구현한다.

- `start_task()`는 background task를 만들고 즉시 handle을 반환한다.
- 같은 session의 겹치는 task는 모델 호출 전에 거절한다.
- 서로 다른 session은 독립적으로 실행될 수 있다.
- `await_outcome()`은 `asyncio.shield()`를 사용해 waiter 취소가 실제 작업을 취소하지
  못하게 한다.
- terminal outcome은 한 번만 결정되고 모든 waiter에 재사용된다.
- 늦게 붙은 semantic subscriber도 terminal event 하나를 받는다.
- semantic/diagnostic subscriber마다 별도 bounded queue를 둔다.
- overflow는 각각 `ObservationGap` / `DiagnosticGap`으로 드러낸다.
- 완료 시 `AIMessage`를 `MessageCompleted`, `TextOutput`, `AgentUsage`로 변환한다.
- 예외는 자연어 문자열로 추측 분류하지 않고 우선 `FailureKind.UNKNOWN`으로 보존한다.
- `release()`와 `aclose()`는 handle을 먼저 닫고 별도 cleanup task를 shield하여 caller
  cancellation이 cleanup 자체를 중단하지 못하게 한다.

## 현재 코드에서 반드시 먼저 해결할 것

코드는 골격 단계이며 아직 실행 테스트를 하지 않았다. 직전 정적 검사 결과를 기준으로
다음이 남아 있다.

1. `harness.py` import 정렬과 unused import를 Ruff로 정리한다. `Completed` 누락은 이미
   import에 추가했다.
2. Pyright strict 오류를 정리한다.
   - LangGraph 패키지의 부분 typing/missing stub 진단
   - 인접 private 구현 클래스 사이의 접근 진단
   - compiled graph `ainvoke()` 반환 타입 narrowing
   - `_message_text()`의 불필요한 `Mapping` 검사
3. `BaseCheckpointSaver[str]` generic annotation이 설치 버전의 실제 typing과 맞는지
   확인한다.
4. graph `ainvoke()` 결과가 dict임을 안전하게 확인하고 messages를 추출한다. 타입을
   숨기기 위한 광범위한 `Any` 대신 작은 adapter 함수 하나에서 검증/narrowing한다.
5. `_fanout.close()`의 overflow 계산과 terminal + end sentinel 보존을 부하 테스트로
   확인한다. 특히 작은 capacity에서도 빈 queue를 과도하게 `get_nowait()`하지 않아야
   한다.
6. `release()` 중 `adelete_thread()`가 실패해도 active task outcome과 handle 폐쇄가
   흔들리지 않게 failure 처리 방침을 정한다. 진단으로 남기거나 cleanup 실패를 밖으로
   전달하되 task terminal 판단을 되돌리면 안 된다.
7. 명시적 cancellation 뒤 LangGraph가 입력 checkpoint를 남기는지 검사한다. session
   재사용 시 취소된 turn을 문맥에 포함할지 말지는 provider 동작을 숨기지 않도록
   disposition/docs에 기록해야 한다.
8. 사용량 누적을 검증한다. task usage가 하나라도 unknown이면 session usage의 알 수 없는
   항목도 `None`을 유지해야 한다. measured zero와 unknown을 합치면 안 된다.

Pyright 설정 초안에는 LangGraph가 제공하지 않는 stub 및 인접 private-class 진단을
끄는 항목을 넣었다. 이를 그대로 유지하기 전에 실제 우리 코드의 타입 오류를 숨기지
않는지 검토해야 한다. 가능하면 외부 라이브러리 경계만 typed wrapper로 좁히고, 내부
타입 오류는 strict로 유지한다.

## 테스트 구현 계획

### 1. 실제 runtime 공통 suite

우선 `RuntimeProfileConformanceTests`를 그대로 상속해 연결한다. 이 suite는 다음을
검증한다.

- observer가 없어도 실제 graph 완료
- system instruction과 원문 입력이 실제 model boundary에 도달
- whitespace 입력을 trim하지 않음
- 동일 session 문맥 연속성과 새 session 격리
- 동일 session overlap 거절
- 한 waiter의 취소가 native work를 취소하지 않음
- harness close가 active work를 bounded하게 종결
- 명시적 취소가 확인된 구성에서 `Cancelled`이며 session 재사용 가능
- unsupported structured output이 모델 호출 전에 거절
- semantic/diagnostic stream 종료와 late terminal 보존

fixture의 boundary는 `BaseChatModel` subclass다. 모델이 받은 `BaseMessage` 목록을 별도
관찰 기록에 저장하고, `asyncio.Event`로 모델 호출을 hold/release하며,
`AIMessage(content="native-result")`를 반환한다. fixture 관찰은 AHP task 상태가 아니라
이 모델 경계에서 얻어야 한다.

### 2. requirements 공통 suite

`RequirementsConformanceTests`를 연결한다. 최소 한 profile에 다음 case를 넣는다.

- 기본 spec + text output: compatible
- diagnostics required: compatible
- ephemeral retention: compatible
- hidden history: compatible
- deny-all approval: compatible
- caller approval: incompatible
- caller answers: incompatible
- persistence: incompatible
- workspace: incompatible
- execution constraint: incompatible
- structured output: session compatible, task incompatible
- 요청 model id mismatch: incompatible

각 rejected case 뒤에는 공통 suite가 기본 task retry를 실제 graph로 실행하므로, validation이
모델 호출 전이어야 하고 session을 오염시키지 않아야 한다.

### 3. cleanup 공통 suite

같은 실제 model boundary fixture로 `CleanupBudgetConformanceTests`를 연결한다. 4개 session의
held model call을 동시에 만들고 `aclose()` 총예산, release caller cancellation,
완료된 outcome 불변성을 검증한다.

### 4. adapter 고유 테스트

공통 suite 외에 아래 작은 테스트만 추가한다.

- 기본 cancellation semantics가 `Cancelled`가 아니라 `Unresolved`인지
- model id 일치/불일치
- `AIMessage.usage_metadata`가 AHP `AgentUsage`의 unknown과 zero를 보존하는지
- LangGraph의 실제 compiled graph/checkpointer를 거쳐 두 turn 문맥이 쌓이는지

가짜 harness를 만들어 conformance suite 자체를 테스트하지 않는다.

### 5. 이후 확장

첫 green 이후 실제 streaming을 `graph.astream(..., stream_mode="messages")`로 연결하면
`MessageDelta`와 고부하 observation suite를 검증할 수 있다. 지금 `ainvoke()` 기반 초안에
가짜 delta를 만들어 넣으면 안 된다. tool/interrupt graph를 실제로 추가하기 전에는
approval/question suite에도 연결하지 않는다.

## 문서에서 반영할 사실

구현과 테스트가 green이 된 뒤 다음을 갱신한다.

- 상위 `README.md`: core wheel은 protocol/conformance만 포함하고, 별도 LangGraph adapter가
  같은 저장소에 있다는 점
- `docs/conformance.md`: LangGraph에서 실제로 연결해 통과한 suite와 아직 지원하지 않는
  capability
- `implementations/langgraph/README.md`: 설치, 최소 예제, support matrix, cancellation
  semantics

“추후 구현체가 붙는다”는 기존 문장이 있으면 실제 현황으로 바꾼다. 통과하지 않은 suite를
통과했다고 적지 않는다.

## 검증 명령과 완료 조건

Windows 경로의 한글 때문에 `.venv/Scripts/python.exe` launcher가 비승격 실행에서 깨진
적이 있다. 필요하면 시스템 Python에 `.venv/Lib/site-packages`를 `PYTHONPATH`로 추가해
실행한다.

완료 조건:

1. core 기존 13 tests가 그대로 통과한다.
2. LangGraph adapter에 연결한 모든 공통 conformance test가 통과한다.
3. Ruff format/check가 core와 adapter 전체에서 통과한다.
4. Pyright strict가 core와 adapter 우리 코드에서 0 error다.
5. core wheel 내용에 구현체나 LangChain/LangGraph 의존성이 섞이지 않는다.
6. adapter wheel이 별도로 build되고 필요한 파일만 포함한다.
7. `git diff --check`가 통과한다.
8. 최종 diff에서 mock/reference harness가 추가되지 않았음을 확인한다.

## 공식 API 근거

- LangGraph graph API: <https://docs.langchain.com/oss/python/langgraph/graph-api>
- LangGraph persistence와 `thread_id`: <https://docs.langchain.com/oss/python/langgraph/persistence>
- LangGraph streaming: <https://docs.langchain.com/oss/python/langgraph/streaming>
- LangGraph interrupts: <https://docs.langchain.com/oss/python/langgraph/interrupts>
- LangChain short-term memory: <https://docs.langchain.com/oss/python/langchain/short-term-memory>

핵심 판단은 API 공통분모를 억지로 넓히는 것이 아니다. LangChain + LangGraph가 실제로
제공하는 목적을 AHP 의미로 옮기고, 증명하지 못하는 상태는 unsupported 또는 unresolved로
남기는 것이다.
