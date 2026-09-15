# LangGraph adapter 설계 기록

`implementations/langgraph`에서 마주친 문제와, 어떤 의도로 구현했는지만 짧게 남긴다.
protocol과 conformance 계약은 바꾸지 않았다.

## 1. terminal 판단은 graph 실행이 끝나는 한 곳에서 확정

- 문제: 시작 직후 취소되면 worker 본문이 실행되지 않아 outcome이 영원히 비었다. 여러
  경로가 각자 outcome을 만들면서 session usage가 두 번 더해졌다.
- 의도: 결과는 한 번만, 증거가 있는 곳에서 확정한다.
- 구현: worker의 done-callback이 성공/실패/취소를 판정해 settle한다. 예산 초과 같은
  강제 판정만 예외적으로 먼저 settle하고, 이후 callback은 아무것도 바꾸지 않는다.
  session usage는 settle 시점에 한 번만 누적한다. 모델에 도달하기 전 취소는 native
  작업이 없었으므로 `Cancelled`, 사용량은 측정된 0이다.

## 2. 취소는 한 번만 보내고, 끝을 확인하지 못하면 Unresolved

- 문제: 취소를 무시하는 모델이면 LangGraph `ainvoke()`는 모델이 끝날 때까지 반환하지
  않는다. 취소를 반복하면 모델의 정리 작업까지 끊는다.
- 의도: 호출자는 예산 안에 답을 받고, 확인하지 못한 종료를 확인했다고 말하지 않는다.
- 구현: 취소는 task당 한 번만 전달한다. `per_task` 예산 안에 graph 실행이 끝나지
  않으면 `Unresolved`로 확정한다. 호출자 자신의 취소는 그대로 전파한다.
- 모델 호출 중 취소의 결과는 `CancellationSemantics`가 정한다. 기본값은
  `Unresolved(CANCELLATION_UNCONFIRMED)`다.

## 3. 끝나지 않은 graph 실행이 있으면 session을 막음

- 문제: task가 `Unresolved`로 확정돼도 이전 graph 실행이 같은 `thread_id`에 계속 쓸 수 있다.
- 의도: 한 대화 기록에 두 실행이 동시에 쓰지 않게 한다.
- 구현: 이전 실행이 실제로 끝날 때까지 `start_task()`는 `SessionBlockedError`를
  던진다. 끝나면 다시 사용할 수 있다. 진행 중인 task와 겹치는 요청은 기존대로
  `RuntimeError`다.

## 4. 이벤트 버퍼는 손실 위치를 보존

- 문제: 기존 fanout은 가득 찬 큐를 닫을 때 `QueueFull`/`QueueEmpty`로 터졌고, 그 예외가
  settle을 막았다.
- 의도: 느린 구독자도 정확한 손실 수와 마지막 terminal 하나를 받는다.
- 구현: 구독자마다 `capacity`개 항목의 버퍼를 둔다. 연속으로 버린 항목은 gap 하나가
  한 칸을 차지한다. 가득 차면 최신 항목을 버리고, 버린 개수를 그 위치의 gap에 센다.
  terminal은 버퍼 밖에 보관해 닫을 때 기존 항목을 밀어내지 않는다. 구독은 `events()`
  호출 시점에 등록된다.

## 5. checkpoint 삭제 실패는 숨기지 않되 task 판단은 건드리지 않음

- 문제: release 중 `adelete_thread()`가 실패하면 예외가 새고 session이 harness에 남았다.
  삭제가 멈추면 cleanup 예산도 넘겼다.
- 의도: 저장된 문맥이 남았다는 사실은 알리고, task 결과와 handle 폐쇄는 흔들지 않는다.
- 구현: 삭제는 남은 `total` 예산 안에서만 시도하고, 실패하면 `CheckpointCleanupError`를
  던진다. 그래도 session은 닫히고 harness에서 빠진다. `aclose()`는 모든 task를 정리한
  뒤 실패를 모아 한 번에 던진다. cleanup 뒤에도 graph 실행이 남아 있으면, 그 실행이
  끝날 때 해당 thread를 한 번 더 지운다. 이 백그라운드 삭제는 실패를 보고할 경로가 없다.

## 6. 취소된 turn의 입력은 문맥에 남김

- 사실: LangGraph는 model node 실행 전에 입력 message를 checkpoint에 기록한다. 그래서
  취소된 turn의 입력이 다음 task의 문맥에 포함된다. 대응하는 AI 응답은 없다.
- 결정: provider 동작을 숨기지 않고 그대로 유지하며 문서로 알린다.

## 7. disposition은 어댑터가 소유하거나 선언받은 만큼만 보고

- 문제: 항상 `EPHEMERAL`/`HIDDEN`으로 보고했다. 그런데 외부 checkpointer의 영속성은 알 수
  없고(`InMemorySaver(factory=PersistentDict)`도 디스크에 쓴다), 입력의 외부 노출은
  provider 로그나 LangSmith tracing에 달려 있다.
- 의도: 모르는 것은 모른다고 말하고, 요구사항은 조용히 낮추지 않는다.
- 구현: 생성자의 `context_retention`, `history_visibility`로 선언받는다.

| 설정 | retention | visibility |
|---|---|---|
| checkpointer 미지정 | `EPHEMERAL` (어댑터 소유, 다른 값 선언 시 `ValueError`) | 선언값, 기본 `UNKNOWN` |
| checkpointer 지정 | 선언값, 기본 `UNKNOWN` | 선언값, 기본 `UNKNOWN` |

선언은 `support`, `validate`, `session.disposition`에 같게 반영된다.

| 선언 | support | `EPHEMERAL`/`HIDDEN` 요구 |
|---|---|---|
| `EPHEMERAL` / `HIDDEN` | `Supported` | `COMPATIBLE` |
| `MATERIALIZED` / `VISIBLE` | `Unsupported` | `INCOMPATIBLE` |
| `UNKNOWN` | `UnknownSupport` | `UNCONFIRMED` |

## 8. conformance suite 연결 방식

- 연결한 suite: `RuntimeProfileConformanceTests`, `RequirementsConformanceTests`,
  `CleanupBudgetConformanceTests` (`implementations/langgraph/tests/test_conformance.py`).
- 대체한 것은 모델 경계뿐이다. `ControlledChatModel`(LangChain `BaseChatModel`)이 받은
  message를 기록하고, 관찰은 모두 그 기록에서 읽는다. harness, `StateGraph`,
  checkpointer는 실제 구현이다.
- Runtime suite는 확인된 취소(`Cancelled`)를 요구한다. 테스트 모델에서는 모델
  coroutine이 native 작업의 전부이므로 `COROUTINE_TERMINATION_CONFIRMS_WORK_STOPPED`를
  쓰는 것이 사실과 맞다. 기본값 `UNCONFIRMED`는 cleanup suite와 어댑터 테스트가 검증한다.
- Cleanup suite는 세 구성으로 돌린다: 확인된 취소, 미확인 취소, 취소를 무시하는 모델.
- Requirements suite는 disposition 선언 네 가지(어댑터 소유 / 미선언 외부 checkpointer /
  `EPHEMERAL`+`HIDDEN` 선언 / `MATERIALIZED`+`VISIBLE` 선언)를 profile로 둔다.
- 일부러 어댑터를 망가뜨려(instructions 누락, approval 수용, `HIDDEN` 무조건 수용, 문맥
  분리, cleanup 예산 무시) 각 suite가 실패하는 것을 확인했다.
- streaming(`MessageDelta`), approval/question, persistence suite는 해당 경로가 graph에
  없으므로 연결하지 않았다.

## 9. 타입 경계와 배포

- 문제: LangGraph 1.x의 `langgraph.graph`는 타입 정보가 불완전하다. 초안은 이를 가리려고
  Pyright 규칙 다섯 개를 어댑터 전체에서 꺼 두었고, 그 사이 인접 클래스끼리 private 속성을
  직접 읽는 코드가 숨어 있었다.
- 의도: 외부 라이브러리의 빈틈만 좁은 경계에서 허용하고, 우리 코드는 strict로 검사한다.
- 구현: graph 생성과 실행을 `_graph.ModelOnlyGraph`로 옮기고, 규칙 완화는 그 파일 상단에만
  둔다. harness/session/task가 공유하는 상태는 `_Runtime` 객체로 모아 private 접근을
  없앴다. 설정의 전역 완화는 모두 제거했고, 남은 예외는 테스트의 `InMemorySaver`
  생성자 한 줄이다. core가 venv에 설치되지 않아도 검사되도록 `extraPaths`로 소스를 읽는다.
- 배포 확인: core wheel에는 `agent_harness_protocol`만 있고 의존성이 없다. adapter wheel은
  `agent_harness_protocol_langgraph`만 담고 `agent-harness-protocol==0.2.3`,
  `langchain-core`, `langgraph`에 의존한다. 두 wheel을 새 venv에 설치해 실제 모델
  (`FakeListChatModel`)로 두 turn을 실행했다.
