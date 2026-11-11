from app.orchestrator import Orchestrator, Session


def _commands(session: Session) -> list[str]:
    orchestrator = object.__new__(Orchestrator)
    return [command for _, command in orchestrator._next_actions(session)]


def test_stale_waiting_state_keeps_quiz_launch_available() -> None:
    session = Session(
        user_id="test",
        phase="waiting_answers",
        current_kc_id="g_k5",
        quiz_questions={},
        diagnostic_done=True,
    )

    assert "quiz" in _commands(session)


def test_active_quiz_is_reachable_from_lesson() -> None:
    session = Session(
        user_id="test",
        phase="waiting_answers",
        current_kc_id="g_k5",
        quiz_questions={"1": {"number": 1}},
        diagnostic_done=True,
    )

    orchestrator = object.__new__(Orchestrator)
    assert "quiz" not in _commands(session)
    assert "quiz" in [command for _, command in orchestrator._lesson_actions(session)]
