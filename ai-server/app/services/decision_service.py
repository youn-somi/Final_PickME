"""결정 도우미 로직. 기존 pickme-flask 의 services/decision_service.py 를 옮긴 것."""

import logging

from google.genai import types

from app.core.config import get_settings
from app.services import gemini

logger = logging.getLogger(__name__)

# 고민 카테고리 (복수 선택 가능)
CATEGORIES = [
    "음식/식사", "진로/직업", "소비/쇼핑", "연애/관계",
    "일상/생활", "학습/자기계발", "여행/여가", "기타",
]

# AI 조언 스타일 (상담 중 변경 가능)
STYLES = {
    "soft": (
        "따뜻한 공감형: 친구 같은 어조로 먼저 충분히 공감해주고, "
        "장점 중심으로 부드럽게 조언한다."
    ),
    "objective": (
        "객관적인 분석형: 근거 기반 논리와 데이터처럼 객관적인 시각으로 "
        "균형 잡힌 조언을 한다."
    ),
    "fact": "냉정한 팩트형: 우유부단함 없이 현실적이고 냉정하게, 핵심만 간결하게 짚어준다.",
}


class ConsultationError(RuntimeError):
    """사용자에게 보여줄 수 있는 상담 실패 메시지."""


def get_ai_consultation(prompt: str) -> str:
    if not gemini.has_api_key():
        raise ConsultationError("AI 서버에 GEMINI_API_KEY가 설정되지 않았습니다.")
    try:
        return gemini.generate(
            prompt,
            models=get_settings().decision_models,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=1200,
            ),
        ).strip()
    except gemini.GeminiUnavailable as error:
        logger.warning("결정 도우미 Gemini 호출 실패: %s", error.__cause__ or error)
        raise ConsultationError(
            "AI 응답을 생성하지 못했습니다. 잠시 후 다시 시도해 주세요."
        ) from error


def build_prompt(
    categories: list[str],
    style: str,
    title: str,
    content: str,
    conditions: str,
    options: list[dict] | None = None,
    criteria: str | None = None,
) -> str:
    """카테고리(복수), 스타일, 제목, 내용, 추가 조건, 선택지, 판단 기준(가중치 포함)을 반영."""
    style_guide = STYLES.get(style, STYLES["objective"])
    cat_text = ", ".join(categories) if categories else "지정 없음"

    opt_lines = ""
    if options:
        opt_lines += "[비교할 선택지]\n"
        for i, opt in enumerate(options, 1):
            line = f"{i}. {opt['title']}"
            if opt.get("note"):
                line += f" - {opt['note']}"
            opt_lines += line + "\n"
        opt_lines += "\n"

    # 원본은 이미 만들어진 문자열을 다시 ", ".join 해서 글자 단위로 쪼개지는 버그가 있었음
    crit_text = criteria.strip() if criteria and criteria.strip() else "지정 없음"

    lines = [
        "당신은 고민 해결을 돕는 상담 전문가입니다. 아래 상담 요청에 답해주세요.",
        "답변은 500자 안팎으로 간결하게 작성해주세요.",
        "",
        f"[고민 카테고리] {cat_text}",
        f"[조언 스타일] {style_guide}",
        f"[고민 제목] {title or '제목 없음'}",
        "",
        "[고민 내용]",
        content,
        "",
        opt_lines or "[비교할 선택지] 없음\n",
        f"[판단 기준(중요도 1~5점)] {crit_text}",
        f"[추가 조건 및 특별한 상황] {conditions or '없음'}",
        "",
        "요구사항:",
        "1. 고민 내용을 바탕으로 상황을 짧게 정리해주세요.",
        "2. 선택지가 있다면 선택지별 장단점을 현실적으로 비교해주세요.",
        "3. 판단 기준이 있다면 중요도가 높은 기준을 우선해 평가해주세요.",
        "4. 지정된 조언 스타일의 말투와 태도를 유지해주세요.",
        "5. 미래나 관계의 결과를 단정하지 말고, 판단 재료를 제시하는 방식으로 조언해주세요.",
        "6. 마지막에 한 줄 추천 결론을 정리해주세요.",
        "7. 선택하지 않은 다른 대안을 포기함으로써 얻거나 잃게 되는 기회비용 분석도 함께 포함해주세요.", #기회비용분석 기능 추가 
        "   - 각 선택지를 선택했을 때 얻는 이점을 설명해주세요.",
        "   - 선택하지 않은 대안에서 포기하게 되는 가장 중요한 가치를 설명해주세요.",
        "   - 선택지 A와 B 양쪽 모두 동일한 기준으로 비교해주세요.",
        "   - 단순히 포기한 대안의 이름만 나열하지 말고, 무엇을 얻지 못하는지 설명해주세요.",
        "   - 사용자가 제공하지 않은 선호도나 사실을 임의로 만들어내지 마세요.",
        "8. 객관적인 사실과 개인적인 의견 및 추측을 구분해서 설명해주세요.",
        "9. 사용자가 제공하지 않은 정보나 확인되지 않은 효과를 사실처럼 단정하지 마세요.",
        "10. 근거가 부족한 경우에는 추측이라고 밝히고, 확인할 수 없는 내용은 모른다고 답해주세요.",
        "11. 사용자의 선호도와 객관적인 판단 기준이 다를 경우 각각 구분해서 분석해주세요.",
        "12. 특정 선택지를 추천할 때는 추천 이유와 판단 근거를 함께 제시해주세요.",    
    ]
    return "\n".join(lines)
