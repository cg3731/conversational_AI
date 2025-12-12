#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import statsapi

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory


# =========================
# Dataset (Play-by-play)
# =========================

PLAY_BY_PLAY_CSV_PATH = os.getenv(
    "PLAY_BY_PLAY_CSV_PATH",
    os.path.join(os.path.dirname(__file__), "dataset", "play_by_play_2017.csv"),
)
_PBP_CACHE: Optional[pd.DataFrame] = None


def _load_play_by_play() -> pd.DataFrame:
    """
    play_by_play CSV를 읽어 들이고 재사용하기 위해 메모리에 캐싱합니다.
    """
    global _PBP_CACHE
    if _PBP_CACHE is not None:
        return _PBP_CACHE

    if not os.path.exists(PLAY_BY_PLAY_CSV_PATH):
        raise FileNotFoundError(f"play_by_play csv not found: {PLAY_BY_PLAY_CSV_PATH}")

    use_cols = [
        "Inning",
        "Score",
        "Outs",
        "RoB",
        "Pitches..Count.",
        "Pitch.Sequence",
        "Team",
        "Batter",
        "Pitcher",
        "WPA",
        "WE",
        "Play.Description",
        "Game.ID",
    ]
    try:
        df = pd.read_csv(PLAY_BY_PLAY_CSV_PATH, usecols=use_cols)
    except Exception:
        df = pd.read_csv(PLAY_BY_PLAY_CSV_PATH)

    _PBP_CACHE = df
    return df


# =========================
# Helpers
# =========================

SPECIAL_KEYWORDS: Dict[str, List[str]] = {
    "home_run": ["home run", "homers", "grand slam"],
    "triple_play": ["triple play"],
    "double_play": ["double play"],
    "error": ["error"],
    "wild_pitch": ["wild pitch"],
    "passed_ball": ["passed ball"],
    "steal": ["stole", "stolen base"],
    "caught_stealing": ["caught stealing"],
    "pickoff": ["pickoff"],
    "injury": ["injured", "injury"],
    "review": ["challenge", "reviewed", "replay"],
    "hit_by_pitch": ["hit by pitch"],
}


def _parse_score(score_text: Any) -> Optional[Tuple[int, int]]:
    """
    Score가 '1-0' 형태일 때 (away, home)로 파싱
    """
    if not isinstance(score_text, str):
        return None
    try:
        left, right = score_text.split("-")
        return int(left), int(right)
    except Exception:
        return None


def _parse_percent(value: Any) -> Optional[float]:
    """
    '5.2%' or '5.2' -> 5.2
    """
    if value is None:
        return None
    text = str(value).strip().replace("%", "")
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _detect_hit_and_walk(desc: Any) -> Tuple[bool, bool]:
    """
    (안타 여부, 볼넷/사구 여부)
    """
    if not isinstance(desc, str):
        return False, False
    d = desc.lower()

    # 더블플레이는 안타로 카운트하지 않음(단순 휴리스틱)
    if "double play" in d:
        is_hit = False
    else:
        hit_keywords = ["single", "double", "triple", "home run", "homers", "homered"]
        is_hit = any(k in d for k in hit_keywords)

    is_walk = ("walk" in d) or ("hit by pitch" in d)
    return is_hit, is_walk


def _parse_inning_hint(text: Any) -> Tuple[Optional[int], Optional[str]]:
    """
    사용자 입력(end_inning 등)에서 이닝 번호 + 초/말 힌트를 파싱
    - '3' / '3회' / '3회말' / 'b3' / 't4' / 'top 3' / 'bottom 6' ...
    return: (inning_num, half) where half in {'t','b',None}
    """
    if text is None:
        return None, None
    s = str(text).strip().lower()
    m = re.search(r"(\d+)", s)
    inning_num = int(m.group(1)) if m else None

    half = None
    if s.startswith("t") or "top" in s or "초" in s:
        half = "t"
    elif s.startswith("b") or "bot" in s or "bottom" in s or "말" in s:
        half = "b"

    return inning_num, half


def _inning_key(inning_label: Any) -> Optional[Tuple[int, int]]:
    """
    't3' -> (3,0), 'b3' -> (3,1)
    """
    if not isinstance(inning_label, str):
        return None
    val = inning_label.strip().lower()
    if len(val) < 2:
        return None
    half = val[0]
    if half not in ("t", "b"):
        return None
    try:
        num = int(val[1:])
    except Exception:
        return None
    return (num, 0) if half == "t" else (num, 1)


def _label_from_key(key: Tuple[int, int]) -> str:
    inning_num, half_i = key
    return ("t" if half_i == 0 else "b") + str(inning_num)


def _next_half_inning(end_key: Tuple[int, int]) -> Tuple[int, int]:
    """
    end_key=(inning, halfIndex)에서 다음 half-inning 계산
    - (3,1)=3회말 끝 -> (4,0)=4회초
    - (3,0)=3회초 끝 -> (3,1)=3회말
    """
    inning_num, half_i = end_key
    if half_i == 0:
        return (inning_num, 1)
    return (inning_num + 1, 0)


# =========================
# Tools (optional stats: HW2 reuse)
# =========================

@tool
def find_player_id(player_name: str) -> Dict:
    """
    선수 이름(예: 'Shohei Ohtani') -> MLB Stats API로 player_id 조회
    """
    try:
        if not isinstance(player_name, str) or not player_name.strip():
            return {"status": "ERROR", "message": "Player not found"}
        query = player_name.strip().strip(",")
        results = statsapi.lookup_player(query)
        if not results:
            return {"status": "ERROR", "message": "Player not found"}
        if len(results) == 1:
            p = results[0]
            fn = p.get("fullName", "Unknown")
            pos = (p.get("primaryPosition") or {}).get("abbreviation", "")
            team = (p.get("currentTeam") or {}).get("name", "") or (p.get("teamName") or "")
            pid = str(p.get("id", ""))
            is_pitcher = (pos == "P")
            is_two_way = fn.lower() == "shohei ohtani"
            return {
                "status": "OK",
                "id": pid,
                "fullName": fn,
                "position": pos,
                "team": team,
                "isPitcher": is_pitcher,
                "isTwoWay": is_two_way,
            }

        lc_name = query.lower()
        for p in results:
            if p.get("fullName", "").strip().lower() == lc_name:
                fn = p.get("fullName", "Unknown")
                pos = (p.get("primaryPosition") or {}).get("abbreviation", "")
                team = (p.get("currentTeam") or {}).get("name", "") or (p.get("teamName") or "")
                pid = str(p.get("id", ""))
                is_pitcher = (pos == "P")
                is_two_way = fn.lower() == "shohei ohtani"
                return {
                    "status": "OK",
                    "id": pid,
                    "fullName": fn,
                    "position": pos,
                    "team": team,
                    "isPitcher": is_pitcher,
                    "isTwoWay": is_two_way,
                }

        candidates = []
        for p in results[:8]:
            fn = p.get("fullName", "Unknown")
            pos = (p.get("primaryPosition") or {}).get("abbreviation", "")
            team = (p.get("currentTeam") or {}).get("name", "") or (p.get("teamName") or "")
            pid = str(p.get("id", ""))
            candidates.append({"id": pid, "fullName": fn, "position": pos, "team": team})
        return {"status": "AMBIGUOUS", "candidates": candidates}
    except Exception as e:
        return {"status": "ERROR", "message": f"lookup failed: {e}"}


@tool
def get_player_stats(player_id: str, time_info: str, stat_type: str) -> Dict:
    """
    statsapi.player_stat_data wrapper
    """
    raw_group = str(stat_type).strip()
    if raw_group.startswith("[") and raw_group.endswith("]"):
        group_param = raw_group
    else:
        group_tokens = raw_group.replace("/", " ").replace(",", " ").lower().split()
        allowed = {"hitting", "pitching", "fielding"}
        group_tokens = [g for g in group_tokens if g in allowed]
        if not group_tokens:
            group_tokens = ["hitting"]
        group_param = group_tokens[0] if len(group_tokens) == 1 else "[" + ",".join(group_tokens) + "]"

    ti = str(time_info).strip()
    try:
        person_id = int(player_id)
    except Exception:
        return {"status": "ERROR", "message": "invalid player_id", "player_id": player_id}

    try:
        if ti.lower() == "career":
            data = statsapi.player_stat_data(personId=person_id, group=group_param, type="career", sportId=1, season=None)
            return {"status": "OK", "player_id": player_id, "time": "career", "type": "career", "group": group_param, "raw": data}

        if ti.lower() in {"yearbyyear", "year_by_year", "yby"}:
            data = statsapi.player_stat_data(personId=person_id, group=group_param, type="yearByYear", sportId=1, season=None)
            return {"status": "OK", "player_id": player_id, "time": "yearByYear", "type": "yearByYear", "group": group_param, "raw": data}

        seasons: List[str] = []
        tmp = ti.replace("/", " ").replace(",", " ")
        toks = [t for t in tmp.split() if t]
        for token in toks:
            if token.isdigit() and len(token) == 4:
                seasons.append(token)

        if not seasons and ti.isdigit() and len(ti) == 4:
            seasons = [ti]

        if seasons:
            results = []
            for yr in seasons:
                try:
                    data = statsapi.player_stat_data(personId=person_id, group=group_param, type="season", sportId=1, season=yr)
                    results.append({"season": yr, "type": "season", "group": group_param, "raw": data})
                except Exception as e_season:
                    results.append({"season": yr, "error": str(e_season)})
            return {"status": "OK", "player_id": player_id, "time": ",".join(seasons), "type": "season", "group": group_param, "seasons": results}

        return {
            "status": "ERROR",
            "player_id": player_id,
            "time": time_info,
            "group": group_param,
            "message": "Unsupported time_info format. Use 'career', 'yearByYear', or 4-digit year(s).",
        }
    except Exception as e:
        return {"status": "ERROR", "message": f"player_stat_data failed: {e}"}


# =========================
# Tool: Scenario 2 core
# =========================

@tool
def summarize_window_with_preview(
    game_id: str,
    end_inning: str,
    window_innings: int = 3,
    wpa_threshold: float = 5.0,
    preview_batters: int = 3,
) -> Dict:
    """
    (시나리오2) play_by_play CSV 기반으로
    - 직전 N이닝(기본 3이닝) 경기 흐름 요약용 JSON(summary)
    - 다음 half-inning 프리뷰용 JSON(preview)
    를 반환합니다.

    Inputs
    - game_id: Game.ID (예: 'ARI201704020')
    - end_inning: 현재까지 진행된 시점 (예: '3회말', 'b3', '3회', 't6' 등)
    - window_innings: 요약할 이닝 수(기본 3)  -> 3이면 1~3회(초/말) 또는 4~6회(초/말) 같은 범위
    - wpa_threshold: abs(WPA)% 기준 (기본 5.0)
    - preview_batters: 다음 half-inning에서 등장하는 타자 n명(기본 3)

    Output (핵심)
    - summary: 득점 플레이/큰 WPA 변동/특이 이벤트/간단 통계/시작·종료 스코어
    - preview: 다음 half-inning 라벨(t4 등), 다음 타자 리스트, 상대 투수(가능하면)
    """
    try:
        df = _load_play_by_play().copy()
    except FileNotFoundError as e:
        return {"status": "ERROR", "message": str(e)}
    except Exception as e:
        return {"status": "ERROR", "message": f"failed to load csv: {e}"}

    if not isinstance(game_id, str) or not game_id.strip():
        return {"status": "ERROR", "message": "game_id required"}

    inning_num, half_hint = _parse_inning_hint(end_inning)
    if inning_num is None:
        return {"status": "ERROR", "message": "end_inning not recognized", "end_inning": end_inning}

    # half가 없으면 '말(b)'까지 진행된 것으로 가정 (시나리오2에서 3회까지 = 3회말까지로 보는 편이 자연스러움)
    if half_hint is None:
        half_hint = "b"

    end_key = (inning_num, 0 if half_hint == "t" else 1)

    try:
        win_n = int(window_innings)
        if win_n <= 0:
            win_n = 3
    except Exception:
        win_n = 3

    try:
        wpa_cut = float(wpa_threshold)
    except Exception:
        wpa_cut = 5.0

    try:
        prev_n = int(preview_batters)
        if prev_n <= 0:
            prev_n = 3
    except Exception:
        prev_n = 3

    df.rename(columns={"Play.Description": "PlayDescription"}, inplace=True)
    df = df[df["Game.ID"] == game_id].reset_index(drop=True)
    if df.empty:
        return {"status": "ERROR", "message": f"game_id not found in CSV: {game_id}", "game_id": game_id}

    # inning key 생성
    df["inning_key"] = df["Inning"].apply(_inning_key)
    df = df[df["inning_key"].notna()].reset_index(drop=True)
    if df.empty:
        return {"status": "ERROR", "message": "no inning rows found after parsing inning labels", "game_id": game_id}

    # 범위 계산: (end inning num - (N-1)) ~ end inning num
    start_inning_num = max(1, inning_num - (win_n - 1))
    start_key = (start_inning_num, 0)  # 항상 top부터 시작

    # inclusive filter for summary window
    def _in_range(k: Any) -> bool:
        if not isinstance(k, tuple) or len(k) != 2:
            return False
        return (k >= start_key) and (k <= end_key)

    window_rows = df[df["inning_key"].apply(_in_range)].copy().reset_index(drop=True)
    if window_rows.empty:
        return {
            "status": "ERROR",
            "message": "no rows found for summary window",
            "game_id": game_id,
            "start_inning": _label_from_key(start_key),
            "end_inning": _label_from_key(end_key),
        }

    # score tuples
    window_rows["score_tuple"] = window_rows["Score"].apply(_parse_score)
    window_rows["prev_score_tuple"] = window_rows["score_tuple"].shift(1)

    scoring_plays: List[Dict[str, Any]] = []
    wpa_swings: List[Dict[str, Any]] = []
    special_events: List[Dict[str, Any]] = []
    special_seen = set()

    # 통계(전체 window를 half(top/bottom) 단위로 누적)
    half_accum: Dict[str, Dict[str, int]] = {
        "top": {"runs_for": 0, "runs_against": 0, "hits": 0, "walks": 0, "strikeouts": 0, "errors": 0, "plays": 0},
        "bottom": {"runs_for": 0, "runs_against": 0, "hits": 0, "walks": 0, "strikeouts": 0, "errors": 0, "plays": 0},
    }

    def _half_name(k: Tuple[int, int]) -> str:
        return "top" if k[1] == 0 else "bottom"

    def _format_score(t: Optional[Tuple[int, int]]) -> Optional[str]:
        if not isinstance(t, tuple):
            return None
        return f"{t[0]}-{t[1]}"

    # window 시작/끝 스코어
    start_score = window_rows["score_tuple"].iloc[0]
    end_score = window_rows["score_tuple"].iloc[-1]

    for idx, row in window_rows.iterrows():
        k = row["inning_key"]
        if not isinstance(k, tuple):
            continue
        half = _half_name(k)

        desc = str(row.get("PlayDescription", "") or "")
        batter = row.get("Batter")
        pitcher = row.get("Pitcher")
        score_text = row.get("Score")
        score_tuple = row.get("score_tuple")
        prev_score = row.get("prev_score_tuple")
        if prev_score is None:
            prev_score = score_tuple

        half_accum[half]["plays"] += 1

        # hit/walk
        hit_flag, walk_flag = _detect_hit_and_walk(desc)
        if hit_flag:
            half_accum[half]["hits"] += 1
        if walk_flag:
            half_accum[half]["walks"] += 1

        # strikeout / error 키워드
        dlow = desc.lower()
        if "strikeout" in dlow or "struck out" in dlow:
            half_accum[half]["strikeouts"] += 1
        if "error" in dlow:
            half_accum[half]["errors"] += 1

        # scoring play: 점수 변화 기반(견고)
        delta_away = delta_home = 0
        if isinstance(score_tuple, tuple) and isinstance(prev_score, tuple):
            delta_away = score_tuple[0] - prev_score[0]
            delta_home = score_tuple[1] - prev_score[1]

        runs_for = delta_away if half == "top" else delta_home
        runs_against = delta_home if half == "top" else delta_away

        if (runs_for != 0) or (runs_against != 0):
            half_accum[half]["runs_for"] += max(runs_for, 0)
            half_accum[half]["runs_against"] += max(runs_against, 0)
            scoring_plays.append(
                {
                    "inning": _label_from_key(k),
                    "half": half,
                    "batter": batter,
                    "pitcher": pitcher,
                    "description": desc,
                    "score_before": _format_score(prev_score),
                    "score_after": _format_score(score_tuple) if isinstance(score_tuple, tuple) else score_text,
                    "runs_for": runs_for,
                    "runs_against": runs_against,
                }
            )

        # WPA swing
        wpa_val = _parse_percent(row.get("WPA"))
        if wpa_val is not None and abs(wpa_val) >= wpa_cut:
            wpa_swings.append(
                {
                    "inning": _label_from_key(k),
                    "half": half,
                    "batter": batter,
                    "pitcher": pitcher,
                    "description": desc,
                    "wpa_percent": wpa_val,
                    "score": score_text,
                }
            )

        # special events
        for label, patterns in SPECIAL_KEYWORDS.items():
            if any(p in dlow for p in patterns):
                key = (idx, label)
                if key not in special_seen:
                    special_seen.add(key)
                    special_events.append(
                        {
                            "inning": _label_from_key(k),
                            "half": half,
                            "label": label,
                            "description": desc,
                            "score": score_text,
                        }
                    )

    # 다음 half-inning preview
    next_key = _next_half_inning(end_key)
    next_label = _label_from_key(next_key)

    preview_rows = df[df["inning_key"] == next_key].copy().reset_index(drop=True)

    next_batters: List[str] = []
    next_pitcher: Optional[str] = None
    if not preview_rows.empty:
        # pitcher: 첫 row 기준
        next_pitcher = preview_rows["Pitcher"].iloc[0] if "Pitcher" in preview_rows.columns else None

        # batters: 등장 순서대로 unique
        for b in preview_rows["Batter"].tolist():
            if b is None:
                continue
            s = str(b).strip()
            if not s:
                continue
            if s not in next_batters:
                next_batters.append(s)
            if len(next_batters) >= prev_n:
                break

    # half별 통계는 plays>0만 반환
    stats_by_half = {k: v for k, v in half_accum.items() if v["plays"] > 0}

    return {
        "status": "OK",
        "game_id": game_id,
        "summary_window": {
            "start": _label_from_key(start_key),
            "end": _label_from_key(end_key),
            "window_innings": win_n,
        },
        "summary": {
            "score_start": _format_score(start_score),
            "score_end": _format_score(end_score),
            "scoring_plays": scoring_plays,
            "wpa_threshold": wpa_cut,
            "wpa_swings": wpa_swings,
            "special_events": special_events,
            "stats_by_half": stats_by_half,
        },
        "preview": {
            "next_half_inning": next_label,
            "next_batters": next_batters,
            "pitcher": next_pitcher,
        },
    }


# =========================
# Prompt (System) for Scenario 2
# =========================

SYSTEM_INSTRUCTIONS = (
    "당신은 시각장애인을 위한 MLB 라디오 해설형 챗봇입니다.\n\n"
    "[핵심 기능: 경기 흐름 요약 + 다음 이닝 프리뷰]\n"
    "- 사용자가 '경기 흐름 요약', '지금까지 요약', '다음 이닝 프리뷰'처럼 요청하면,\n"
    "  먼저 현재 보고 있는 경기의 Game.ID와 현재까지 진행된 시점(end_inning: 예 '3회말')이 대화에 있는지 확인하세요.\n"
    "  둘 중 하나라도 없으면 정중히 되물어 확보하세요.\n"
    "- Game.ID와 end_inning이 확보되면 summarize_window_with_preview 도구를 호출하세요.\n"
    "  기본값: window_innings=3, wpa_threshold=5.0, preview_batters=3\n\n"
    "[출력 규칙]\n"
    "1) 반드시 20~30초 분량의 한국어 라디오 해설 톤으로 말하세요.\n"
    "2) 요약 80% / 프리뷰 20% 비중을 지키세요.\n"
    "3) 도구에서 받은 점수, 이닝, 선수 이름, WPA 숫자는 절대 변경하지 마세요.\n"
    "4) 요약 파트에서는 득점 플레이(scoring_plays), 큰 WPA 변동(wpa_swings), 특이 이벤트(special_events)를 우선 언급하세요.\n"
    "   그리고 top/bottom별 득점·안타·볼넷·삼진·실책 수치를 짧게 묶어 설명하세요.\n"
    "5) 프리뷰 파트에서는 preview.next_half_inning, preview.next_batters(최대 3), preview.pitcher를 짧게 말하세요.\n"
    "6) 도구 JSON을 그대로 읽어주지 말고, 자연스러운 문장으로 재구성하세요.\n\n"
    "[선수 스탯(옵션)]\n"
    "- 사용자가 프리뷰에서 '이 타자/투수 어떤 선수냐'처럼 스탯을 원하면 find_player_id/get_player_stats를 추가로 호출해도 됩니다.\n"
    "- 다만 프리뷰의 길이를 길게 늘리지 말고 한두 문장 수준의 맥락만 더하세요.\n"
)


def build_agent() -> AgentExecutor:
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0.2)

    tools = [
        summarize_window_with_preview,
        # optional tools for enrichment
        find_player_id,
        get_player_stats,
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_INSTRUCTIONS),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=5,
        return_messages=True,
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
    )
    return executor


def main() -> None:
    # OPENAI_API_KEY 없으면 입력받아 설정
    if not os.getenv("OPENAI_API_KEY"):
        try:
            user_key = input("OpenAI API 키를 입력하세요 (미입력시 종료): ").strip()
        except (EOFError, KeyboardInterrupt):
            user_key = ""
        if user_key:
            os.environ["OPENAI_API_KEY"] = user_key
            print("OPENAI_API_KEY가 런타임에 설정되었습니다.")
        else:
            print("ERROR: OPENAI_API_KEY가 설정되어 있지 않습니다.", file=sys.stderr)
            print("예) export OPENAI_API_KEY=sk-... 또는 프로그램 시작 시 키 입력", file=sys.stderr)
            sys.exit(1)

    executor = build_agent()
    print("시나리오2(경기 흐름 요약+다음 이닝 프리뷰) 에이전트가 준비되었습니다. 종료: 'exit'/'quit'")
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("종료합니다.")
            break
        if not user_input:
            continue

        try:
            result = executor.invoke({"input": user_input})
            print(f"Agent: {result.get('output', '')}")
        except Exception as e:
            print(f"에러가 발생했습니다: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
