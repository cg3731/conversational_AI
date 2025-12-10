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


# Play-by-play CSV 경로 및 캐시
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


def _parse_inning_value(target_inning: Any) -> Tuple[Optional[int], Optional[str]]:
	"""이닝 번호와 초/말(half)를 파싱"""
	if target_inning is None:
		return None, None
	text = str(target_inning).strip().lower()
	match = re.search(r"(\d+)", text)
	inning_num = int(match.group(1)) if match else None
	half = None
	if text.startswith("t") or "top" in text or "초" in text:
		half = "t"
	elif text.startswith("b") or "bot" in text or "bottom" in text or "말" in text:
		half = "b"
	return inning_num, half


def _parse_score(score_text: Any) -> Optional[Tuple[int, int]]:
	if not isinstance(score_text, str):
		return None
	try:
		left, right = score_text.split("-")
		return int(left), int(right)
	except Exception:
		return None


def _parse_percent(value: Any) -> Optional[float]:
	if value is None:
		return None
	text = str(value).strip().replace("%", "")
	if not text:
		return None
	try:
		return float(text)
	except Exception:
		return None


def _detect_hit_and_walk(desc: str) -> Tuple[bool, bool]:
	"""튜플을 반환: (안타 여부, 볼넷/사구 여부)"""
	if not isinstance(desc, str):
		return False, False
	d = desc.lower()
	# 더블플레이는 안타로 계산하지 않음
	if "double play" in d:
		is_hit = False
	else:
		hit_keywords = [
			"single",
			"double",
			"triples",
			"triple",
			"home run",
			"homers",
			"homered",
		]
		is_hit = any(k in d for k in hit_keywords)
	is_walk = "walk" in d or "intent walk" in d or "hit by pitch" in d
	return is_hit, is_walk


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


# =========================
# Tools
# =========================

@tool
def find_player_id(player_name: str) -> Dict:
	"""
	선수 이름(예: 'Shohei Ohtani')을 입력받아 MLB Stats API에서
	고유한 'player_id'(예: '660271')를 검색하여 반환합니다.
	선수를 찾지 못하면 에러 메시지를 반환합니다.
	여러 명이면 fullName/포지션/팀 정보를 포함해 AMBIGUOUS로 반환합니다.
	반환 형식:
	- {"status":"OK","id":"660271","fullName":"Shohei Ohtani","position":"DH","team":"Los Angeles Dodgers","isPitcher":false,"isTwoWay":true}
	- {"status":"AMBIGUOUS","candidates":[{"id":"123","fullName":"...","position":"P","team":"..."}]}
	- {"status":"ERROR","message":"..."}
	"""
	print(f"[Tool Log] find_player_id 호출됨: {player_name}")
	# statsapi를 사용하여 실제 ID 조회
	try:
		if not isinstance(player_name, str) or not player_name.strip():
			return {"status": "ERROR", "message": "Player not found"}
		# 간단 정규화(말미 콤마/공백 제거)
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
			# 간단한 투타겸업 휴리스틱: 오타니는 two-way
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
		# 다수 일치 시, fullName 정확 일치 우선
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
		# 정확 일치가 없으면 모호함 반환 (후속 질문 유도)
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
	'player_id', 'time_info' (예: '2025', 'career', '2025-05-01_2025-05-31'),
	'stat_type' (예: 'hitting', 'pitching', 'fielding' 혹은 다중: 'hitting,pitching')을 받아
	선수의 스탯 데이터를 JSON(dict) 형태로 반환합니다.
	- group: 'hitting' | 'pitching' | 'fielding' 또는 다중일 경우 문자열 '[hitting,pitching]' 형식
	- type: 'career' | 'season' | 'yearByYear' (v0.1.7+) — 다중일 경우 '[career,season]' 형식
	- season 인자는 type에 'season'이 포함될 때만 사용할 수 있습니다.
	"""
	print(f"[Tool Log] get_player_stats 호출됨: {player_id}, {time_info}, {stat_type}")
	# statsapi를 사용해 가능한 한 실제 데이터 시도, 실패 시 목업 반환
	# 1) group 파싱: 단일 또는 다중
	raw_group = str(stat_type).strip()
	if raw_group.startswith("[") and raw_group.endswith("]"):
		group_param = raw_group  # 이미 올바른 포맷으로 들어온 경우
	else:
		# 허용 구분자: 콤마/슬래시/공백
		group_tokens = (
			raw_group.replace("/", " ")
			.replace(",", " ")
			.lower()
			.split()
		)
		allowed = {"hitting", "pitching", "fielding"}
		group_tokens = [g for g in group_tokens if g in allowed]
		if not group_tokens:
			group_tokens = ["hitting"]
		if len(group_tokens) == 1:
			group_param = group_tokens[0]
		else:
			group_param = "[" + ",".join(group_tokens) + "]"
	# time_info 해석:
	# - 'career' → type='career'
	# - 'yearbyyear' → type='yearByYear'
	# - 'YYYY' → 단일 시즌
	# - 'YYYY,YYYY,...' 또는 'YYYY YYYY' → 여러 시즌 각각 조회하여 리스트로 반환
	ti = str(time_info).strip()
	try:
		person_id = int(player_id)
	except Exception:
		return {"error": "invalid player_id", "player_id": player_id}
	try:
		# 커리어
		if ti.lower() == "career":
			data = statsapi.player_stat_data(
				personId=person_id,
				group=group_param,
				type="career",
				sportId=1,
				season=None
			)
			return {
				"player_id": player_id,
				"time": "career",
				"type": "career",
				"group": group_param,
				"raw": data
			}
		# 연도별 전체(yearByYear)
		if ti.lower() in {"yearbyyear", "year_by_year", "yby"}:
			data = statsapi.player_stat_data(
				personId=person_id,
				group=group_param,
				type="yearByYear",
				sportId=1,
				season=None
			)
			return {
				"player_id": player_id,
				"time": "yearByYear",
				"type": "yearByYear",
				"group": group_param,
				"raw": data
			}
		# 여러 시즌 파싱
		# 허용 구분자: 콤마/공백/슬래시
		seasons: list[str] = []
		if any(sep in ti for sep in [",", " ", "/"]):
			candidates = [t for sep in [",", " ", "/"] for t in ti.split(sep)]
			# 위 방식은 중복 분해가 있으므로 간단 정규화
			tmp = ti.replace("/", " ").replace(",", " ")
			candidates = [t for t in tmp.split(" ") if t]
			for token in candidates:
				if token.isdigit() and len(token) == 4:
					seasons.append(token)
		elif ti.isdigit() and len(ti) == 4:
			seasons.append(ti)
		# 단일/복수 시즌 처리
		if seasons:
			results = []
			for yr in seasons:
				try:
					data = statsapi.player_stat_data(
						personId=person_id,
						group=group_param,
						type="season",  # season 파라미터는 type에 'season'이 포함될 때만 사용
						sportId=1,
						season=yr
					)
					results.append({"season": yr, "type": "season", "group": group_param, "raw": data})
				except Exception as e_season:
					results.append({"season": yr, "error": str(e_season)})
			return {
				"player_id": player_id,
				"time": ",".join(seasons),
				"type": "season",
				"group": group_param,
				"seasons": results
			}
		# 기타 형식(예: 날짜 범위)은 현재 미지원
		return {
			"player_id": player_id,
			"time": time_info,
			"type": "unknown",
			"group": group_param,
			"error": "Unsupported time_info format. Use 'career', 'yearByYear', or 4-digit year(s)."
		}
	except Exception as e:
		print(f"[Tool Log] statsapi.player_stat_data 실패: {e}")
		# 폴백 목업
		return {
			"player_id": player_id,
			"time": time_info,
			"type": "fallback",
			"group": group_param,
			"stats": {"HR": 40, "AVG": 0.301, "ERA": 3.10}
		}


@tool
def summarize_inning_events(
	game_id: str,
	target_inning: str,
	current_inning: str = "",
	wpa_threshold: float = 5.0,
) -> Dict:
	"""
	play_by_play CSV를 기반으로 특정 이닝(초/말)을 요약하는 데이터 JSON을 반환합니다.
	- game_id: Game.ID (예: 'ARI201704020')
	- target_inning: '3', '3회', 't5' 등 숫자 및 초/말 힌트 포함 문자열
	- current_inning: 현재 경기 진행 이닝(예: '5회초'), 내레이션에서 언급하도록 전달
	- wpa_threshold: WPA 절대값이 이 값(%) 이상인 플레이를 큰 변동으로 간주 (기본 5)
	"""
	print(f"[Tool Log] summarize_inning_events 호출: game={game_id}, target_inning={target_inning}, current_inning={current_inning}")
	try:
		df = _load_play_by_play().copy()
	except FileNotFoundError as e:
		return {"status": "ERROR", "message": str(e)}
	except Exception as e:
		return {"status": "ERROR", "message": f"failed to load csv: {e}"}
	inning_num, half_filter = _parse_inning_value(target_inning)
	if inning_num is None:
		return {"status": "ERROR", "message": "target_inning not recognized", "target_inning": target_inning}
	try:
		wpa_cut = float(wpa_threshold)
	except Exception:
		wpa_cut = 5.0
	df.rename(columns={"Play.Description": "PlayDescription"}, inplace=True)
	df = df[df["Game.ID"] == game_id].reset_index(drop=True)
	if df.empty:
		return {"status": "ERROR", "message": f"game_id not found in CSV: {game_id}"}
	df["score_tuple"] = df["Score"].apply(_parse_score)
	df["prev_score_tuple"] = df["score_tuple"].shift(1)

	def _match_inning(value: Any) -> bool:
		if not isinstance(value, str):
			return False
		val = value.strip().lower()
		if not val or len(val) < 2:
			return False
		try:
			num = int(val[1:])
		except Exception:
			return False
		if num != inning_num:
			return False
		if half_filter:
			return val.startswith(half_filter)
		return True

	target_rows = df[df["Inning"].apply(_match_inning)]
	if target_rows.empty:
		return {
			"status": "ERROR",
			"message": f"inning data not found for {target_inning}",
			"game_id": game_id,
		}

	start_score_tuple = target_rows["prev_score_tuple"].iloc[0]
	if start_score_tuple is None:
		start_score_tuple = target_rows["score_tuple"].iloc[0]
	end_score_tuple = target_rows["score_tuple"].iloc[-1]

	scoring_plays: List[Dict[str, Any]] = []
	wpa_swings: List[Dict[str, Any]] = []
	special_events: List[Dict[str, Any]] = []
	special_seen = set()
	half_accum: Dict[str, Dict[str, int]] = {
		"top": {"runs_for": 0, "runs_against": 0, "hits": 0, "walks": 0, "plays": 0},
		"bottom": {"runs_for": 0, "runs_against": 0, "hits": 0, "walks": 0, "plays": 0},
	}

	for idx, row in target_rows.iterrows():
		inning_label = str(row["Inning"]).strip().lower()
		half = "top" if inning_label.startswith("t") else "bottom"
		desc = str(row.get("PlayDescription", "") or "")
		batter = row.get("Batter")
		pitcher = row.get("Pitcher")
		score_text = row.get("Score")
		prev_score = row.get("prev_score_tuple")
		score_tuple = row.get("score_tuple")
		if prev_score is None:
			prev_score = score_tuple
		delta_away = delta_home = 0
		if isinstance(score_tuple, tuple) and isinstance(prev_score, tuple):
			delta_away = score_tuple[0] - prev_score[0]
			delta_home = score_tuple[1] - prev_score[1]
		runs_for = delta_away if half == "top" else delta_home
		runs_against = delta_home if half == "top" else delta_away
		half_accum[half]["plays"] += 1
		hit_flag, walk_flag = _detect_hit_and_walk(desc)
		if hit_flag:
			half_accum[half]["hits"] += 1
		if walk_flag:
			half_accum[half]["walks"] += 1
		if runs_for > 0 or runs_against > 0:
			half_accum[half]["runs_for"] += max(runs_for, 0)
			half_accum[half]["runs_against"] += max(runs_against, 0)
			score_before = f"{prev_score[0]}-{prev_score[1]}" if isinstance(prev_score, tuple) else None
			score_after = f"{score_tuple[0]}-{score_tuple[1]}" if isinstance(score_tuple, tuple) else score_text
			scoring_plays.append(
				{
					"half": half,
					"batter": batter,
					"pitcher": pitcher,
					"description": desc,
					"score_before": score_before,
					"score_after": score_after,
					"runs_for": runs_for,
					"runs_against": runs_against,
				}
			)
		wpa_val = _parse_percent(row.get("WPA"))
		if wpa_val is not None and abs(wpa_val) >= wpa_cut:
			wpa_swings.append(
				{
					"half": half,
					"batter": batter,
					"pitcher": pitcher,
					"description": desc,
					"wpa_percent": wpa_val,
					"score": score_text,
				}
			)
		desc_lower = desc.lower()
		for label, patterns in SPECIAL_KEYWORDS.items():
			if any(p in desc_lower for p in patterns):
				key = (idx, label)
				if key not in special_seen:
					special_seen.add(key)
					special_events.append(
						{
							"half": half,
							"label": label,
							"description": desc,
							"score": score_text,
						}
					)

	stats_by_half = {k: v for k, v in half_accum.items() if v["plays"] > 0}

	def _format_score(score_tuple_value: Optional[Tuple[int, int]]) -> Optional[str]:
		if not isinstance(score_tuple_value, tuple):
			return None
		return f"{score_tuple_value[0]}-{score_tuple_value[1]}"

	return {
		"status": "OK",
		"game_id": game_id,
		"target_inning": inning_num,
		"target_half": "top" if half_filter == "t" else "bottom" if half_filter == "b" else "full",
		"current_inning": current_inning or None,
		"wpa_threshold": wpa_cut,
		"inning_start_score": _format_score(start_score_tuple),
		"inning_end_score": _format_score(end_score_tuple),
		"scoring_plays": scoring_plays,
		"wpa_swings": wpa_swings,
		"special_events": special_events,
		"stats_by_half": stats_by_half,
	}


# =========================
# Prompt (System) with slot-filling rules
# =========================

SYSTEM_INSTRUCTIONS = (
	"당신은 MLB 야구 스탯을 분석해주는 전문 챗봇입니다.\n\n"
	"[중요 규칙] 최종 목표는 get_player_stats 도구를 사용해 스탯을 가져오는 것입니다. "
	"이 도구를 호출하려면 반드시 3가지 정보가 필요합니다:\n"
	"- player_name (선수 이름)\n"
	"- time_info (시간 정보: 'YYYY' 시즌, 'career', 'YYYY-MM-DD_YYYY-MM-DD' 등)\n"
	"- stat_type (스탯 유형: 'hitting', 'pitching', 'fielding')\n\n"
	"[행동 지침]\n"
	"1) 사용자의 현재 질문과 대화 기록(chat_history)을 모두 검토하여 위 3가지 정보가 모두 있는지 확인하세요.\n"
	"2) 하나라도 부족하면 절대로 도구를 호출하지 마세요. 대신, 부족한 정보를 정중히 되묻는 질문을 하세요.\n"
	"   예: \"오타니 선수 스탯이 궁금하시군요. 어떤 시즌의 기록을 알려드릴까요?\", "
	"\"타격과 투구 중 어떤 기록이 궁금하세요?\"\n"
	"3) 3가지 정보가 모두 모였다면, 먼저 find_player_id 도구를 호출하여 선수 정보(ID/포지션/팀)를 가져오세요.\n"
	"   - player_name이 한글 등 비ASCII여도 사용자에게 다시 묻지 말고, 당신이 스스로 공식 영문 표기로 변환하여 조회하세요. "
	"예: '오타니' → 'Shohei Ohtani', '류현진' → 'Hyun Jin Ryu'.\n"
	"   - lookup_player는 부분 이름/성만으로도 검색이 가능합니다. 풀네임 재확인을 요구하지 말고, 먼저 내부적으로 다양한 정규화(영문화, 성/이름만, 공백/구두점 제거)를 시도하세요.\n"
	"   - 절대 '선수 이름이 맞는가요?'처럼 이름 자체를 재확인하지 마세요. 이름을 물어보는 것은 오직 player_name이 전혀 제공되지 않았을 때만 허용됩니다.\n"
	"   - 이름 자체의 모호성(동명이인 등)으로 후보가 여러 명일 때만, 후보 목록을 요약해 사용자가 선택하도록 되묻습니다(가능하면 id로 선택).\n"
	"   - 이름 번역/정규화(영문화)는 내부적으로 수행하고, time_info와 stat_type이 모두 확보되기 전에는 도구를 호출하지 마세요 "
	"(이름 모호성 해결이 정말 필요한 특별한 경우만 find_player_id를 선행 호출).\n"
	"   - find_player_id의 결과가 status=OK이면 포지션에 따라 stat_type 결정을 보조하세요:\n"
	"     • position='P' (투수)인 경우: 기본값으로 'pitching'을 선택하고 바로 진행하세요(사용자가 특별히 다른 유형을 언급하지 않았다면).\n"
	"     • 포지션이 타자(예: OF/1B/SS/C/DH 등)인 경우: 'hitting'과 'fielding' 중 원하는 유형을 확인하세요.\n"
	"     • 투타겸업(예: Shohei Ohtani)으로 판단되면: 'hitting' vs 'pitching' 중 무엇을 볼지 꼭 물어보세요.\n"
	"   - status=AMBIGUOUS면 후보 목록을 요약해 누구인지 물어보세요(가능하면 id로 선택받기).\n"
	"   - status=ERROR면 이름을 다시 확인하거나 더 구체적인 정보를 요청하세요.\n"
	"4) stat_type이 확정되면 get_player_stats 도구를 호출하세요.\n"
	"5) find_player_id가 'ERROR:'를 반환하면 이름을 다시 확인하거나 더 구체적인 정보를 요청하세요.\n"
	"6) find_player_id가 'AMBIGUOUS:'를 반환하면 후보 목록 중 누구인지 사용자에게 명확히 물어보세요.\n"
	"7) 3가지 정보가 충족되어도, 필요 시 사용자에게 한 번 더 확인하세요: \"여러 선수를 비교하시겠어요, 아니면 이 선수만 분석할까요?\" 비교 의도가 있으면 후속 플레이어도 같은 방식으로 수집하세요.\n"
	"8) 도구에서 받은 JSON 데이터를 그대로 출력하지 말고, 핵심 지표를 요약/분석하여 사용자가 이해하기 쉬운 자연어로 답변하세요. 사용자의 질문 의도에 맞춰 어떤 지표를 우선적으로 해석할지 스스로 결정하세요.\n"
	"\n[특정 이닝 설명]\n"
	"- 사용자가 \"3회에 무슨 일이 있었는지\"처럼 특정 이닝 요약을 요청하면, 먼저 현재 보고 있는 경기의 Game.ID가 대화에 포함되어 있는지 확인하세요(없으면 Game.ID를 물어보세요).\n"
	"- game_id가 확보되면 summarize_inning_events 도구를 호출해 target_inning(예: '3회', 't5')과 current_inning(예: '5회초')을 함께 전달하세요. current_inning을 모르면 먼저 사용자에게 현재 이닝을 물어보거나, 정말 알 수 없으면 빈 문자열로 넘겨도 됩니다. wpa_threshold 기본값 5%를 유지합니다.\n"
	"- 도구에서 받은 JSON을 그대로 읽어주지 말고, 점수/이닝/선수 이름/WPA 숫자는 절대 변경하지 않은 채 20~30초 분량의 시각장애인용 라디오 해설 톤으로 풀어서 설명하세요.\n"
	"- 득점 플레이, |WPA|가 기준 이상인 큰 변동 플레이, 특이 이벤트(키워드 감지)를 우선적으로 언급하고, top/bottom별 득점·피득점·안타·볼넷 수치를 짧게 묶어서 알려주세요.\n"
	"- current_inning 값을 확보하면 \"이 경기는 현재 (current_inning)\"처럼 먼저 알려주고, 이어서 target_inning의 흐름을 요약하세요. current_inning이 모호하면 먼저 물어보고 진행합니다.\n"
)


def build_agent() -> AgentExecutor:
	# Model
	model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
	llm = ChatOpenAI(
		model=model_name,
		temperature=0.2,
	)

	# Tools
	tools = [find_player_id, get_player_stats, summarize_inning_events]

	# Prompt
	prompt = ChatPromptTemplate.from_messages(
		[
			("system", SYSTEM_INSTRUCTIONS),
			MessagesPlaceholder(variable_name="chat_history"),
			("user", "{input}"),
			MessagesPlaceholder(variable_name="agent_scratchpad"),
		]
	)

	# Agent
	agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)

	# Memory
	memory = ConversationBufferWindowMemory(
		memory_key="chat_history",
		k=5,
		return_messages=True,
	)

	# Executor
	executor = AgentExecutor(
		agent=agent,
		tools=tools,
		memory=memory,
		verbose=True,
	)
	return executor


def main() -> None:
	# OPENAI_API_KEY가 없으면 입력받아 설정
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
	print("대화형 MLB 스탯 에이전트가 준비되었습니다. 종료하려면 'exit' 또는 'quit'를 입력하세요.")
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
			# AgentExecutor returns a dict with 'output'
			print(f"Agent: {result.get('output', '')}")
		except Exception as e:
			print(f"에러가 발생했습니다: {e}", file=sys.stderr)


if __name__ == "__main__":
	main()
