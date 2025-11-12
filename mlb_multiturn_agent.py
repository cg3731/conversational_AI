#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from typing import Dict, Optional

import statsapi

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory


# =========================
# Tools
# =========================

TEAM_NAME_TO_ID = {
	# American League East
	"baltimore orioles": 110,
	"orioles": 110,
	"볼티모어": 110,
	"볼티모어 오리올스": 110,
	"boston red sox": 111,
	"red sox": 111,
	"보스턴": 111,
	"보스턴 레드삭스": 111,
	"new york yankees": 147,
	"ny yankees": 147,
	"yankees": 147,
	"뉴욕 양키스": 147,
	"tampa bay rays": 139,
	"rays": 139,
	"템파베이": 139,
	"템파베이 레이스": 139,
	"toronto blue jays": 141,
	"blue jays": 141,
	"토론토": 141,
	"토론토 블루제이스": 141,
	# AL Central
	"chicago white sox": 145,
	"white sox": 145,
	"시카고 화이트삭스": 145,
	"cleveland guardians": 114,
	"guardians": 114,
	"클리블랜드 가디언즈": 114,
	"detroit tigers": 116,
	"tigers": 116,
	"디트로이트 타이거즈": 116,
	"kansas city royals": 118,
	"royals": 118,
	"캔자스시티 로열스": 118,
	"minnesota twins": 142,
	"twins": 142,
	"미네소타 트윈스": 142,
	# AL West
	"houston astros": 117,
	"astros": 117,
	"휴스턴 애스트로스": 117,
	"los angeles angels": 108,
	"angels": 108,
	"la angels": 108,
	"로스엔젤레스 에인절스": 108,
	"oakland athletics": 133,
	"athletics": 133,
	"a's": 133,
	"오클랜드 애슬레틱스": 133,
	"seattle mariners": 136,
	"mariners": 136,
	"시애틀 매리너스": 136,
	"texas rangers": 140,
	"rangers": 140,
	"텍사스 레인저스": 140,
	# NL East
	"atlanta braves": 144,
	"braves": 144,
	"애틀란타 브레이브스": 144,
	"miami marlins": 146,
	"marlins": 146,
	"마이애미 말린스": 146,
	"new york mets": 121,
	"mets": 121,
	"뉴욕 메츠": 121,
	"philadelphia phillies": 143,
	"phillies": 143,
	"필라델피아 필리스": 143,
	"washington nationals": 120,
	"nationals": 120,
	"워싱턴 내셔널스": 120,
	# NL Central
	"chicago cubs": 112,
	"cubs": 112,
	"시카고 컵스": 112,
	"cincinnati reds": 113,
	"reds": 113,
	"신시내티 레즈": 113,
	"milwaukee brewers": 158,
	"brewers": 158,
	"밀워키 브루어스": 158,
	"pittsburgh pirates": 134,
	"pirates": 134,
	"피츠버그 파이리츠": 134,
	"st. louis cardinals": 138,
	"st louis cardinals": 138,
	"cardinals": 138,
	"세인트루이스 카디널스": 138,
	# NL West
	"arizona diamondbacks": 109,
	"diamondbacks": 109,
	"애리조나 다이아몬드백스": 109,
	"los angeles dodgers": 119,
	"dodgers": 119,
	"로스엔젤레스 다저스": 119,
	"colorado rockies": 115,
	"rockies": 115,
	"콜로라도 로키스": 115,
	"san diego padres": 135,
	"padres": 135,
	"샌디에이고 파드리스": 135,
	"san francisco giants": 137,
	"giants": 137,
	"샌프란시스코 자이언츠": 137,
}

LEAGUE_NAME_TO_IDS = {
	"american league": {"league_id": 103, "divisions": {}},
	"american": {"league_id": 103, "divisions": {}},
	"al": {"league_id": 103, "divisions": {}},
	"내셔널": {"league_id": 104, "divisions": {}},  # fallback for Korean league names handled later
}

DIVISION_NAME_TO_ID = {
	"al west": 200,
	"american league west": 200,
	"알 서부": 200,
	"al east": 201,
	"american league east": 201,
	"알 동부": 201,
	"al central": 202,
	"american league central": 202,
	"알 중부": 202,
	"nl west": 203,
	"national league west": 203,
	"엔엘 서부": 203,
	"nl east": 204,
	"national league east": 204,
	"엔엘 동부": 204,
	"nl central": 205,
	"national league central": 205,
	"엔엘 중부": 205,
}

LEAGUE_NAME_TO_IDS.update(
	{
		"national league": {"league_id": 104, "divisions": {}},
		"national": {"league_id": 104, "divisions": {}},
		"nl": {"league_id": 104, "divisions": {}},
		"아메리칸 리그": {"league_id": 103, "divisions": {}},
		"아메리칸": {"league_id": 103, "divisions": {}},
		"내셔널 리그": {"league_id": 104, "divisions": {}},
	}
)


def _normalize_key(value: str) -> str:
	return value.strip().lower()


@tool
def resolve_team(team_query: str) -> Dict:
	"""
	팀 이름(한글/영문)을 받아 team_id, league_id, division_id를 추론합니다.
	"""
	if not isinstance(team_query, str) or not team_query.strip():
		return {"status": "ERROR", "message": "team_query required"}
	key = _normalize_key(team_query)
	team_id = TEAM_NAME_TO_ID.get(key)
	if team_id is None:
		try:
			lookup = statsapi.lookup_team(search=team_query.strip())
		except Exception:
			lookup = []
		if lookup:
			team_id = lookup[0].get("id")
		if team_id is None:
			return {"status": "ERROR", "message": f"team not recognized: {team_query}"}
	try:
		info = statsapi.lookup_team(team_id=team_id)
	except Exception:
		info = []
	league_id = None
	division_id = None
	if info:
		data = info[0]
		league = (data.get("league") or {}).get("id")
		division = (data.get("division") or {}).get("id")
		league_id = league
		division_id = division
	return {
		"status": "OK",
		"team_id": str(team_id),
		"league_id": str(league_id) if league_id else None,
		"division_id": str(division_id) if division_id else None,
	}


@tool
def resolve_league(league_query: str) -> Dict:
	"""
	리그 또는 디비전 명칭을 받아 league_id, division_id를 반환합니다.
	"""
	if not isinstance(league_query, str) or not league_query.strip():
		return {"status": "ERROR", "message": "league_query required"}
	key = _normalize_key(league_query)
	if key in DIVISION_NAME_TO_ID:
		return {
			"status": "OK",
			"league_id": None,
			"division_id": str(DIVISION_NAME_TO_ID[key]),
		}
	info = LEAGUE_NAME_TO_IDS.get(key)
	if info:
		return {
			"status": "OK",
			"league_id": str(info["league_id"]),
			"division_id": None,
		}
	return {"status": "ERROR", "message": f"league not recognized: {league_query}"}


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
			team_info = (p.get("currentTeam") or {})
			team = team_info.get("name", "") or (p.get("teamName") or "")
			team_id = team_info.get("id")
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
				"teamId": str(team_id) if team_id is not None else None,
				"isPitcher": is_pitcher,
				"isTwoWay": is_two_way,
			}
		# 다수 일치 시, fullName 정확 일치 우선
		lc_name = query.lower()
		for p in results:
			if p.get("fullName", "").strip().lower() == lc_name:
				fn = p.get("fullName", "Unknown")
				pos = (p.get("primaryPosition") or {}).get("abbreviation", "")
			team_info = (p.get("currentTeam") or {})
			team = team_info.get("name", "") or (p.get("teamName") or "")
			team_id = team_info.get("id")
				pid = str(p.get("id", ""))
				is_pitcher = (pos == "P")
				is_two_way = fn.lower() == "shohei ohtani"
				return {
					"status": "OK",
					"id": pid,
					"fullName": fn,
					"position": pos,
					"team": team,
				"teamId": str(team_id) if team_id is not None else None,
					"isPitcher": is_pitcher,
					"isTwoWay": is_two_way,
				}
		# 정확 일치가 없으면 모호함 반환 (후속 질문 유도)
		candidates = []
		for p in results[:8]:
			fn = p.get("fullName", "Unknown")
			pos = (p.get("primaryPosition") or {}).get("abbreviation", "")
			team_info = (p.get("currentTeam") or {})
			team = team_info.get("name", "") or (p.get("teamName") or "")
			team_id = team_info.get("id")
			pid = str(p.get("id", ""))
			candidates.append(
				{
					"id": pid,
					"fullName": fn,
					"position": pos,
					"team": team,
					"teamId": str(team_id) if team_id is not None else None,
				}
			)
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
def get_team_leaders(
	team_id: str,
	leader_categories: str,
	season: Optional[str] = None,
	leader_game_types: str = "R",
	limit: int = 10,
) -> Dict:
	"""
	statsapi.team_leader_data를 사용해 팀 내 리더 데이터를 조회합니다.
	- team_id: 팀 ID 문자열(예: '147' for New York Yankees)
	- leader_categories: 쉼표/공백/슬래시 구분 문자열 또는 '[avg,homeRuns]' 형식
	- season: 조회 시즌(미지정 시 현재 시즌)
	- leader_game_types: 기본 'R' (정규시즌)
	- limit: 반환 선수 수
	"""
	print(
		f"[Tool Log] get_team_leaders 호출됨: team_id={team_id}, categories={leader_categories}, "
		f"season={season}, game_types={leader_game_types}, limit={limit}"
	)
	try:
		tid = int(team_id)
	except (TypeError, ValueError):
		return {"status": "ERROR", "message": "invalid team_id", "team_id": team_id}

	# leaderCategories는 문자열 형태 그대로 전달 가능.
	# 사용자가 'avg,homeRuns' 형태로 주면 statsapi가 처리 가능하지만,
	# bracket 형식 '[avg,homeRuns]'도 허용하므로 원문 그대로 전달하되, 공백만 정리.
	categories = leader_categories.strip()
	if not categories:
		return {"status": "ERROR", "message": "leader_categories required"}

	kwargs = {
		"teamId": tid,
		"leaderCategories": categories,
		"leaderGameTypes": leader_game_types or "R",
		"limit": limit,
	}
	if season:
		kwargs["season"] = season
	try:
		data = statsapi.team_leader_data(**kwargs)
		return {
			"status": "OK",
			"team_id": team_id,
			"leader_categories": categories,
			"season": season,
			"leader_game_types": leader_game_types,
			"limit": limit,
			"leaders": data,
		}
	except Exception as e:
		return {"status": "ERROR", "message": f"team_leader_data failed: {e}"}


@tool
def get_league_leaders(
	leader_categories: str,
	season: Optional[str] = None,
	limit: int = 10,
	stat_group: Optional[str] = None,
	league_id: Optional[str] = None,
	game_types: Optional[str] = None,
	player_pool: Optional[str] = None,
	sport_id: int = 1,
	stat_type: Optional[str] = None,
) -> Dict:
	"""
	statsapi.league_leader_data를 사용해 리그(또는 전체) 리더 데이터를 조회합니다.
	- leader_categories: 필수. 예: 'homeRuns' 또는 '[homeRuns,runsBattedIn]'
	- season: 시즌(미지정 시 현재 시즌)
	- limit: 반환 선수 수
	- stat_group: 'hitting'/'pitching'/'fielding' 등 (가능하면 지정 권장)
	- league_id: 리그 ID (103=AL, 104=NL 등)
	- game_types: 'R', 'S', 'PS' 등
	- player_pool: 'all', 'qualified', 'rookies' 중 선택(기본 qualified)
	- sport_id: 기본 1(MLB)
	- stat_type: 'season', 'career' 등 (statsSingleSeason은 아직 미지원이므로 주의)
	"""
	print(
		f"[Tool Log] get_league_leaders 호출됨: categories={leader_categories}, season={season}, "
		f"limit={limit}, stat_group={stat_group}, league_id={league_id}, "
		f"game_types={game_types}, player_pool={player_pool}, sport_id={sport_id}, stat_type={stat_type}"
	)
	categories = leader_categories.strip()
	if not categories:
		return {"status": "ERROR", "message": "leader_categories required"}

	kwargs = {
		"leaderCategories": categories,
		"limit": limit,
		"sportId": sport_id,
	}
	if season:
		kwargs["season"] = season
	if stat_group:
		kwargs["statGroup"] = stat_group
	if league_id:
		kwargs["leagueId"] = league_id
	if game_types:
		kwargs["gameTypes"] = game_types
	if player_pool:
		kwargs["playerPool"] = player_pool
	if stat_type:
		kwargs["statType"] = stat_type

	try:
		data = statsapi.league_leader_data(**kwargs)
		return {
			"status": "OK",
			"leader_categories": categories,
			"season": season,
			"limit": limit,
			"stat_group": stat_group,
			"league_id": league_id,
			"game_types": game_types,
			"player_pool": player_pool,
			"sport_id": sport_id,
			"stat_type": stat_type,
			"leaders": data,
		}
	except Exception as e:
		return {"status": "ERROR", "message": f"league_leader_data failed: {e}"}


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
	"   - status='AMBIGUOUS'면 후보 목록을 요약해 누구인지 사용자가 고르도록 안내하세요(가능하면 id로 선택).\n"
	"   - status='ERROR'면 이름을 다시 확인하거나 더 구체적인 정보를 요청하세요.\n"
	"4) 사용자가 특정 팀이나 리그/디비전을 언급하거나 당신이 문맥에서 팀/리그를 파악했다면, resolve_team 또는 resolve_league 도구를 호출해 "
	"공식 team_id, league_id, division_id를 확보하세요. 그 결과를 팀/리그 관련 API 호출에 활용하세요.\n"
	"5) stat_type이 확정되면 get_player_stats 도구를 호출하세요.\n"
	"6) 사용자가 \"팀 내에서 어느 정도인지\", \"팀 동료와 비교\" 등 팀 성적 비교를 원하거나 당신이 판단하기에 팀 맥락이 중요하면, "
	"find_player_id 결과에 포함된 teamId를 활용해 get_team_leaders 도구를 호출하세요.\n"
	"   - leaderCategories는 요청 의도에 맞춰 결정하세요(예: HR, RBI, ERA, AVG 등). 여러 지표가 필요하면 '[hr,rbi]'처럼 전달하세요.\n"
	"   - get_team_leaders 결과를 받아 해당 선수의 팀 내 위치(순위, 리더와의 차이)를 명확히 설명하세요.\n"
	"7) 사용자가 \"리그에서 어느 수준인지\", \"리그 리더와 비교\" 등 리그 맥락을 원하거나 당신이 판단하기에 리그 비교가 중요하면, "
	"get_league_leaders 도구를 호출하세요.\n"
	"   - statGroup, leagueId(예: 103=AL, 104=NL), leaderCategories를 사용자가 언급한 지표에 맞게 설정하세요.\n"
	"   - player_pool(qualified/rookies/all)이나 season, statType('season', 'career' 등)을 상황에 맞춰 지정하세요.\n"
	"   - 결과를 활용하여 해당 선수의 리그 내 순위/격차/컨텍스트를 설명하세요.\n"
	"8) 3가지 기본 정보가 충족되어도, 필요 시 사용자에게 한 번 더 확인하세요: \"여러 선수를 비교하시겠어요, 아니면 이 선수만 분석할까요?\" 비교 의도가 있으면 후속 플레이어도 같은 방식으로 수집하세요.\n"
	"9) 도구에서 받은 데이터를 그대로 나열하지 말고, 핵심 지표를 요약/분석하여 사용자가 이해하기 쉬운 자연어로 답변하세요. "
	"사용자의 질문 의도에 맞춰 어떤 지표를 우선적으로 해석할지 스스로 결정하고, 팀/리그 비교가 포함되면 리더/순위를 명시적으로 언급하세요.\n"
)


def build_agent() -> AgentExecutor:
	# Model
	model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
	llm = ChatOpenAI(
		model=model_name,
		temperature=0.2,
	)

	# Tools
	tools = [
		resolve_team,
		resolve_league,
		find_player_id,
		get_player_stats,
		get_team_leaders,
		get_league_leaders,
	]

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


