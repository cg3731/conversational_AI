#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from typing import Dict

import statsapi

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory


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
)


def build_agent() -> AgentExecutor:
	# Model
	model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
	llm = ChatOpenAI(
		model=model_name,
		temperature=0.2,
	)

	# Tools
	tools = [find_player_id, get_player_stats]

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


