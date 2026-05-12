  File "/data/AIGC_Video_Reasonableness_Evaluation/scripts/./debug_dynamics.py", line 1174, in <module>
    main()
  File "/data/AIGC_Video_Reasonableness_Evaluation/scripts/./debug_dynamics.py", line 1093, in main
    r = run_motion_logic_analysis(str(v), args, mllm_client)
  File "/data/AIGC_Video_Reasonableness_Evaluation/scripts/./debug_dynamics.py", line 974, in run_motion_logic_analysis
    return analyzer.analyze(hub)
  File "/data/AIGC_Video_Reasonableness_Evaluation/src/motion_logic/analyzer.py", line 129, in analyze
    result = judge_naturalness_mllm(
  File "/data/AIGC_Video_Reasonableness_Evaluation/src/motion_logic/naturalness_judge.py", line 32, in judge_naturalness_mllm
    return mllm_client.judge_video_clip(frames, MOTION_NATURALNESS_PROMPT)
  File "/data/AIGC_Video_Reasonableness_Evaluation/src/mllm/client.py", line 29, in judge_video_clip
    return self._call_api(frames, prompt)
  File "/data/AIGC_Video_Reasonableness_Evaluation/src/mllm/client.py", line 52, in _call_api
    return self._call_huawei_custom(images_b64, prompt)
  File "/data/AIGC_Video_Reasonableness_Evaluation/src/mllm/client.py", line 207, in _call_huawei_custom
    parsed = self._parse_custom_response_content(result)
  File "/data/AIGC_Video_Reasonableness_Evaluation/src/mllm/client.py", line 178, in _parse_custom_response_content
    parsed = self._parse_custom_response_content(nested)
  File "/data/AIGC_Video_Reasonableness_Evaluation/src/mllm/client.py", line 186, in _parse_custom_response_content
    return parse_json_from_model_text(text)
  File "/data/AIGC_Video_Reasonableness_Evaluation/src/mllm/dashscope_video_reasonableness.py", line 73, in parse_json_from_model_text
    raise ValueError(f"无法解析为 JSON，模型原文前 500 字: {text[:500]!r}")
ValueError: 无法解析为 JSON，模型原文前 500 字: 'At most 10 image(s) may be provided in one prompt.'
