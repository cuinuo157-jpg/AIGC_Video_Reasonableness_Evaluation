    sock.connect(sa)
ConnectionRefusedError: [Errno 111] Connection refused

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/urllib3/connectionpool.py", line 788, in urlopen
    response = self._make_request(
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/urllib3/connectionpool.py", line 493, in _make_request
    conn.request(
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/urllib3/connection.py", line 500, in request
    self.endheaders()
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/http/client.py", line 1278, in endheaders
    self._send_output(message_body, encode_chunked=encode_chunked)
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/http/client.py", line 1038, in _send_output
    self.send(msg)
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/http/client.py", line 976, in send
    self.connect()
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/urllib3/connection.py", line 331, in connect
    self.sock = self._new_conn()
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/urllib3/connection.py", line 219, in _new_conn
    raise NewConnectionError(
urllib3.exceptions.NewConnectionError: HTTPConnection(host='localhost', port=8201): Failed to establish a new connection: [Errno 111] Connection refused

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/requests/adapters.py", line 645, in send
    resp = conn.urlopen(
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/urllib3/connectionpool.py", line 842, in urlopen
    retries = retries.increment(
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/urllib3/util/retry.py", line 543, in increment
    raise MaxRetryError(_pool, url, reason) from reason  # type: ignore[arg-type]
urllib3.exceptions.MaxRetryError: HTTPConnectionPool(host='localhost', port=8201): Max retries exceeded with url: /v1 (Caused by NewConnectionError("HTTPConnection(host='localhost', port=8201): Failed to establish a new connection: [Errno 111] Connection refused"))

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
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
  File "/data/AIGC_Video_Reasonableness_Evaluation/src/mllm/client.py", line 199, in _call_huawei_custom
    response = requests.post(
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/requests/api.py", line 115, in post
    return request("post", url, data=data, json=json, **kwargs)
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/requests/api.py", line 59, in request
    return session.request(method=method, url=url, **kwargs)
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/requests/sessions.py", line 592, in request
    resp = self.send(prep, **send_kwargs)
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/requests/sessions.py", line 706, in send
    r = adapter.send(request, **kwargs)
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/requests/adapters.py", line 678, in send
    raise ConnectionError(e, request=request)
requests.exceptions.ConnectionError: HTTPConnectionPool(host='localhost', port=8201): Max retries exceeded with url: /v1 (Caused by NewConnectionError("HTTPConnection(host='localhost', port=8201): Failed to establish a new connection: [Errno 111] Connection refused"))
