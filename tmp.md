表情自然度 (Expression Naturalness) 分析
==================================================
视频: /data/AIGC_Video_Reasonableness_Evaluation/data/videos/6706926-hd_1920_1080_25fps.mp4
设备: cuda
视频信息: 1920x1080, 25.0fps, 376 帧
采样: 每 1 帧取 1 帧, 共 376 帧

加载 Py-Feat AU 检测模型...
^CTraceback (most recent call last):
  File "/data/AIGC_Video_Reasonableness_Evaluation/scripts/debug_expression.py", line 406, in <module>
    main()
  File "/data/AIGC_Video_Reasonableness_Evaluation/scripts/debug_expression.py", line 374, in main
    au_per_frame = extract_aus(frames)
  File "/data/AIGC_Video_Reasonableness_Evaluation/scripts/debug_expression.py", line 86, in extract_aus
    extractor._ensure_detector()
  File "/data/AIGC_Video_Reasonableness_Evaluation/src/expression_naturalness/au_extractor.py", line 51, in _ensure_detector
    self._detector = Detector(au_model="xgb")
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/feat/detector.py", line 122, in __init__
    face, landmark, au, emotion, facepose, identity = get_pretrained_models(
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/feat/pretrained.py", line 130, in get_pretrained_models
    download_url(url, get_resource_path(), verbose=verbose)
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/feat/utils/io.py", line 81, in download_url
    return tv_download_url(*args, **kwargs)
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/torchvision/datasets/utils.py", line 121, in download_url
    url = _get_redirect_url(url, max_hops=max_redirect_hops)
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/torchvision/datasets/utils.py", line 66, in _get_redirect_url
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/urllib/request.py", line 216, in urlopen
    return opener.open(url, data, timeout)
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/urllib/request.py", line 519, in open
    response = self._open(req, data)
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/urllib/request.py", line 536, in _open
    result = self._call_chain(self.handle_open, protocol, protocol +
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/urllib/request.py", line 496, in _call_chain
    result = func(*args)
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/urllib/request.py", line 1391, in https_open
    return self.do_open(http.client.HTTPSConnection, req,
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/urllib/request.py", line 1348, in do_open
    h.request(req.get_method(), req.selector, req.data, headers,
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/http/client.py", line 1283, in request
    self._send_request(method, url, body, headers, encode_chunked)
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/http/client.py", line 1329, in _send_request
    self.endheaders(body, encode_chunked=encode_chunked)
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/http/client.py", line 1278, in endheaders
    self._send_output(message_body, encode_chunked=encode_chunked)
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/http/client.py", line 1038, in _send_output
    self.send(msg)
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/http/client.py", line 976, in send
    self.connect()
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/http/client.py", line 1448, in connect
    super().connect()
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/http/client.py", line 942, in connect
    self.sock = self._create_connection(
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/socket.py", line 845, in create_connection
    sock.connect(sa)
