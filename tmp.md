INFO:     127.0.0.1:42078 - "POST /api/evaluate HTTP/1.1" 500 Internal Server Error
ERROR:    Exception in ASGI application
Traceback (most recent call last):
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/uvicorn/protocols/http/h11_impl.py", line 415, in run_asgi
    result = await app(  # type: ignore[func-returns-value]
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/uvicorn/middleware/proxy_headers.py", line 56, in __call__
    return await self.app(scope, receive, send)
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/fastapi/applications.py", line 1135, in __call__
    await super().__call__(scope, receive, send)
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/starlette/applications.py", line 107, in __call__
    await self.middleware_stack(scope, receive, send)
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/starlette/middleware/errors.py", line 186, in __call__
    raise exc
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/starlette/middleware/errors.py", line 164, in __call__
    await self.app(scope, receive, _send)
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/starlette/middleware/cors.py", line 85, in __call__
    await self.app(scope, receive, send)
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/starlette/middleware/exceptions.py", line 63, in __call__
    await wrap_app_handling_exceptions(self.app, conn)(scope, receive, send)
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/starlette/_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/fastapi/middleware/asyncexitstack.py", line 18, in __call__
    await self.app(scope, receive, send)
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/starlette/routing.py", line 716, in __call__
    await self.middleware_stack(scope, receive, send)
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/starlette/routing.py", line 736, in app
    await route.handle(scope, receive, send)
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/starlette/routing.py", line 290, in handle
    await self.app(scope, receive, send)
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/fastapi/routing.py", line 118, in app
    await wrap_app_handling_exceptions(app, request)(scope, receive, send)
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/starlette/_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/fastapi/routing.py", line 104, in app
    response = await f(request)
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/fastapi/routing.py", line 428, in app
    raw_response = await run_endpoint_function(
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/fastapi/routing.py", line 316, in run_endpoint_function
    return await run_in_threadpool(dependant.call, **values)
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/starlette/concurrency.py", line 32, in run_in_threadpool
    return await anyio.to_thread.run_sync(func)
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/anyio/to_thread.py", line 63, in run_sync
    return await get_async_backend().run_sync_in_worker_thread(
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/anyio/_backends/_asyncio.py", line 2518, in run_sync_in_worker_thread
    return await future
  File "/home/ethan/anaconda3/envs/AIGC_Badcase/lib/python3.10/site-packages/anyio/_backends/_asyncio.py", line 1002, in run
    result = context.run(func, *args)
  File "/data/AIGC_Video_Reasonableness_Evaluation/src/api/server.py", line 180, in evaluate
    payload = request.model_dump(exclude_none=True)
AttributeError: 'EvaluateRequest' object has no attribute 'model_dump'
