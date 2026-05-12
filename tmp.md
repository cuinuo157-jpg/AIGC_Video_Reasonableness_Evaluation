原因很明确：这次远程并没有打到华为服务，而是被错误路由到了本机 localhost:8201/v1，所以报了
  Connection refused。这不是模型返回错误，也不是鉴权错误，而是目标地址上根本没有服务在监听。

  关键链路是这样的：

  - 你运行的是 debug_dynamics.py
  - 当前脚本默认 provider 确实是 huawei_custom，默认华为地址也是
    http://aitest-beta.rnd.huawei.com/v1，见 /D:/workspace/projects/activate/AIGC_Video_Reaso
    nableness_Evaluation/scripts/debug_dynamics.py
  - 但它在解析 --mllm-base-url 时，默认值来源是：
      - MLLM_API_BASE_URL
      - DASHSCOPE_BASE_URL
      - VLLM_OPENAI_BASE_URL
  - 也就是如果远程环境里设置了 VLLM_OPENAI_BASE_URL=http://localhost:8201/v1，哪怕 provider 是
    huawei_custom，args.mllm_base_url 也会先拿到这个值
  - 然后在 build_mllm_client() 里，因为 base_url 已经非空，就不会再回退到华为默认地址，见 /D:/
    workspace/projects/activate/AIGC_Video_Reasonableness_Evaluation/scripts/debug_dynamics.py
  - 最终 huawei_custom 调用的就是这个错误地址，见 /D:/workspace/projects/activate/
    AIGC_Video_Reasonableness_Evaluation/src/mllm/client.py

  所以本质问题是：

  - provider = huawei_custom
  - 但 base_url 被环境变量里的 VLLM_OPENAI_BASE_URL 污染成了 http://localhost:8201/v1

  直接验证方式：

  echo $MLLM_PROVIDER
  echo $MLLM_API_BASE_URL
  echo $VLLM_OPENAI_BASE_URL

  大概率你会看到：

  - MLLM_PROVIDER=huawei_custom 或命令行传了这个
  - VLLM_OPENAI_BASE_URL=http://localhost:8201/v1

  临时修复有 3 种，任选其一：

  1. 显式传对的地址

  python scripts/debug_dynamics.py ... --enable-mllm --mllm-provider huawei_custom --mllm-bas
  e-url http://aitest-beta.rnd.huawei.com/v1 --mllm-service-name simple_client

  2. 设置正确环境变量

  export MLLM_API_BASE_URL=http://aitest-beta.rnd.huawei.com/v1
  export MLLM_API_SERVICE_NAME=simple_client

  3. 清掉会污染它的 vllm 变量

  unset VLLM_OPENAI_BASE_URL

  结论：

  - 当前报错不是华为服务不可用。
  - 是远程环境变量把 huawei_custom 的 base_url 指到了本机 8201。
  - 如果你希望彻底避免这类问题，应该改代码：huawei_custom 不要再读取 VLLM_OPENAI_BASE_URL 作为
    兜底来源。

  如果你要，我可以直接把这段 provider-specific 的 base_url 解析逻辑改掉。
