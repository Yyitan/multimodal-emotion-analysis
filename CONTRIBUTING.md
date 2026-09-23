# 协作规范

## 分支策略

- `main`：稳定基线，只通过 Pull Request 更新；不要直接推送。
- `feature/<name>`：新功能，例如 `feature/audio-encoder-v2`。
- `fix/<name>`：缺陷修复，例如 `fix/ctc-normalization`。
- `experiment/<name>`：尚未确认的实验代码。

## 推荐流程

```bash
git switch main
git pull origin main
git switch -c feature/your-change
# 修改并测试
pytest -q
git add <files>
git commit -m "feat: describe the change"
git push -u origin feature/your-change
```

随后在 GitHub 创建 Pull Request。PR中说明：改动目的、数据接口变化、实际指标、复现命令和风险。

## 合并要求

1. 自动测试通过；
2. 不提交数据、预训练大权重、缓存、凭据、服务器地址或绝对路径；
3. 指标必须来自真实运行结果；
4. 至少一名队友审核后合并；
5. 如修改输入/输出格式，必须同步更新配置、测试与README。
