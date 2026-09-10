# 浏览器回归检查

`stage-navigation.cjs` 在隔离的无头浏览器中加载本地 `docs/`，验证左侧阶段导航、中间节点集合、右侧详情和筛选范围一致。它不连接个人浏览器，也不修改学习进度。

使用支持所选 Playwright 版本的 Node.js。可把测试依赖安装在临时目录：

```bash
npm install --prefix /tmp/transformer-ui-tests playwright
NODE_PATH=/tmp/transformer-ui-tests/node_modules node tests/browser/stage-navigation.cjs
```

运行前通过 Playwright CLI 安装 Chromium，或用 `BROWSER_EXECUTABLE_PATH` 指向本机 Chrome 可执行文件。可设置 `UI_SCREENSHOT_DIR` 保存两个不同阶段的截图。
