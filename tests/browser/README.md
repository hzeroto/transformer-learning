# 浏览器回归检查

`stage-navigation.cjs` 在隔离的无头浏览器中加载本地 `docs/`，验证阶段导航不自动选中节点、全局/阶段标题至少 13px 且完整可读、按宽度排布与纵向滚动、短阶段图不留下大片空白，以及筛选、键盘、全屏/列表切换和显式深链接。覆盖 320–3400px 六种屏宽；不连接个人浏览器，也不修改学习进度。

`default-landing.cjs` 验证首页初次打开和刷新时直接定位到全部阶段的依赖拓扑，进度卡片仍可向上滚动查看，并保留显式知识点深链接。覆盖桌面/手机、有无当前学习任务和减少动态效果设置；测试只替换浏览器内的数据，不修改仓库进度。

使用支持所选 Playwright 版本的 Node.js。可把测试依赖安装在临时目录：

```bash
npm install --prefix /tmp/transformer-ui-tests playwright
NODE_PATH=/tmp/transformer-ui-tests/node_modules node tests/browser/stage-navigation.cjs
NODE_PATH=/tmp/transformer-ui-tests/node_modules node tests/browser/default-landing.cjs
```

运行前通过 Playwright CLI 安装 Chromium，或用 `BROWSER_EXECUTABLE_PATH` 指向本机 Chrome 可执行文件。可设置 `UI_SCREENSHOT_DIR` 保存全局图顶部/底部、输入管线阶段及不同屏宽的截图。
