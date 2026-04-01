# Assignments for CS146S: The Modern Software Developer

This is the home of the assignments for [CS146S: The Modern Software Developer](https://themodernsoftware.dev), taught at Stanford University fall 2025.


1. **学习路径** **：**

* **✅ 先读官方Slides + 中文导读建立框架**
  * [sildes](https://themodernsoftware.dev/)
  * [导读](https://zhuanlan.zhihu.com/p/1985789368133261187)
* **✅ 用GitHub作业仓库动手实践**
* **✅ 结合B站视频辅助理解难点**
* **✅ 加入中文社区（如CS146S_CN交流群）讨论问题**

## Repo Setup

These steps work with Python 3.12.

1. Install Anaconda

   - Download and install: [Anaconda Individual Edition](https://www.anaconda.com/download)
   - Open a new terminal so `conda` is on your `PATH`.
2. Create and activate a Conda environment (Python 3.12)

   ```bash
   conda create -n cs146s python=3.12 -y
   conda activate cs146s
   ```
3. Install Poetry

   ```bash
   curl -sSL https://install.python-poetry.org | python -
   ```
4. Install project dependencies with Poetry (inside the activated Conda env)
   From the repository root:

   ```bash
   poetry install --no-interaction
   ```
