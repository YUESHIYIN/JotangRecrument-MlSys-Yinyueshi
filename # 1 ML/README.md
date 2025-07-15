# Jotang Summer Recruitment - AI track

## Task 3 - Image Restoration

1. 使用你的模型对 `task_3/image_pairs/blurred` 目录下的图像进行推理（恢复/还原）。
2. 将你的推理结果保存到 `task_3/your_result` 文件夹中。
3. 运行评估脚本：

   ```bash
   python test.py
   ```

---

**注意：运行评估脚本之前，请确保已安装以下依赖包：**

```
opencv-python
scikit-image
```

## Task 5 - Pacman

欢迎来到Pacman（吃豆人）
下载代码、并进入目录后，你可以通过以下命令玩Pacman游戏：

```bash
python pacman.py
```

Pacman生活在一个蓝色的迷宫世界，里面有弯曲的走廊和美味的圆形点心。高效地在这个世界中移动，是Pacman掌握它领域的第一步。

searchAgents.py里最简单的agent叫做GoWestAgent，它总是向西走（一个简单的反射式智能体）。它偶尔能赢：

```bash
python pacman.py --layout testMaze --pacman GoWestAgent
```

但当需要转弯时，这个agent表现很差：

```bash
python pacman.py --layout tinyMaze --pacman GoWestAgent
```

如果Pacman卡住了，可以用 CTRL-c 退出游戏。

很快，你的agent将能解决tinyMaze乃至任何你想要的迷宫。

注意：pacman.py支持多种命令行参数，既可以用长格式（如`--layout`），也可以用短格式（如`-l`）。你可以通过以下命令查看所有选项及默认值：

```bash
python pacman.py -h
```

另外，项目中的commands.txt文件列出了所有命令，方便复制粘贴。在UNIX/Mac OS X中，你甚至可以用 `bash commands.txt` 一次运行所有命令。

---

### Q1：使用深度优先搜索（DFS）寻找固定食物点

在searchAgents.py里有一个完整的SearchAgent，它会规划路径然后一步步执行。搜索算法还没写，这是你的任务。

首先测试SearchAgent是否正常：

```bash
python pacman.py -l tinyMaze -p SearchAgent -a fn=tinyMazeSearch
```

上面命令让SearchAgent用tinyMazeSearch算法（已在search.py中实现）完成迷宫，Pacman应能成功走出迷宫。

接下来你要实现通用搜索函数，帮助Pacman规划路径！算法伪代码见课堂PPT。搜索节点不仅要包含状态，还要包含构造路径所需的信息。

**重要**：所有搜索函数应返回一组动作列表，动作必须是合法的（合法方向，不能穿墙）。

**重要**：务必使用util.py中提供的Stack、Queue和PriorityQueue数据结构！它们的特殊实现对自动评分系统很重要。

提示：DFS、BFS、UCS和A\*只在边界管理策略上不同，先做好DFS，其他较容易。

实现`depthFirstSearch`函数（search.py），写一个图搜索版本，避免重复扩展已访问状态。

运行测试：

```bash
python pacman.py -l tinyMaze -p SearchAgent
python pacman.py -l mediumMaze -p SearchAgent
python pacman.py -l bigMaze -z .5 -p SearchAgent
```

游戏界面会显示搜索路径的探索状态（越早探索的方格颜色越亮）。探索顺序是否符合预期？Pacman是否走过所有探索的方格？

提示：如果用Stack，DFS在mediumMaze的解路径长度应为130（如果后继节点入栈顺序是getSuccessors返回的顺序，反序可能是246）。这是否最优？如果不是，思考DFS哪里出错。

自动评分测试：

```bash
python autograder.py -q q1
```

---

### Q2：广度优先搜索（BFS）

实现`breadthFirstSearch`函数（search.py），同样写图搜索，避免扩展已访问状态。测试方法同DFS。

运行测试：

```bash
python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs
python pacman.py -l bigMaze -p SearchAgent -a fn=bfs -z .5
```

BFS是否找到代价最小解？如果没有，检查实现。

提示：Pacman走得太慢可以加参数 `--frameTime 0`。

注意：如果搜索代码写得通用，修改后应能直接用于八数码问题。

运行：

```bash
python eightpuzzle.py
```

自动评分测试：

```bash
python autograder.py -q q2
```

---

### Q3：变化代价函数

虽然BFS找到最少步数路径，但有时我们想找到“最优”路径，比如避开鬼群的危险区域。

考虑mediumDottedMaze和mediumScaryMaze。通过修改代价函数，可以鼓励Pacman找到不同路径（鬼多区域代价高，食物多区域代价低），Pacman应调整行为。

实现`uniformCostSearch`函数（search.py），写统一代价图搜索。util.py里有你可能用得上的数据结构。

成功运行示例：

```bash
python pacman.py -l mediumMaze -p SearchAgent -a fn=ucs
python pacman.py -l mediumDottedMaze -p StayEastSearchAgent
python pacman.py -l mediumScaryMaze -p StayWestSearchAgent
```

说明：StayEastSearchAgent和StayWestSearchAgent分别用指数代价函数，路径代价极低或极高（详情见searchAgents.py）。

自动评分测试：

```bash
python autograder.py -q q3
```

---

### Q4：A\*搜索

实现A*搜索算法`aStarSearch`函数（search.py）。A*需要一个启发函数作为参数，启发函数接收两个参数：当前状态和问题实例（用于参考）。search.py中已有一个简单启发函数`nullHeuristic`。

你可以用已实现的曼哈顿距离启发函数`manhattanHeuristic`（searchAgents.py）测试A\*在固定位置路径搜索问题中的表现。

示例运行：

```bash
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
```

A*应比统一代价搜索稍快找到最优解（本项目中A*扩展的节点约549，UCS约620，具体数值因优先级相等时的处理略有差异）。

在openMaze上不同搜索策略表现如何？

自动评分测试：

```bash
python autograder.py -q q4
```

### Q5：PPO + 神经网络

PPO（Proximal Policy Optimization）是一种强化学习算法，可以用在很多领域，包括游戏、机器人控制、自动驾驶等。

用 PPO 训练智能体，让它在游戏中学习如何在迷宫中找到食物和躲避幽灵步骤大致如下：

- 用神经网络表示策略（Policy Network），输入环境状态（比如地图、Pac-Man位置、幽灵位置等），输出动作概率分布（向左、向右、向上、向下等）。

- 让智能体根据策略在环境中行动，收集状态-动作-奖励数据。

- 用PPO算法更新神经网络，优化策略以获得更高的长期奖励。

- 重复训练直到智能体表现满意。

请你在 `searchAgents.py` 中实现你的神经网络推理，训练的代码放在另一个 `.py` 文件，在 `searchAgents.py` 中调用。最后录制运行展示视频提交即可。