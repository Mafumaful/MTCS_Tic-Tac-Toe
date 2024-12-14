import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys
import math
import random
from copy import deepcopy

# -------------------------
# 1. 定义 TicTacToe 环境
# -------------------------

class TicTacToeEnv(gym.Env):
    """
    自定义井字棋环境，集成 Pygame 渲染
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, screen=None):
        super(TicTacToeEnv, self).__init__()
        # 定义动作空间：0-8 对应棋盘的 9 个位置
        self.action_space = spaces.Discrete(9)
        # 定义观察空间：9 个格子，每个格子可以是 0（空）、1（玩家1）、2（玩家2）
        self.observation_space = spaces.Box(low=0, high=2, shape=(3, 3), dtype=np.int8)
        
        # 保存 Pygame 的屏幕对象
        self.screen = screen

        # 定义棋盘大小和颜色等
        self.WIDTH, self.HEIGHT = 300, 300
        self.LINE_WIDTH = 5
        self.BOARD_ROWS = 3
        self.BOARD_COLS = 3
        self.SQUARE_SIZE = self.WIDTH // self.BOARD_COLS
        self.CIRCLE_RADIUS = self.SQUARE_SIZE // 3
        self.CIRCLE_WIDTH = 5
        self.CROSS_WIDTH = 5
        self.SPACE = self.SQUARE_SIZE // 4

        # 颜色定义
        self.BG_COLOR = (28, 170, 156)
        self.LINE_COLOR = (23, 145, 135)
        self.CIRCLE_COLOR = (239, 231, 200)
        self.CROSS_COLOR = (84, 84, 84)

        self.reset()
        if self.screen:
            self.draw_lines()

    def reset(self):
        # 初始化棋盘
        self.board = np.zeros((3, 3), dtype=np.int8)
        # 设置当前玩家，1 为 AI，2 为 玩家
        self.current_player = 1
        # 游戏是否结束
        self.done = False
        # 结果：0-继续，1-玩家1胜利，2-玩家2胜利，3-平局，4-非法动作
        self.result = 0
        if self.screen:
            self.draw_lines()
        return self.board.copy(), {}

    def step(self, action):
        if self.done:
            raise ValueError("Game is over. Please reset the environment.")

        # 将动作转换为棋盘位置
        row, col = divmod(action, 3)
        if self.board[row, col] != 0:
            # 无效动作，给予负奖励并结束游戏
            self.done = True
            self.result = 4  # 非法动作
            if self.screen:
                self.render()
            return self.board.copy(), -1, self.done, False, {}

        # 执行动作
        self.board[row, col] = self.current_player
        if self.screen:
            self.render()  # 渲染更新

        # 检查是否有胜利
        if self.check_winner(self.current_player):
            self.done = True
            if self.current_player == 1:
                reward = 1
                self.result = 1
            else:
                reward = -10
                self.result = 2
        elif not np.any(self.board == 0):
            # 平局
            self.done = True
            self.result = 3
            reward = 0
        else:
            # 游戏继续
            reward = 0
            self.result = 0
            self.done = False
            # 切换玩家
            self.current_player = 2 if self.current_player == 1 else 1

        return self.board.copy(), reward, self.done, False, {}

    def check_winner(self, player):
        # 检查行、列和对角线
        for i in range(3):
            if np.all(self.board[i, :] == player):
                return True
            if np.all(self.board[:, i] == player):
                return True
        if np.all(np.diag(self.board) == player):
            return True
        if np.all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False

    def draw_lines(self):
        # 填充背景色
        self.screen.fill(self.BG_COLOR)
        # 绘制横线
        pygame.draw.line(self.screen, self.LINE_COLOR, (0, self.SQUARE_SIZE), (self.WIDTH, self.SQUARE_SIZE), self.LINE_WIDTH)
        pygame.draw.line(self.screen, self.LINE_COLOR, (0, 2 * self.SQUARE_SIZE), (self.WIDTH, 2 * self.SQUARE_SIZE), self.LINE_WIDTH)
        # 绘制竖线
        pygame.draw.line(self.screen, self.LINE_COLOR, (self.SQUARE_SIZE, 0), (self.SQUARE_SIZE, self.HEIGHT), self.LINE_WIDTH)
        pygame.draw.line(self.screen, self.LINE_COLOR, (2 * self.SQUARE_SIZE, 0), (2 * self.SQUARE_SIZE, self.HEIGHT), self.LINE_WIDTH)
        pygame.display.update()

    def render(self, mode='human'):
        symbols = {0: ' ', 1: 'X', 2: 'O'}
        for row in range(self.BOARD_ROWS):
            for col in range(self.BOARD_COLS):
                if self.board[row][col] == 1:
                    # 绘制 X
                    start_desc = (col * self.SQUARE_SIZE + self.SPACE, row * self.SQUARE_SIZE + self.SPACE)
                    end_desc = (col * self.SQUARE_SIZE + self.SQUARE_SIZE - self.SPACE, row * self.SQUARE_SIZE + self.SQUARE_SIZE - self.SPACE)
                    pygame.draw.line(self.screen, self.CROSS_COLOR, start_desc, end_desc, self.CROSS_WIDTH)
                    start_asc = (col * self.SQUARE_SIZE + self.SPACE, row * self.SQUARE_SIZE + self.SQUARE_SIZE - self.SPACE)
                    end_asc = (col * self.SQUARE_SIZE + self.SQUARE_SIZE - self.SPACE, row * self.SQUARE_SIZE + self.SPACE)
                    pygame.draw.line(self.screen, self.CROSS_COLOR, start_asc, end_asc, self.CROSS_WIDTH)
                elif self.board[row][col] == 2:
                    # 绘制 O
                    center = (col * self.SQUARE_SIZE + self.SQUARE_SIZE // 2, row * self.SQUARE_SIZE + self.SQUARE_SIZE // 2)
                    pygame.draw.circle(self.screen, self.CIRCLE_COLOR, center, self.CIRCLE_RADIUS, self.CIRCLE_WIDTH)
        pygame.display.update()

# -------------------------
# 2. 定义 MCTS 相关类和函数
# -------------------------

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state  # 当前状态
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.reward = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state['available_actions'])

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.reward / (child.visits + 1)) + c_param * math.sqrt((2 * math.log(self.visits) / (child.visits + 1)))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]
    
    def print_tree(self, depth=0):
        total_visits = sum([child.visits for child in self.children])
        print('  ' * depth + f"Total Visits: {total_visits}")
        for child in self.children:
            print('  ' * depth + f"Action: {child.action}, Reward: {child.reward}, Visits: {child.visits}")

def get_available_actions(board):
    return [i for i in range(9) if board[i // 3, i % 3] == 0]

def simulate(env):
    while True:
        available_actions = get_available_actions(env.board)
        if not available_actions:
            return 0  # 平局
        action = random.choice(available_actions)
        try:
            env.step(action)
        except ValueError:
            break  # 游戏已经结束
        if env.done:
            return env.result
    return env.result

def mcts(env, iterations=10000):
    root_state = {
        'board': deepcopy(env.board),
        'current_player': env.current_player,
        'available_actions': get_available_actions(env.board)
    }
    root = MCTSNode(state=root_state)

    for _ in range(iterations):
        node = root
        # 创建一个没有 screen 的环境复制
        env_copy = TicTacToeEnv()
        env_copy.board = deepcopy(env.board)
        env_copy.current_player = env.current_player
        env_copy.done = env.done
        env_copy.result = env.result

        # 选择（Selection）
        while node.children and not env_copy.done:
            node = node.best_child()
            try:
                env_copy.step(node.action)
            except ValueError:
                break  # 如果游戏已经结束，跳出循环

        # 扩展（Expansion）
        if not env_copy.done:
            available_actions = get_available_actions(env_copy.board)
            for action in available_actions:
                if action not in [child.action for child in node.children]:
                    try:
                        env_copy.step(action)
                        child_state = {
                            'board': deepcopy(env_copy.board),
                            'current_player': env_copy.current_player,
                            'available_actions': get_available_actions(env_copy.board)
                        }
                        child_node = MCTSNode(state=child_state, parent=node, action=action)
                        node.children.append(child_node)
                    except ValueError:
                        continue  # 跳过无效动作

        # 模拟（Simulation）
        simulation_result = simulate(env_copy)

        # 反向传播（Backpropagation）
        while node is not None:
            node.visits += 1
            if simulation_result == 0:
                # 平局不增加奖励
                pass
            else:
                # AI 胜利
                if simulation_result == 1:
                    node.reward += 1
                else:
                    node.reward -= 10
            node = node.parent

    # 选择访问次数最多的子节点
    if root.children:
        best_move = sorted(root.children, key=lambda c: c.visits, reverse=True)[0].action
    else:
        best_move = random.choice(get_available_actions(env.board))
    
    # 打印 MCTS 树
    root.print_tree()
    return best_move

# -------------------------
# 3. 定义 Pygame 渲染和主游戏循环
# -------------------------

def show_result(screen, message):
    font = pygame.font.SysFont(None, 40)
    text = font.render(message, True, (255, 0, 0))
    text_rect = text.get_rect(center=(150, 150))
    screen.blit(text, text_rect)
    pygame.display.update()
    pygame.time.delay(2000)

def main():
    # 初始化 Pygame
    pygame.init()

    # 定义窗口大小和颜色
    WIDTH, HEIGHT = 300, 300
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Tic-Tac-Toe with MCTS and Pygame')

    # 创建环境实例，并传入 Pygame 的屏幕
    env = TicTacToeEnv(screen)
    env.reset()
    env.render()

    player = 1  # 1 为 AI，2 为 对手（玩家）

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # 处理玩家点击
            if event.type == pygame.MOUSEBUTTONDOWN and player == 2 and not env.done:
                mouseX = event.pos[0]  # x 坐标
                mouseY = event.pos[1]  # y 坐标

                clicked_row = mouseY // env.SQUARE_SIZE
                clicked_col = mouseX // env.SQUARE_SIZE

                action = clicked_row * 3 + clicked_col
                if env.board[clicked_row][clicked_col] == 0:
                    env.step(action)

                    if env.done:
                        if env.result == 1:
                            show_result(env.screen, "AI Wins!")
                        elif env.result == 2:
                            show_result(env.screen, "You Win!")
                        elif env.result == 3:
                            show_result(env.screen, "Draw!")
                        elif env.result == 4:
                            show_result(env.screen, "Invalid Move!")
                        env.reset()
                        env.render()
                        player = 1  # 重置为 AI
                        continue

                    player = 1  # 切换到 AI

        # AI 的回合
        if player == 1 and not env.done:
            action = mcts(env, iterations=1000)
            env.step(action)

            if env.done:
                if env.result == 1:
                    show_result(env.screen, "AI Wins!")
                elif env.result == 2:
                    show_result(env.screen, "You Win!")
                elif env.result == 3:
                    show_result(env.screen, "Draw!")
                elif env.result == 4:
                    show_result(env.screen, "Invalid Move!")
                env.reset()
                env.render()
                player = 1  # 重置为 AI
                continue

            player = 2  # 切换到 玩家

        pygame.display.update()

# -------------------------
# 4. 运行主函数
# -------------------------

if __name__ == "__main__":
    main()
