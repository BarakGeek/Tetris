import pygame
import random

pygame.init()

O = [[[0, 6, 6, 0],
      [0, 6, 6, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0]],

     [[0, 6, 6, 0],
      [0, 6, 6, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0]],

     [[0, 6, 6, 0],
      [0, 6, 6, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0]],

     [[0, 6, 6, 0],
      [0, 6, 6, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0]]]

I = [[[0, 1, 0, 0],
      [0, 1, 0, 0],
      [0, 1, 0, 0],
      [0, 1, 0, 0]],

     [[0, 0, 0, 0],
      [1, 1, 1, 1],
      [0, 0, 0, 0],
      [0, 0, 0, 0]],

     [[0, 0, 1, 0],
      [0, 0, 1, 0],
      [0, 0, 1, 0],
      [0, 0, 1, 0]],

     [[0, 0, 0, 0],
      [0, 0, 0, 0],
      [1, 1, 1, 1],
      [0, 0, 0, 0]]]

Z = [[[2, 2, 0, 0],
      [0, 2, 2, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0]],

     [[0, 0, 2, 0],
      [0, 2, 2, 0],
      [0, 2, 0, 0],
      [0, 0, 0, 0]],

     [[0, 0, 0, 0],
      [2, 2, 0, 0],
      [0, 2, 2, 0],
      [0, 0, 0, 0]],

     [[0, 2, 0, 0],
      [2, 2, 0, 0],
      [2, 0, 0, 0],
      [0, 0, 0, 0]]]

S = [[[0, 5, 5, 0],
      [5, 5, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0]],

     [[0, 5, 0, 0],
      [0, 5, 5, 0],
      [0, 0, 5, 0],
      [0, 0, 0, 0]],

     [[0, 0, 0, 0],
      [0, 5, 5, 0],
      [5, 5, 0, 0],
      [0, 0, 0, 0]],

     [[5, 0, 0, 0],
      [5, 5, 0, 0],
      [0, 5, 0, 0],
      [0, 0, 0, 0]]]

J = [[[4, 0, 0, 0],
      [4, 4, 4, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0]],

     [[0, 4, 4, 0],
      [0, 4, 0, 0],
      [0, 4, 0, 0],
      [0, 0, 0, 0]],

     [[0, 0, 0, 0],
      [4, 4, 4, 0],
      [0, 0, 4, 0],
      [0, 0, 0, 0]],

     [[0, 4, 0, 0],
      [0, 4, 0, 0],
      [4, 4, 0, 0],
      [0, 0, 0, 0]]]

L = [[[0, 0, 7, 0],
      [7, 7, 7, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0]],

     [[0, 7, 0, 0],
      [0, 7, 0, 0],
      [0, 7, 7, 0],
      [0, 0, 0, 0]],

     [[0, 0, 0, 0],
      [7, 7, 7, 0],
      [7, 0, 0, 0],
      [0, 0, 0, 0]],

     [[7, 7, 0, 0],
      [0, 7, 0, 0],
      [0, 7, 0, 0],
      [0, 0, 0, 0]]]

T = [[[0, 3, 0, 0],
      [3, 3, 3, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0]],

     [[0, 3, 0, 0],
      [0, 3, 3, 0],
      [0, 3, 0, 0],
      [0, 0, 0, 0]],

     [[0, 0, 0, 0],
      [3, 3, 3, 0],
      [0, 3, 0, 0],
      [0, 0, 0, 0]],

     [[0, 3, 0, 0],
      [3, 3, 0, 0],
      [0, 3, 0, 0],
      [0, 0, 0, 0]]]

sky_rows = 4
game_rows = 20
floor_rows = 4
dummy_rows = sky_rows + floor_rows

left_dummy_col = 4
game_columns = 10
right_dummy_col = 4
rows = + game_rows + dummy_rows
columns = game_columns + left_dummy_col + right_dummy_col

shapes_lst = [O, I, Z, T, S, J, L]
line_cleared = 0


class Blocks:
    shapes_str_lst = ["O", "I", "Z", "S", "J", "L", "T"]

    def __init__(self, col, row, shape, rotation):
        self.row = row
        self.col = col
        self.shape = shapes_lst[shape]
        self.rotation = rotation % 4
        self.num_shape = shape
        self.fix = 0

    def __repr__(self):
        return f"Block(col={self.col},row={self.row},shape={self.num_shape},rotation={self.rotation})"

    def create_block(self, grid):

        for i in range(len(self.shape)):
            for j in range(len(self.shape[self.rotation][i])):
                if self.shape[self.rotation][i][j] == 0:
                    pass
                else:
                    grid[self.row + i][self.col + j] += self.shape[self.rotation][i][j]

    def remove_block(self, grid):
        for i in range(len(self.shape)):
            for j in range(len(self.shape[self.rotation][i])):
                if self.shape[self.rotation][i][j] == 0:
                    pass
                else:
                    grid[self.row + i][self.col + j] -= self.shape[self.rotation][i][j]

    def valid_move(self, grid):
        for i in range(len(self.shape)):
            for j in range(len(self.shape[self.rotation][i])):
                if self.shape[self.rotation][i][j] == 0:
                    pass
                else:
                    cell = grid[self.row + i][self.col + j] + self.shape[self.rotation][i][j]
                    if cell != 0 and cell != self.shape[self.rotation][i][j]:
                        return False
        return True

    def col_bounds(self):
        left_wall = 0
        block_place = 0
        right_wall = 0

        for i in range(len(self.shape[self.rotation])):
            col_sum = 0
            for j in range(len(self.shape[self.rotation][i])):
                col_sum += self.shape[self.rotation][j][i]
            if col_sum == 0 and block_place != 0:
                right_wall += 1
            elif col_sum == 0:
                left_wall += 1
            elif col_sum != 0:
                block_place += 1

        return left_wall, block_place, right_wall

    def row_bounds(self):
        top_wall = 0
        block_place = 0
        bottom_wall = 0

        for i in range(len(self.shape[self.rotation])):
            row_sum = 0
            for j in range(len(self.shape[self.rotation][i])):
                row_sum += self.shape[self.rotation][i][j]
            if row_sum == 0 and block_place != 0:
                bottom_wall += 1
            elif row_sum == 0:
                top_wall += 1
            elif row_sum != 0:
                block_place += 1

        return bottom_wall

    def rotate(self, grid):
        var_rotation = self.rotation
        self.remove_block(grid)
        current_floor = game_rows + self.row_bounds()
        floor_flag = False
        if self.row == current_floor:
            floor_flag = True
        self.rotation = (self.rotation + 1) % 4
        if self.valid_move(grid):
            if self.col < left_dummy_col:
                self.col = left_dummy_col
            elif self.col > right_dummy_col + game_columns:
                self.col = right_dummy_col + game_columns
            if floor_flag:
                self.row = game_rows + self.row_bounds()
            self.create_block(grid)
        else:
            self.rotation = var_rotation
            self.create_block(grid)

    def drop_once(self, grid):
        self.remove_block(grid)
        self.row += 1
        if self.valid_move(grid):
            self.create_block(grid)
        else:
            self.row -= 1
            self.create_block(grid)

    def hard_drop(self, grid):
        floor = game_rows + self.row_bounds() - 1

        while self.row <= floor:
            current_row = self.row

            self.drop_once(grid)
            if self.row == current_row:
                break

    @staticmethod
    def score(line_clear):
        if line_clear == 0:
            return 0
        elif line_clear == 1:
            return 40
        elif line_clear == 2:
            return 100
        elif line_clear == 3:
            return 300
        elif line_clear == 4:
            return 1200
        else:
            return 0

    @staticmethod
    def clear_rows(grid):
        lines = 0
        cleared_grid = grid[:sky_rows]

        for i in range(sky_rows, game_rows + floor_rows):
            full_row = 0
            for j in range(left_dummy_col, game_columns + right_dummy_col):
                if grid[i][j] != 0:
                    full_row += 1

            if full_row == 10:
                lines += 1
            else:
                cleared_grid.append(grid[i])

        for i in range(lines):
            cleared_grid.insert(0, [0 for _ in range(columns)])

        for i in range(floor_rows):
            cleared_grid.append(grid[i + game_rows + floor_rows])

        points = Blocks.score(lines)

        return cleared_grid, lines, points

    def move_right(self, grid):
        fix_move = self.col_bounds()
        check = self.col + 1
        if check > fix_move[2] + game_columns:
            return
        self.remove_block(grid)
        self.col += 1
        if self.col <= fix_move[2] + game_columns and self.valid_move(grid):
            self.create_block(grid)
        else:
            self.col -= 1
            self.create_block(grid)

    def move_left(self, grid):
        fix_move = self.col_bounds()
        self.remove_block(grid)
        self.col -= 1
        if self.col + fix_move[0] >= left_dummy_col and self.valid_move(grid):
            self.create_block(grid)
        else:
            self.col += 1
            self.create_block(grid)

    def move_block(self, grid, direction):

        if direction == -1:
            self.move_left(grid)
        if direction == 0:
            pass
        if direction == 1:
            self.move_right(grid)

    def drop_move_left(self, grid):
        floor = game_rows + self.row_bounds() - 1
        move_flag = True
        while self.row <= floor:
            current_row = self.row
            self.drop_once(grid)
            if self.row == current_row:
                if move_flag:
                    self.move_left(grid)
                    move_flag = False
                else:
                    break
        if move_flag:
            self.move_left(grid)

    def drop_move_right(self, grid):
        floor = game_rows + self.row_bounds() - 1
        move_flag = True
        while self.row <= floor:
            current_row = self.row
            self.drop_once(grid)
            if self.row == current_row:
                if move_flag:
                    self.move_right(grid)
                    move_flag = False
                else:
                    break
        if move_flag:
            self.move_right(grid)

    @staticmethod
    def grid_stats(grid):
        score = Blocks.clear_rows(grid)[2]

        cols_heights = [0 for _ in range(10)]
        for i in range(sky_rows, game_rows + floor_rows):
            for j in range(left_dummy_col, game_columns + right_dummy_col):
                if grid[i][j] != 0:
                    if cols_heights[j - left_dummy_col] == 0:
                        cols_heights[j - left_dummy_col] = game_rows + floor_rows - i
        max_col_height = max(cols_heights)

        holes = [0 for _ in range(10)]
        for j in range(left_dummy_col, game_columns + right_dummy_col):
            for i in range(sky_rows, game_rows + floor_rows):
                if game_rows + floor_rows - i < cols_heights[j - left_dummy_col]:
                    if grid[i][j] == 0:
                        holes[j - left_dummy_col] += 1

        smooth = 0
        for i in range(len(cols_heights) - 1):
            col_smooth = abs(cols_heights[i] - cols_heights[i + 1])
            smooth += col_smooth

        pillars = [0 for _ in range(10)]
        for i in range(1, len(pillars) - 1):
            if cols_heights[i - 1] - 3 >= cols_heights[i] and cols_heights[i + 1] - 2 >= cols_heights[i]:
                pillars[i] += (cols_heights[i - 1] * cols_heights[i + 1] * max_col_height)

        if cols_heights[0] + 3 <= cols_heights[1]:
            pillars[0] += (cols_heights[1]) * 2

        if cols_heights[9] + 3 <= cols_heights[8]:
            pillars[9] += (cols_heights[8]) * 2

        holes_bias = [0 for _ in range(10)]
        for j in range(left_dummy_col, game_columns + right_dummy_col):
            c = 1
            for i in range(sky_rows, game_rows + floor_rows):
                if game_rows + floor_rows - i < cols_heights[j - left_dummy_col]:
                    if grid[i][j] == 0:
                        holes[j - left_dummy_col] += c
                        c += 1
        bias = 4
        for i in range(len(holes)):
            if pillars[i] != 0:
                holes[i] *= (cols_heights[i] ** bias) * smooth ** holes_bias[i]

            if i < 3:
                bias -= 1
            if i >= 6:
                bias += 1

        return [score, holes, max_col_height, pillars, smooth]

    @staticmethod
    def current_positions_stats(grid, shape_1):

        sim_pos = []
        sim_stats = []
        current_holes = sum(Blocks.grid_stats(grid)[1])
        for i in range(4):
            block = Blocks(7, 0, shape_1, i)
            fix_pos_1 = block.col_bounds()
            fix_start_1 = fix_pos_1[0]
            fix_end_1 = fix_pos_1[2]

            for j in range(left_dummy_col - fix_start_1, game_columns + fix_end_1 + 1):
                block_mid = Blocks(j, 0, shape_1, i)
                block_mid.create_block(grid)
                block_mid.hard_drop(grid)
                drop_col = block_mid.col
                drop_row = block_mid.row - 4
                # mid

                sim_pos.append([drop_col, drop_row, block_mid.col, block_mid.row, i])
                sim_stats.append(Blocks.grid_stats(grid))

                block_mid.remove_block(grid)

                block_left = Blocks(drop_col, drop_row, shape_1, i)
                block_left.create_block(grid)
                block_left.drop_move_left(grid)
                stats = Blocks.grid_stats(grid)
                holes = sum(stats[1])
                # left

                if current_holes == 0:
                    pass
                else:
                    if holes < current_holes:
                        sim_stats.append(stats)
                        sim_pos.append([drop_col, drop_row, block_left.col, block_left.row, i])

                block_left.remove_block(grid)

                block_right = Blocks(drop_col, drop_row, shape_1, i)
                block_right.create_block(grid)
                block_right.drop_move_right(grid)
                stats = Blocks.grid_stats(grid)
                holes = sum(stats[1])
                # right

                if current_holes == 0:
                    pass
                else:
                    if holes < current_holes:
                        sim_stats.append(stats)
                        sim_pos.append([drop_col, drop_row, block_right.col, block_right.row, i])

                block_right.remove_block(grid)

        min_holes = sum(sim_stats[0][1])

        for i in range(len(sim_stats)):
            if sum(sim_stats[i][1]) <= min_holes:
                min_holes = (sum(sim_stats[i][1]))

        min_holes_stats = []
        min_holes_pos = []

        for i in range(len(sim_stats)):
            if sum(sim_stats[i][1]) <= min_holes:
                min_holes_stats.append(sim_stats[i])
                min_holes_pos.append(sim_pos[i])

        sim_stats = min_holes_stats
        sim_pos = min_holes_pos

        min_smooth = (sim_stats[0][4])

        for i in range(len(sim_stats)):
            if (sim_stats[i][4]) <= min_smooth:
                min_smooth = sim_stats[i][4]

        min_smooth_stats = []
        min_smooth_pos = []

        for i in range(len(sim_stats)):
            if (sim_stats[i][4]) <= min_smooth:
                min_smooth_stats.append(sim_stats[i])
                min_smooth_pos.append(sim_pos[i])

        sim_stats = min_smooth_stats
        sim_pos = min_smooth_pos

        pos_col_heights = []
        for i in range(len(sim_stats)):
            if (sim_stats[i][2]) not in pos_col_heights:
                pos_col_heights.append((sim_stats[i][2]))

        pos_col_heights.sort()
        half_cols = pos_col_heights[:len(pos_col_heights) // 4]
        if len(half_cols) >= 1:
            clean_sim_pos = []
            clean_sim_stats = []
            for i in range(len(sim_stats)):
                if (sim_stats[i][2]) in half_cols:
                    clean_sim_pos.append(sim_pos[i])
                    clean_sim_stats.append(sim_stats[i])

            return [clean_sim_pos, clean_sim_stats]
        return [sim_pos, sim_stats]

    @staticmethod
    def current_and_next_positions_stats(grid, shape_1, shape_2):
        positions_stats_shape_1 = Blocks.current_positions_stats(grid, shape_1)
        positions_shape_1 = positions_stats_shape_1[0]

        sim_pos_stats_shape_2 = []
        for p in range(len(positions_shape_1)):
            shape_1_col = positions_shape_1[p][0]
            shape_1_row = positions_shape_1[p][1]
            shape_1_move_col = positions_shape_1[p][2]
            shape_1_rot = positions_shape_1[p][4]
            block_1 = Blocks(shape_1_col, shape_1_row, shape_1, shape_1_rot)
            block_1.create_block(grid)

            if shape_1_col - shape_1_move_col == 1:
                block_1.drop_move_left(grid)
            elif shape_1_col - shape_1_move_col == 0:
                block_1.hard_drop(grid)
            elif shape_1_col - shape_1_move_col == -1:
                block_1.drop_move_right(grid)

            sim_stats = []
            current_holes = sum(Blocks.grid_stats(grid)[1])
            for i in range(4):
                block_2 = Blocks(7, 0, shape_2, i)
                fix_pos_2 = block_2.col_bounds()
                fix_start_2 = fix_pos_2[0]
                fix_end_2 = fix_pos_2[2]
                for j in range(left_dummy_col - fix_start_2, game_columns + fix_end_2 + 1):
                    block_mid_2 = Blocks(j, 0, shape_2, i)
                    block_mid_2.create_block(grid)
                    block_mid_2.hard_drop(grid)
                    # mid
                    sim_stats.append(Blocks.grid_stats(grid))
                    block_mid_2.remove_block(grid)

                    drop_col = block_mid_2.col
                    drop_row = block_mid_2.row - 4
                    # left

                    block_left_2 = Blocks(drop_col, drop_row, shape_1, i)
                    block_left_2.create_block(grid)
                    block_left_2.drop_move_left(grid)
                    stats = Blocks.grid_stats(grid)
                    holes = sum(stats[1])

                    if current_holes == 0:
                        pass
                    else:
                        if holes < current_holes:
                            sim_stats.append(Blocks.grid_stats(grid))
                    block_left_2.remove_block(grid)

                    block_right_2 = Blocks(drop_col, drop_row, shape_1, i)
                    block_right_2.create_block(grid)

                    block_right_2.drop_move_right(grid)
                    stats = Blocks.grid_stats(grid)
                    holes = sum(stats[1])

                    if current_holes == 0:
                        pass
                    else:
                        if holes < current_holes:
                            sim_stats.append(Blocks.grid_stats(grid))

                    block_right_2.remove_block(grid)
            sim_pos_stats_shape_2.append(sim_stats)
            block_1.remove_block(grid)

        return [positions_stats_shape_1, sim_pos_stats_shape_2]

    @staticmethod
    def find_fitness(grid, shape_1, shape_2):
        positions_stats_1_stats_2 = Blocks.current_and_next_positions_stats(grid, shape_1, shape_2)

        positions_stats_1 = positions_stats_1_stats_2[0]
        positions_1 = positions_stats_1[0]
        stats_1 = positions_stats_1[1]

        stats_2 = positions_stats_1_stats_2[1]

        score_shape_1 = stats_1[0][0]
        holes_shape_1 = stats_1[0][1]
        max_col_shape_1 = stats_1[0][2]
        pillars_shape_1 = stats_1[0][3]
        smooth_shape_1 = stats_1[0][4]

        score_shape_2 = stats_2[0][0][0]
        holes_shape_2 = stats_2[0][0][1]
        max_col_shape_2 = stats_2[0][0][2]
        pillars_shape_2 = stats_2[0][0][3]
        smooth_shape_2 = stats_2[0][0][4]

        fitness_shape_1 = score_shape_1 // 10 - (
                sum(holes_shape_1) + max_col_shape_1 ** 2 + sum(pillars_shape_1) + smooth_shape_1)
        fitness_shape_2 = score_shape_2 - (
                sum(holes_shape_2) + max_col_shape_2 ** 2 + sum(pillars_shape_2) + smooth_shape_2)

        fitness = fitness_shape_1 + fitness_shape_2
        bias = Blocks.grid_stats(grid)[2]
        fit_moves = positions_1[0]
        for i in range(len(positions_1)):
            fit_score_shape_1 = stats_1[i][0]
            fit_holes_shape_1 = stats_1[i][1]
            fit_max_col_shape_1 = stats_1[i][2]
            fit_pillars_shape_1 = stats_1[i][3]
            fit_smooth_shape_1 = stats_1[i][4]
            if fit_score_shape_1 >= 1200:
                return positions_1[i]

            if bias >= 8:
                if fit_score_shape_1 >= 300:
                    return positions_1[i]

            if bias >= 10:
                if fit_score_shape_1 >= 100:
                    return positions_1[i]

            if bias >= 12:
                if fit_score_shape_1 >= 40:
                    return positions_1[i]
            cal_fitness_shape_1 = fit_score_shape_1 // 10 - (
                    sum(fit_holes_shape_1) + fit_max_col_shape_1 + sum(fit_pillars_shape_1) + fit_smooth_shape_1)

            for j in range(len(stats_2[i])):
                fit_score_shape_2 = stats_2[i][j][0]
                fit_holes_shape_2 = stats_2[i][j][1]
                fit_max_col_shape_2 = stats_2[i][j][2]
                fit_pillars_shape_2 = stats_2[i][j][3]
                fit_smooth_shape_2 = stats_2[i][j][4]

                cal_fitness_shape_2 = fit_score_shape_2 // 5 - (sum(fit_holes_shape_2) + fit_max_col_shape_2 ** 2 + sum(
                    fit_pillars_shape_2) + fit_smooth_shape_2)
                cal_fitness_shape_1_2 = cal_fitness_shape_1 + cal_fitness_shape_2

                if cal_fitness_shape_1_2 >= fitness:
                    fitness = cal_fitness_shape_1_2
                    fit_moves = positions_1[i]

        return fit_moves


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
Grey = (220, 220, 220)
BG_GREY = (241, 239, 242)


def create_draw_grid(screen):
    def grid():
        x1 = 400
        x2 = 800
        y = 800
        cell_size = 40
        horizontal_lines = []
        for i in range(game_rows):
            horizontal_lines.append(((x1, y), (x2, y)))
            y -= cell_size

        y1 = 40
        y2 = 840
        x = 440
        vertical_lines = []
        for i in range(game_columns):
            vertical_lines.append(((x, y1), (x, y2)))
            x += cell_size
        return horizontal_lines, vertical_lines

    board = grid()
    grid_horizontal = board[0]
    grid_vertical = board[1]

    for i in grid_horizontal:
        pygame.draw.line(screen, Grey, i[0], i[1], 2)
    for i in grid_vertical:
        pygame.draw.line(screen, Grey, i[0], i[1], 2)


def draw_window(screen, grid, next_piece, line_cleared, points, cleared_1_line, cleared_2_line, cleared_3_line,
                cleared_4_line,blocks_used):
    screen.fill(BG_GREY)
    pygame.draw.rect(screen, WHITE, (400, 40, 400, 800))
    create_draw_grid(screen)
    pygame.draw.rect(screen, BLACK, pygame.Rect(398, 38, 405, 805), 3, 3)
    radios = 0
    size_black = 39
    size_color = 36
    pygame.draw.rect(screen, BLACK, (900, 260, 200, 200))
    pygame.draw.rect(screen, WHITE, (905, 265, 190, 190))

    for i in range(sky_rows, game_rows + floor_rows):
        for j in range(left_dummy_col, game_columns + right_dummy_col):
            if grid[i][j] != 0:
                color = 0
                if grid[i][j] == 1:
                    color = (0, 255, 255)
                elif grid[i][j] == 2:
                    color = (255, 0, 0)
                elif grid[i][j] == 3:
                    color = (128, 0, 128)
                elif grid[i][j] == 4:
                    color = (0, 0, 255)
                elif grid[i][j] == 5:
                    color = (0, 255, 0)
                elif grid[i][j] == 6:
                    color = (255, 255, 0)
                elif grid[i][j] == 7:
                    color = (255, 127, 0)

                pygame.draw.rect(screen, BLACK, (401 + 40 * (j - 4), 41 + 40 * (i - 4), size_black, size_black),
                                 border_radius=radios)
                pygame.draw.rect(screen, color, (402 + 40 * (j - 4), 42 + 40 * (i - 4), size_color, size_color),
                                 border_radius=radios)

    next_block = Blocks(0, 0, next_piece, 0)
    buffer_row = 0
    buffer_col = 20
    for i in range(len(next_block.shape[0])):
        for j in range((len(next_block.shape[0][i]))):
            cell = next_block.shape[0][i][j]
            if cell != 0:
                color = 0
                if cell == 1:
                    color = (0, 255, 255)
                    buffer_col = -10
                    buffer_row = 0
                elif cell == 2:
                    color = (255, 0, 0)
                elif cell == 3:
                    color = (128, 0, 128)
                elif cell == 4:
                    color = (0, 0, 255)
                elif cell == 5:
                    color = (0, 255, 0)
                elif cell == 6:
                    color = (255, 255, 0)
                    buffer_row = 20
                    buffer_col = 30
                elif cell == 7:
                    color = (255, 127, 0)

                pygame.draw.rect(screen, BLACK, (
                    1101 - buffer_row + 40 * (j - 4), 451 + buffer_col + 40 * (i - 4), size_black, size_black),
                                 border_radius=radios)
                pygame.draw.rect(screen, color, (
                    1102 - buffer_row + 40 * (j - 4), 452 + buffer_col + 40 * (i - 4), size_color, size_color),
                                 border_radius=radios)

    font = pygame.font.SysFont('javanesetext', 38)

    text_score = font.render(f'Score: {points:08d}', True, BLACK)
    text_score_rect = text_score.get_rect()
    text_score_rect.center = (1000, 100)

    text_lines = font.render(f'Lines: {line_cleared:06d}', True, BLACK)
    text_lines_rect = text_lines.get_rect()
    text_lines_rect.center = (980, 200)

    text_block_used = font.render(f'Blocks: {blocks_used:06d}', True, BLACK)
    text_block_used_rect = text_block_used.get_rect()
    text_block_used_rect.center = (1000, 550)

    def percentage(part, whole):
        percentage = 100 * float(part) / float(whole)
        return percentage


    font_lines = pygame.font.SysFont('javanesetext', 38)

    text_cleared_1_line = font_lines.render(f'One line: {cleared_1_line:06d}', True, BLACK)
    text_cleared_1_line_rect = text_cleared_1_line.get_rect()
    text_cleared_1_line_rect.center = (200, 100)

    text_cleared_2_line = font_lines.render(f'Two line: {cleared_2_line:06d}', True, BLACK)
    text_cleared_2_line_rect = text_cleared_2_line.get_rect()
    text_cleared_2_line_rect.center = (200, 250)

    text_cleared_3_line = font_lines.render(f'Three line: {cleared_3_line:06d}', True, BLACK)
    text_cleared_3_line_rect = text_cleared_3_line.get_rect()
    text_cleared_3_line_rect.center = (200, 400)

    text_cleared_4_line = font_lines.render(f'Four line: {cleared_4_line:06d}', True, BLACK)
    text_cleared_4_line_rect = text_cleared_4_line.get_rect()
    text_cleared_4_line_rect.center = (200, 550)

    cleared_1_line_pct=0
    cleared_2_line_pct=0
    cleared_3_line_pct=0
    cleared_4_line_pct=0
    if line_cleared==0:
        pass
    else:
        cleared_1_line_pct=percentage(cleared_1_line,line_cleared)
        cleared_2_line_pct=percentage(cleared_2_line,line_cleared)*2
        cleared_3_line_pct=percentage(cleared_3_line,line_cleared)*3
        cleared_4_line_pct=percentage(cleared_4_line,line_cleared)*4

    font_lines_pct = pygame.font.SysFont('javanesetext', 20)

    text_cleared_1_line_pct = font_lines_pct.render(f'One lines percentage: {cleared_1_line_pct:.2f}%', True, BLACK)
    text_cleared_1_line_pct_rect = text_cleared_1_line_pct.get_rect()
    text_cleared_1_line_pct_rect.center = (200, 150)


    text_cleared_2_line_pct = font_lines_pct.render(f'Two lines percentage: {cleared_2_line_pct:.2f}%', True, BLACK)
    text_cleared_2_line_pct_rect = text_cleared_2_line_pct.get_rect()
    text_cleared_2_line_pct_rect.center = (200, 300)


    text_cleared_3_line_pct = font_lines_pct.render(f'Three lines percentage: {cleared_3_line_pct:.2f}%', True, BLACK)
    text_cleared_3_line_pct_rect = text_cleared_3_line_pct.get_rect()
    text_cleared_3_line_pct_rect.center = (200, 450)

    text_cleared_4_line_pct = font_lines_pct.render(f'Four lines percentage: {cleared_4_line_pct:.2f}%', True, BLACK)
    text_cleared_4_line_pct_rect = text_cleared_4_line_pct.get_rect()
    text_cleared_4_line_pct_rect.center = (200, 600)



    screen.blit(text_cleared_1_line_pct, text_cleared_1_line_pct_rect)
    screen.blit(text_cleared_2_line_pct, text_cleared_2_line_pct_rect)
    screen.blit(text_cleared_3_line_pct, text_cleared_3_line_pct_rect)
    screen.blit(text_cleared_4_line_pct, text_cleared_4_line_pct_rect)





    screen.blit(text_score, text_score_rect)
    screen.blit(text_lines, text_lines_rect)
    screen.blit(text_block_used, text_block_used_rect)

    screen.blit(text_cleared_1_line, text_cleared_1_line_rect)
    screen.blit(text_cleared_2_line, text_cleared_2_line_rect)
    screen.blit(text_cleared_3_line, text_cleared_3_line_rect)
    screen.blit(text_cleared_4_line, text_cleared_4_line_rect)

    pygame.display.update()

def pieces_order(lst, ind):
    next_ind = ind + 1

    if next_ind <= 6:
        piece = lst[ind]
        next_piece = lst[next_ind]
        return lst, piece, next_piece

    if next_ind == 7:
        piece = lst[ind]
        random.shuffle(lst)
        ind = 0
        next_piece = lst[ind]
        return lst, piece, next_piece


def print_grid_t(grid):
    print()
    for i in range(4, len(grid) - 4):
        st = ""
        if i <= 9:
            st = " "
        print(i, st, (grid[i][left_dummy_col:game_columns + right_dummy_col]), i)
    print("    ------------------------------")
    print("   ", [i for i in range(10)])
    print()


def print_grid(grid):
    for i in range(len(grid)):
        print(grid[i], i)
    print()


def tetris():
    grid = []
    lines = 0
    one_line = 0
    two_line = 0
    three_line = 0
    four_line = 0
    points = 0
    for i in range(rows):
        if i < rows - 2:
            grid.append([0 for _ in range(columns)])
        else:
            grid.append([9 for _ in range(columns)])

    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 900

    FPS = 60
    ticks = 2
    clock = pygame.time.Clock()
    clock.tick(FPS)
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

    run = True
    play = True
    drop_time = 0
    check_lost = 0
    shuffled_blocks = [i for i in range(7)]
    random.shuffle(shuffled_blocks)
    ind = 0

    block = Blocks(7, 0, shuffled_blocks[ind], 0)
    next_piece = shuffled_blocks[ind + 1]
    fitness_moves = Blocks.find_fitness(grid, block.num_shape, next_piece)
    fit_col = fitness_moves[0]
    fit_row = fitness_moves[1]
    fit_col_move = fitness_moves[0] - fitness_moves[2]
    fit_row_move = fitness_moves[3]
    fit_rot = fitness_moves[4]
    block.create_block(grid)

    block_used = 1

    while run:
        drop_time += 1
        floor = game_rows + block.row_bounds() - 1
        current_cleared_lines = lines


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.KEYDOWN and block.row >= 2:
                if event.key == pygame.K_ESCAPE:
                    run = False
                    break

                if event.key == pygame.K_LEFT:
                    block.move_left(grid)

                if event.key == pygame.K_RIGHT:
                    block.move_right(grid)

                if event.key == pygame.K_DOWN:
                    block.hard_drop(grid)
                    break

                if event.key == pygame.K_UP and block.shape != 0 and block.row < floor:
                    block.rotate(grid)
                    floor = game_rows + block.row_bounds() - 1
                    if block.row >= floor:
                        block.row = floor
                        break

        if block.row >= 0:
            if block.rotation != fit_rot:
                block.rotate(grid)
            else:
                pass

            if block.row < fit_row:
                if block.col < fit_col:
                    block.move_right(grid)
                if block.col > fit_col:
                    block.move_left(grid)
            if fit_col_move == 0 and block.row >= 2:
                block.hard_drop(grid)
            elif block.row >= 2:
                if block.row == fit_row_move:
                    if fit_col_move == 1:
                        block.drop_move_left(grid)
                    elif fit_col_move == -1:
                        block.drop_move_right(grid)

        if drop_time == ticks and block.row <= floor and play:
            drop_time = 0
            current_row = block.row
            block.drop_once(grid)
            check_lost += 1

            if check_lost == 1:
                for i in range(check_lost):
                    for j in range(columns):
                        if grid[i][j] != 0:
                            play = False

            if current_row == block.row and play:

                grid_and_lines = Blocks.clear_rows(grid)
                grid = grid_and_lines[0]
                points += Blocks.score(grid_and_lines[1])
                lines += grid_and_lines[1]

                ind += 1
                if ind == 7:
                    ind = 0
                bag = pieces_order(shuffled_blocks, ind)
                shuffled_blocks = bag[0]
                piece = bag[1]
                next_piece = bag[2]

                block = Blocks(7, 0, piece, 0)
                check_lost = 0

                fitness_moves = Blocks.find_fitness(grid, block.num_shape, next_piece)
                fit_col = fitness_moves[0]
                fit_row = fitness_moves[1]
                fit_col_move = fitness_moves[0] - fitness_moves[2]
                fit_row_move = fitness_moves[3]
                fit_rot = fitness_moves[4]
                block.create_block(grid)
                block_used += 1
        if drop_time > ticks and play:
            drop_time = 0

            grid_and_lines = Blocks.clear_rows(grid)
            grid = grid_and_lines[0]
            points += Blocks.score(grid_and_lines[1])

            lines += grid_and_lines[1]

            ind += 1
            if ind == 7:
                ind = 0
            bag = pieces_order(shuffled_blocks, ind)
            shuffled_blocks = bag[0]
            piece = bag[1]
            next_piece = bag[2]

            block = Blocks(7, 0, piece, 0)

            fitness_moves = Blocks.find_fitness(grid, block.num_shape, next_piece)
            fit_col = fitness_moves[0]
            fit_row = fitness_moves[1]
            fit_col_move = fitness_moves[0] - fitness_moves[2]
            fit_row_move = fitness_moves[3]
            fit_rot = fitness_moves[4]

            block.create_block(grid)
            block_used += 1
            check_lost = 0

        if lines - current_cleared_lines == 1:
            one_line += 1
        elif lines - current_cleared_lines == 2:
            two_line += 1

        elif lines - current_cleared_lines == 3:
            three_line += 1

        elif lines - current_cleared_lines == 4:
            four_line += 1
        pygame.display.set_caption("Tetris by Barak Price")

        draw_window(screen, grid, next_piece, lines, points, one_line, two_line, three_line, four_line,block_used)
    pygame.quit()
    quit()


tetris()
