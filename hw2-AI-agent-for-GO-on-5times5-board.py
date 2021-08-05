import numpy as np
import random
from copy import deepcopy
import time

size = 5
def readinput(path = "input.txt"):
	inputt = open(path, 'r')
	lines = inputt.readlines()
	stone = int(lines[0])
	previous_board = []
	current_board = []
	for i in range(1,size+1):
		previous_board.append( [int(s) for s in lines[i] if s.isdigit()] )
	for i in range(size+1, 2*size+1):
		current_board.append( [int(s) for s in lines[i] if s.isdigit()] )

	previous_board = np.array(previous_board)
	current_board = np.array(current_board)

	inputt.close()

	step_file = open("step.txt", 'r')
	line = step_file.readline()
	step = int(line)
	step_file.close()

	return stone, previous_board, current_board, step

def writeoutput(content, path = "output.txt"):
	output = open(path, 'w')
	output.write(content)
	output.close()

def array_to_string(state): #can be optimized to a string
	s = state.flatten()
	state = " ".join(str(n) for n in s)
	return state

def action_to_result(action):
	if action == 25:
		return "PASS"
	else:
		return ( int(action/5), int(action%5) )

def empty_board(board):
	for i in range(0,size):
		for j in range(0,size):
			if board[i][j] != 0:
				return False
	return True

class Q_player:
	def __init__(self, stone):

		self.previous_board = None
		self.current_board = None
		self.stone = stone
		self.step = 0 + stone - 1

		self.type = 'Q_player'

		self.transposition_table = {}

		self.position_value = np.zeros(25)

		self.center = 12
		self.around_center = [6,8,16,18,7,11,13,17]
		self.edge = [1,2,3,5,10,15,9,14,19,21,22,23]
		self.corner = [0,4,20,24]


		for i in range(0,25):
			m,n = action_to_result(i)
			if m == 0 or n == 0 or m == 4 or n == 4: #edges
				self.position_value[i] = 1
			if m == 0 and n == 0 or m == 0 and n == 4 or m == 4 and n == 0 or m == 4 and n == 4:
				self.position_value[i] = 0.5
			if i in [6,8,16,18,7,11,13,17]: #around center
				self.position_value[i] = 1.5
			if i == 12: #center
				self.position_value[i] = 2

	def set_state(self, previous_board, current_board, step = -1):
		self.previous_board = previous_board
		self.current_board = current_board

		if step != -1:
			self.step = step

	def show(self):
		print("stone",self.stone)
		print("previous_board", self.previous_board)
		print("current_board", self.current_board)


    #functions for is valid
	def neighbor(self, i, j):
		neighbor = []
		if i > 0:
			neighbor.append((i-1,j))
		if i < size-1:
			neighbor.append((i+1,j))
		if j > 0:
			neighbor.append((i,j-1))
		if j < size-1:
			neighbor.append((i,j+1))
		return neighbor

	def friend_stones(self, i, j, board):
		neighbor = self.neighbor(i,j)
		friend = []
		for item in neighbor:
			if board[item[0]][item[1]] == board[i][j]:
				friend.append(item)
		return friend

	def friend_bfs(self, i, j, board):
		queue = [(i,j)]
		result = []
		while queue:
			temp = queue.pop(0)
			result.append(temp)
			friend = self.friend_stones(temp[0],temp[1], board)
			for item in friend:
				if item not in queue and item not in result:
					queue.append(item)
		return result

	def check_breath(self, i, j, board):
		all_friends = self.friend_bfs(i, j, board)
		for item in all_friends:
			neighbor = self.neighbor(item[0],item[1])
			for n in neighbor:
				if board[n[0]][n[1]] == 0:
					return True
		return False

	def check_breath_num(self, i, j, board):
		all_friends = self.friend_bfs(i, j, board)
		free_count = 0
		total_count = 0
		visited = {}
		for item in all_friends:
			total_count += 1
			neighbor = self.neighbor(item[0],item[1])
			for n in neighbor:
				if board[n[0]][n[1]] == 0 and (n[0],n[1]) not in visited:
					free_count += 1
					visited[(n[0],n[1])] = 1
		return free_count, total_count

	def simulate(self, stone, board):
		captured = []
		enemy = 3-stone
		for i in range(0,size):
			for j in range(0,size):
				if board[i][j] == enemy and self.check_breath(i,j,board) == False:
					captured.append((i,j))
		for item in captured:
			board[item[0]][item[1]] = 0

		return board

	def is_valid(self, stone, action, previous_board, board): #game rules
		if action < 0 or action > 25:
			return False
		if action == 25:
			return True
		else:
			previous_board = deepcopy(previous_board)
			board = deepcopy(board)

			point = action_to_result(action)
			i = point[0]
			j = point[1]
			if board[i][j] != 0:
				return False
			else:

				captured = []
				for m in range(0,size):
					for n in range(0,size):
						if previous_board[m][n] == stone and board[m][n] != stone:
							captured.append((m,n))

				board[i][j] = stone
				if self.check_breath(i,j,board):
					return True

				board = self.simulate(stone,board)
				if self.check_breath(i,j,board) == False:
					return False
				else: #KO
					same = True
					for m in range(0,size):
						for n in range(0,size):
							if previous_board[m][n] != board[m][n]:
								same = False
					if captured and same:
						return False

				return True
	#is_valid end


	def evaluate(self, previous_board, board, action, step):
		friend_num = enemy_num = friend_captured = enemy_captured = 0
		friend_score = enemy_score = 0
		addition = 0
		win_reward = 0
		position_value = 0
		for i in range(0,size):
			for j in range(0,size):
				if board[i][j] == self.stone:
					friend_num += 1
				if board[i][j] == 3-self.stone:
					enemy_num += 1
				if previous_board[i][j] == self.stone and board[i][j] != self.stone:
					friend_captured += 1
				if previous_board[i][j] == 3-self.stone and board[i][j] != 3-self.stone:
					enemy_captured += 1

		if action != 25:
			i,j = action_to_result(action)
		if action != 25 and board[i][j] == self.stone:
			neighbor = self.neighbor(i,j)
			for item in neighbor:
				m = item[0]
				n = item[1]

				#last move made capturing enemy possible one step away
				if previous_board[m][n] == 3-self.stone and board[m][n] == 3-self.stone:
					free1, total_count1 = self.check_breath_num(m,n,previous_board)
					free2, total_count2 = self.check_breath_num(m,n,board)
					if free1 > 1 and free2 == 1: #only one space around enemy and landing there will kill it
						addition += total_count2**2
					if free1 > 2 and free2 == 2:
						addition += 0.3*total_count2**2

				#last move saved a friend which could have been captured
				if previous_board[m][n] == self.stone and board[m][n] == self.stone:
					free1, total_count1 = self.check_breath_num(m,n,previous_board)
					free2, total_count2 = self.check_breath_num(m,n,board)
					if free1 == 1 and free2 > 1: #saved a friend which could have been captured
						addition += 0.8*total_count1**2


			breath, count = self.check_breath_num(i,j,board) #move caused friends have only 1~2 breath
			if breath <= 1:
				addition -= 0.7*count**2
			if breath <= 2:
				addition -= 0.05*count**2

			position_value = self.position_value[action]

			if i == 0 and board[i+1][j] == 3-self.stone and board[i+2][j] == 3-self.stone and board[i+3][j] == 3-self.stone:
				addition += 0.2
				if j-1 >= 0 and j-1 < 5 and board[i+1][j-1] == self.stone:
					addition += 0.5
				if j+1 >= 0 and j+1 < 5 and board[i+1][j+1] == self.stone:
					addition += 0.5
			if i == 4 and board[i-1][j] == 3-self.stone and board[i-2][j] == 3-self.stone and board[i-3][j] == 3-self.stone:
				addition += 0.2
				if j-1 >= 0 and j-1 < 5 and board[i-1][j-1] == self.stone:
					addition += 0.5
				if j+1 >= 0 and j+1 < 5 and board[i-1][j+1] == self.stone:
					addition += 0.5
			if j == 0 and board[i][j+1] == 3-self.stone and board[i][j+2] == 3-self.stone and board[i][j+3] == 3-self.stone:
				addition += 0.2
				if i-1 >= 0 and i-1 < 5 and board[i-1][j+1] == self.stone:
					addition += 0.5
				if i+1 >= 0 and i+1 < 5 and board[i+1][j+1] == self.stone:
					addition += 0.5
			if j == 4 and board[i][j-1] == 3-self.stone and board[i][j-2] == 3-self.stone and board[i][j-3] == 3-self.stone:
				addition += 0.2
				if i-1 >= 0 and i-1 < 5 and board[i-1][j-1] == self.stone:
					addition += 0.5
				if i+1 >= 0 and i+1 < 5 and board[i+1][j-1] == self.stone:
					addition += 0.5


		if self.step == 22 or self.step == 23: #last round
			if self.stone == 1:
				if friend_num < enemy_num + 2.5:
					win_reward = -1000
				else:
					win_reward = 1000
			if self.stone == 2:
				if friend_num + 2.5 < enemy_num:
					win_reward = -1000
				else:
					win_reward = 1000



		score = friend_num - 2*enemy_num  + 3*enemy_captured**2 - 3*friend_captured**2 + addition + win_reward + position_value

		#if action == 13 or action == 22:
			#print(friend_num, 2*enemy_num, 3*enemy_captured**2, 3*friend_captured**2, addition, win_reward, position_value)

		return score

	def max(self, depth, max_depth, previous_board, board, action, alpha, beta, step):
		if depth <= 0:
			return self.evaluate(previous_board, board, action, step-1), None
		state = array_to_string(board)
		if state in self.transposition_table:
			return self.transposition_table[state]
		else:
			valid_acts = []
			for i in range(0,26):
				if self.is_valid(self.stone, i, previous_board, board):
					valid_acts.append(i)

			if depth == max_depth: #first level
				collection = []
				max_collections = []

			max_score = -np.inf
			for item in valid_acts:
				new_board = deepcopy(board)
				if item != 25:
					i,j = action_to_result(item)
					new_board[i][j] = self.stone
				new_board = self.simulate(self.stone, new_board)
				score1 = self.evaluate(board, new_board, item, step)
				score2, A = self.min(depth-1, max_depth, board, new_board, item, alpha, beta, step+1)
				score = score1 + score2
				if item == 13 or item == 22:
					print(item, score1, score2)
				if score >= beta:
					return score, item
				if score >= max_score:
					if score > max_score:
						opt_action = item
						max_score = score
					if depth == max_depth: #for multiple opt action, choose random
						collection.append((score, item))
				if score > alpha:
					alpha = score


			if depth == max_depth:
				for element in collection:
					if element[0] == max_score:
						max_collections.append(element)

				max_score, opt_action = random.choice(max_collections)

			self.transposition_table[state] = (max_score, opt_action)
			return max_score, opt_action


	def min(self, depth, max_depth, previous_board, board, action, alpha, beta, step):
		if depth <= 0:
			return self.evaluate(previous_board, board, action, step-1), None
		state = array_to_string(board)
		if state in self.transposition_table:
			return self.transposition_table[state]
		else:
			valid_acts = []
			for i in range(0,26):
				if self.is_valid(3-self.stone, i, previous_board, board):
					valid_acts.append(i)

			min_score = np.inf
			for item in valid_acts:
				new_board = deepcopy(board)
				if item != 25:
					i,j = action_to_result(item)
					new_board[i][j] = 3-self.stone
				new_board = self.simulate(3-self.stone, new_board)
				score, A  = self.max(depth-1, max_depth, board, new_board, item, alpha, beta, step+1)
				if score <= alpha:
					return score, item
				if score < min_score:
					opt_action = item
					min_score = score
				if score < beta:
					beta = score

			self.transposition_table[state] = (min_score, opt_action)
			return min_score, opt_action

	def heuristic(self, depth = 2):#greedy

		threshold1 = 14
		threshold2 = 7
		remain = 24 - self.step
		branch_factor = 0
		for i in range(0,26):
			if self.is_valid(3-self.stone, i, self.previous_board, self.current_board):
				branch_factor += 1

		if self.step == 0 + self.stone - 1: #open move of game
			if self.is_valid(self.stone, self.center, self.previous_board, self.current_board):
				return self.center
			return random.choice(self.around_center)

		if branch_factor > threshold1:
			depth = 2
		if branch_factor <= threshold1 and branch_factor > threshold2:
			depth = 4
		if branch_factor <= threshold2:
			depth = 6

		depth = min(depth, remain)

		#depth = 2 #test

		self.transposition_table = {}

		score, action = self.max(depth, depth, self.previous_board, self.current_board, None,  -np.inf, np.inf, self.step)

		return action



	def choose_move(self):

		start = time.time()

		action = self.heuristic()
		self.step += 2

		end = time.time()
		cost = end - start
		print("total time for decision of step ", self.step, ": ", cost)

		return action_to_result(action) #action



if __name__ == "__main__":

	start = time.time()

	stone, previous_board, current_board, step = readinput()
	Q = Q_player(stone)

	if empty_board(previous_board):
		output = open("step.txt", 'w')
		output.write(str(Q.step))
		output.close()
		step = Q.step

	Q.set_state(previous_board, current_board, step)

	'''
	Q.show()
	for i in range(0,26):
		print(i,Q.is_valid(Q.stone,i,Q.previous_board, Q.current_board))

	print(Q.choose_move())
	'''
	
	
	action = Q.choose_move()
	if action != "PASS":
		content = str(action[0])+","+str(action[1])
	else:
		content = action
	writeoutput(content)

	output = open("step.txt", 'w')
	output.write(str(Q.step))
	output.close()

	
	end = time.time()
	cost = end - start
	print("total time for making this step:", cost)